from typing import Optional, Callable
import warnings
import torch
import torch.nn.functional as F
from torch.nn.modules.sparse import Embedding
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch import nn, Tensor
from torch_geometric.nn import global_max_pool, global_mean_pool, global_sort_pool

import pdb

class CustomMessagePassing(MessagePassing):

    def __init__(self, aggr: Optional[str] = "maxminmean", embed_dim: Optional[int] = None):
        if aggr in ['maxminmean']:
            super().__init__(aggr=None)
            self.aggr = aggr
            assert embed_dim is not None
            self.aggrmlp = nn.Linear(3 * embed_dim, embed_dim)
        else:
            super().__init__(aggr=aggr)

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor],
                  dim_size: Optional[int]) -> Tensor:
        if self.aggr in ['maxminmean']:
            inputs_fp32 = inputs.float()
            input_max = scatter(inputs_fp32,
                                index,
                                dim=self.node_dim,
                                dim_size=dim_size,
                                reduce='max')
            input_min = scatter(inputs_fp32,
                                index,
                                dim=self.node_dim,
                                dim_size=dim_size,
                                reduce='min')
            input_mean = scatter(inputs_fp32,
                                 index,
                                 dim=self.node_dim,
                                 dim_size=dim_size,
                                 reduce='mean')
            aggr_out = torch.cat([input_max, input_min, input_mean], dim=-1).type_as(inputs)
            aggr_out = self.aggrmlp(aggr_out)
            return aggr_out
        else:
            return super().aggregate(inputs, index, ptr, dim_size)

from typing import List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    InstanceNorm1d,
    LayerNorm,
    ReLU,
    Sequential,
)

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.norm import MessageNorm
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import is_torch_sparse_tensor, to_edge_index


class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None,
                 bias: bool = True, dropout: float = 0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias=bias))

            if i < len(channels) - 1:
                if norm and norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm and norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm and norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                elif norm:
                    raise NotImplementedError(
                        f'Normalization layer "{norm}" not supported.')
                m.append(ReLU())
                m.append(Dropout(dropout))

        super().__init__(*m)


class GENConv(CustomMessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = 'softmax',
        t: float = 1.0,
        learn_t: bool = False,
        p: float = 1.0,
        learn_p: bool = False,
        msg_norm: bool = False,
        learn_msg_scale: bool = False,
        norm: str = 'batch',
        num_layers: int = 2,
        expansion: int = 2,
        eps: float = 1e-7,
        bias: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):

        # Backward compatibility:
        semi_grad = True if aggr == 'softmax_sg' else False
        aggr = 'softmax' if aggr == 'softmax_sg' else aggr
        aggr = 'powermean' if aggr == 'power' else aggr

        # Override args of aggregator if `aggr_kwargs` is specified
        if 'aggr_kwargs' not in kwargs:
            if aggr == 'softmax':
                kwargs['aggr_kwargs'] = dict(t=t, learn=learn_t,
                                             semi_grad=semi_grad)
            elif aggr == 'powermean':
                kwargs['aggr_kwargs'] = dict(p=p, learn=learn_p)

        super().__init__(aggr=aggr, embed_dim=out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = eps

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if in_channels[0] != out_channels:
            self.lin_src = Linear(in_channels[0], out_channels, bias=bias)

        if edge_dim is not None and edge_dim != out_channels:
            self.lin_edge = Linear(edge_dim, out_channels, bias=bias)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(out_channels)
        else:
            aggr_out_channels = out_channels

        if aggr_out_channels != out_channels:
            self.lin_aggr_out = Linear(aggr_out_channels, out_channels,
                                       bias=bias)

        if in_channels[1] != out_channels:
            self.lin_dst = Linear(in_channels[1], out_channels, bias=bias)

        channels = [out_channels]
        for i in range(num_layers - 1):
            channels.append(out_channels * expansion)
        channels.append(out_channels)
        self.mlp = MLP(channels, norm=norm, bias=bias)

        if msg_norm:
            self.msg_norm = MessageNorm(learn_msg_scale)

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp)
        if hasattr(self, 'msg_norm'):
            self.msg_norm.reset_parameters()
        if hasattr(self, 'lin_src'):
            self.lin_src.reset_parameters()
        if hasattr(self, 'lin_edge'):
            self.lin_edge.reset_parameters()
        if hasattr(self, 'lin_aggr_out'):
            self.lin_aggr_out.reset_parameters()
        if hasattr(self, 'lin_dst'):
            self.lin_dst.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if hasattr(self, 'lin_src'):
            x = (self.lin_src(x[0]), x[1])

        if isinstance(edge_index, SparseTensor):
            edge_attr = edge_index.storage.value()
        elif is_torch_sparse_tensor(edge_index):
            _, value = to_edge_index(edge_index)
            if value.dim() > 1 or not value.all():
                edge_attr = value

        if edge_attr is not None and hasattr(self, 'lin_edge'):
            edge_attr = self.lin_edge(edge_attr)

        # Node and edge feature dimensionalites need to match.
        if edge_attr is not None:
            edge_attr=edge_attr.repeat(1,x[0].size(-1))
            assert x[0].size(-1) == edge_attr.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        if hasattr(self, 'lin_aggr_out'):
            out = self.lin_aggr_out(out)

        if hasattr(self, 'msg_norm'):
            h = x[1] if x[1] is not None else x[0]
            assert h is not None
            out = self.msg_norm(h, out)

        x_dst = x[1]
        if x_dst is not None:
            if hasattr(self, 'lin_dst'):
                x_dst = self.lin_dst(x_dst)
            out = out + x_dst

        return self.mlp(out)

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        msg = x_j if edge_attr is None else x_j + edge_attr
        return msg.relu() + self.eps

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')