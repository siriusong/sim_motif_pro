import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  global_add_pool,DeepGCNLayer,global_mean_pool,global_max_pool
from torch_geometric.nn.conv import GraphConv,TransformerConv,GCNConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import  softmax
from torch_scatter import scatter
from torch_geometric.utils import degree
from torch_geometric.nn import aggr
from deeperGCN_variant import GENConv as GENConv2

class GlobalAttentionPool(nn.Module):
    '''
    This is the topology-aware global pooling mentioned in the paper.
    '''
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GraphConv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)
        scores = softmax(x_conv, batch, dim=0)
        gx = global_add_pool(x * scores, batch)

        return gx


class DrugEncoder(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=128, n_layer=10,heads= 4):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers_2 = torch.nn.ModuleList()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(in_dim+edge_in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        layer = GENConv2(hidden_dim,hidden_dim,aggr='maxminmean',
                           num_layers=2, norm='batch')
        self.agr = GCNConv(hidden_dim,hidden_dim)
        self.layers.append(layer)
        self.edge_liner = nn.Linear(6,hidden_dim)
        self.lin = nn.Linear(hidden_dim,hidden_dim)
        self.att = GlobalAttentionPool(hidden_dim*heads)
        self.a = nn.Parameter(torch.zeros(1, hidden_dim*heads, n_layer))
        self.lin_gout = nn.Linear(hidden_dim*heads , hidden_dim*heads)
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_layer))
        self.a_fuse = nn.Linear(hidden_dim*heads*n_layer,hidden_dim)
        self.a_import = nn.Linear(hidden_dim*heads,1)

        self.aggr = aggr.PowerMeanAggregation(learn= True)
        self.aggr1=aggr.MaxAggregation()
        self.aggr2=aggr.MinAggregation()
        self.aggr3=aggr.MeanAggregation()
        self.arrmlp=nn.Linear(3*hidden_dim,hidden_dim)
        self.heads = heads
        self.hidden = hidden_dim
        for i  in range(2,n_layer):
            layer = GENConv2(hidden_dim,hidden_dim,aggr='maxminmean',
                           num_layers=2, norm='batch')
            self.layers.append(layer)

        layer = GENConv2(hidden_dim,hidden_dim,aggr='maxminmean',
                           num_layers=2, norm='batch')
        self.layers.append(layer)

        for i in range(1, 12 + 1):
            conv = GENConv2(hidden_dim, hidden_dim, aggr='maxminmean',
                           num_layers=3, norm='batch')
            norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)
            act = nn.PReLU()

            layer = DeepGCNLayer(conv, norm, act, block='res+',dropout=0.3,
                                 ckpt_grad=i % 3)
            self.layers_2.append(layer)

    def forward(self, data,mm_x,training):

        local_features,x_graph = self.encoder_node_graph(data,training)
        edge_index = data['motif','link','motif'].edge_index


        global_features = mm_x.index_select(dim=0,index=data['motif'].type.squeeze())
        x = self.mlp3(torch.cat([local_features,global_features],dim=1))

        X = None
        new_X = []
        Xscore = []
        for layer in self.layers:
            x = layer(x,edge_index)
            x = F.elu(x,inplace=True)
        return x,new_X

    def encoder_node_graph(self,data,training):

        x = data['atom'].x
        edge_index = data['atom','link','atom'].edge_index
        batch = data['atom'].batch
        X = []
        Xscore = []
        node_to_motif_edge_index = data['atom','link','motif'].edge_index
        x = self.mlp(x)
        x = self.layers_2[0].conv(x, edge_index)

        for layer in self.layers_2[1:]:
            x = layer(x, edge_index)
            x = self.layers_2[0].act(self.layers_2[0].norm(x))
        node_features = x.index_select(0,node_to_motif_edge_index[0])
        index = torch.unique(node_to_motif_edge_index[1])
        motif_x1=self.aggr1(node_features,node_to_motif_edge_index[1])
        motif_x2=self.aggr2(node_features,node_to_motif_edge_index[1])
        motif_x3=self.aggr3(node_features,node_to_motif_edge_index[1])
        motif_x=self.arrmlp(torch.cat([motif_x1,motif_x2,motif_x3],dim=-1))
        x_graph = global_max_pool(x,batch)
        return motif_x,x_graph

class MM_KG_encoder(torch.nn.Module):

    def __init__(self, in_dim, edge_in_dim, hidden_dim=128, n_layer=10,heads= 4):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim+edge_in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        for i in range(1, 12 + 1):
            conv = GENConv2(hidden_dim, hidden_dim, aggr='maxminmean',
                           num_layers=2, norm='layer',edge_dim=hidden_dim)
            norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)
            act = nn.PReLU()

            layer = DeepGCNLayer(conv, norm, act, block='res+',dropout=0.3,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.mlp(x)
        for layer in self.layers[1:]:
            x = layer(x, edge_index)
            x = self.layers[0].act(self.layers[0].norm(x))
        return x

class SimMotifPro(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=128, n_layer=3):
        super(SimMotifPro, self).__init__()
        self.mm_encoder = MM_KG_encoder(in_dim,edge_in_dim,hidden_dim, n_layer=n_layer)
        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_layer=n_layer)
        self.h_gpool = GlobalAttentionPool(hidden_dim)
        self.t_gpool = GlobalAttentionPool(hidden_dim)
        self.n_layer = n_layer
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim  * 2, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim ),

        )
        self.realation_mlp =  nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),

        )
        self.drug_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim ),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),

        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)

        )
        self.drug_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        
        )
        self.realtion_embed = nn.Sequential(
            nn.Linear(hidden_dim , hidden_dim ),
            nn.PReLU(),
            nn.Linear(hidden_dim , hidden_dim),
        
        )
        self.drug_fuse_2 = nn.Sequential(
            nn.Linear(hidden_dim , hidden_dim ),
            nn.PReLU(),
            nn.Linear(hidden_dim , hidden_dim),
            nn.LayerNorm(hidden_dim),
        
        )

        self.drugs_logit = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        self.Layer_fuse = nn.Sequential(
            nn.Linear(hidden_dim * n_layer * 4, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_fuse =  nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.a = nn.Parameter(torch.zeros(hidden_dim))
        self.rmodule = nn.Embedding(1318, hidden_dim)

        self.w_j = nn.Linear(hidden_dim, hidden_dim)
        self.w_i = nn.Linear(hidden_dim, hidden_dim)

        self.prj_j = nn.Linear(hidden_dim, hidden_dim)
        self.prj_i = nn.Linear(hidden_dim, hidden_dim)

    def Motif_Ranker(self,drug_a,drug_b,batch,training,relation):
        a = degree(batch, dtype=batch.dtype)
        drug_align = drug_b.repeat_interleave(a, dim=0)
        relation_align = relation.repeat_interleave(a, dim=0)
        fuse_feature = self.drug_fuse(torch.cat([drug_a, drug_align], dim=1))
        fuse_feature = self.realation_mlp(torch.cat([fuse_feature,relation_align],dim=1))
        log_logit = self.drug_fuse_2(fuse_feature)
        log_sig = self.concrete_sample(log_logit,training=training)
        x = log_sig * drug_a
        return x

    def concrete_sample(self,att_log_logit, temp=0.2, training=True):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)

            output = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            output = (att_log_logit).sigmoid()
        return output

    def realation_add(self,x,realations,batch):
        a = degree(batch, dtype=batch.dtype)
        realations_aligh = realations.repeat_interleave(a, dim=0)
        b = torch.cat([x,realations_aligh],dim=1)
        r_agu = self.realation_mlp(b)
        return x+r_agu

    def forward(self, triples,trainging):
        h_data, t_data, rels, mm_graph = triples
        h_data_edge_index= h_data['motif','link','motif'].edge_index
        h_data_batch = h_data['motif'].batch
        t_data_edge_index = t_data['motif','link','motif'].edge_index
        t_data_batch = t_data['motif'].batch
        rfeat = self.rmodule(rels)
        rfeat = self.realtion_embed(rfeat)
        mm_graph_x = self.mm_encoder(mm_graph)
        x_h,h_graph = self.drug_encoder(h_data,mm_graph_x,trainging)
        x_t,t_graph = self.drug_encoder(t_data,mm_graph_x,trainging)
        x_h_g = global_add_pool(x_h,h_data_batch)
        x_t_g = global_add_pool(x_t,t_data_batch)
        x_h_i = self.Motif_Ranker(x_h,x_t_g,batch=h_data_batch,training=trainging,relation = rfeat)
        x_t_i = self.Motif_Ranker(x_t,x_h_g,batch=t_data_batch,training=trainging,relation = rfeat)
        h_final = global_add_pool(x_h_i,h_data_batch)
        t_final = global_add_pool(x_t_i,t_data_batch)
        pair = torch.cat([h_final, t_final], dim=-1)
        logit = (self.lin(pair) * rfeat).sum(-1)

        return logit

    def encoder_motif(self,data):
        x = self.motif_encoder(data)
        return x