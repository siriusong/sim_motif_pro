import networkx as nx
import torch
from tqdm import tqdm
import math
import numpy as np
import copy
import gc
import pickle
from rdkit import Chem
from data_preprocessing_warm_start import CustomData
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData,Data
from torch_geometric.utils import from_scipy_sparse_matrix

Edgetype ={
    1:1,
    0:0,
    2:2,
    3:3,
    1.5:4
}
class MotifData(Data):
    '''
    Since we have converted the node graph to the line graph, we should specify the increase of the index as well.
    '''
    # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
    # Replace with "def __inc__(self, key, value, *args, **kwargs)"
    def __inc__(self, key, value, *args, **kwargs):
        return super().__inc__(key, value, *args, **kwargs)
class GraphMessage(object):
    def __init__(self, g, key,  node_features=None,edge_features = None,node_types = None,edge_index = None):
        self.key = key
        self.g = g
        self.neighbors = []
        self.node_types = node_types
        self.node_features = node_features
        self.edge_features = edge_features
        self.edge_index = edge_index


class GenMotifGraph(object):
    def __init__(self, data):
        self.data = data
        self.vocab = {}
        self.vocab_reverse = {}
        self.whole_node_count = {}
        self.weight_vocab = {}
        self.node_count = {}
        self.edge_count = {}
        self.motif_nodes_features = {}
        self.motif_edges_features = {}
        self.motifs = {}



    def gen_Motif(self,dir):
        node_Motif_datas = {}
        Motif_datas = {}
        g_lists = {}
        mm_g = nx.Graph()
        mm_edge_index = []
        for i in tqdm(range(len(self.data)), desc='Gen Components', unit='graph'):
            data = self.data[i]
            g = data.g

            h_g = nx.Graph()
            node_labels = data.node_types
            clique_list = []
            mcb = nx.cycle_basis(g)
            mcb_tuple = [tuple(ele) for ele in mcb]
            motif_ids = {}
            edges = []
            for e in g.edges():
                count = 0
                for c in mcb_tuple:
                    if e[0] in set(c) and e[1] in set(c):
                        count += 1
                        break
                if count == 0:
                    edges.append(e)
            edges = list(set(edges))

            for e in edges:

                weight = g.get_edge_data(e[0], e[1])['weight']
                edge = ((node_labels[e[0]],node_labels[e[1]]), weight)
                clique_id = self.add_to_vocab(edge)
                motif_ids[(e[0],e[1])] = clique_id 
                self.add_motif_features(data,clique_id,[e[0],e[1]],g)
                clique_list.append(clique_id)
                if clique_id not in self.whole_node_count:
                    self.whole_node_count[clique_id] = 1
                else:
                    self.whole_node_count[clique_id] += 1

            for m in mcb_tuple:
                weight = tuple(self.find_ring_weights(m, g))
                ring = []
                for i in range(len(m)):
                    ring.append(node_labels[m[i]])
                cycle = (tuple(ring), weight)
                cycle_id = self.add_to_vocab(cycle)
                motif_ids[m] = cycle_id
                self.add_motif_features(data,cycle_id,m,g)
                clique_list.append(cycle_id)
                if cycle_id not in self.whole_node_count:
                    self.whole_node_count[cycle_id] = 1
                else:
                    self.whole_node_count[cycle_id] += 1
            c_list=tuple(set(clique_list))
            for e in c_list:
                if e not in self.node_count:
                    self.node_count[e] = 1
                else:
                    self.node_count[e]+=1

            for e in range(len(edges)):
                for i in range(e+1, len(edges)):
                    for j in edges[e]:
                        if j in edges[i]:
                            weight = g.get_edge_data(edges[e][0], edges[e][1])['weight']
                            edge = ((node_labels[edges[e][0]],node_labels[edges[e][1]]), weight)
                            weight_i = g.get_edge_data(edges[i][0], edges[i][1])['weight']
                            edge_i = ((node_labels[edges[i][0]], node_labels[edges[i][1]]), weight_i)
                            final_edge = tuple(sorted((self.add_to_vocab(edge), self.add_to_vocab(edge_i))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1 
                            else:
                                self.edge_count[final_edge] += 1
                                
            for m in range(len(mcb_tuple)):
                for i in range(m+1, len(mcb_tuple)):
                    for j in mcb_tuple[m]:
                        if j in mcb_tuple[i]:
                            weight = tuple(self.find_ring_weights(mcb_tuple[m], g))
                            ring = []
                            for t in range(len(mcb_tuple[m])):
                                ring.append(node_labels[mcb_tuple[m][t]])
                            cycle = (tuple(ring), weight)

                            weight_i = tuple(self.find_ring_weights(mcb_tuple[i], g))
                            ring_i = []
                            for t in range(len(mcb_tuple[i])):
                                ring_i.append(node_labels[mcb_tuple[i][t]])
                            cycle_i = (tuple(ring_i), weight_i)

                            final_edge = tuple(sorted((self.add_to_vocab(cycle), self.add_to_vocab(cycle_i))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1

            for e in range(len(edges)):
                for m in range(len(mcb_tuple)):
                    for i in edges[e]:
                        if i in mcb_tuple[m]:
                            weight_e = g.get_edge_data(edges[e][0], edges[e][1])['weight']
                            edge_e = ((node_labels[edges[e][0]], node_labels[edges[e][1]]), weight_e)
                            weight_m = tuple(self.find_ring_weights(mcb_tuple[m], g))
                            ring_m = []
                            for t in range(len(mcb_tuple[m])):
                                ring_m.append(node_labels[mcb_tuple[m][t]])
                            cycle_m = (tuple(ring_m), weight_m)

                            final_edge = tuple(sorted((self.add_to_vocab(edge_e), self.add_to_vocab(cycle_m))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1
            all_motifs = edges + mcb_tuple 
            node_to_motif_edge_index,node_graph_edge_index,node_graph_edge_attr,line_graph_edge_index = self.create_motif_node_graph(g,all_motifs)
            motif_node_features = []
            motif_edge_features = []
            motifs = []
            for id,motif in enumerate(all_motifs):
                if len(motif)>2:
                    weight = tuple(self.find_ring_weights(motif, g))
                    ring = []
                    for i in range(len(motif)):
                        ring.append(node_labels[motif[i]])
                    cycle = (tuple(ring), weight)
                    motif_id = self.add_to_vocab(cycle)
                elif len(motif) == 2:
                    weight = g.get_edge_data(motif[0], motif[1])['weight']
                    edge = ((node_labels[motif[0]], node_labels[motif[1]]), weight)
                    motif_id = self.add_to_vocab(edge)
                else:
                    edge = ((node_labels[motif[0]],), (0,))
                    motif_id = self.add_to_vocab(edge)

                h_g.add_node(id, motif_type=motif_id)
                motif_edge_features.append(torch.sum(self.motif_edges_features[motif_id],dim=0))
                motif_node_features.append(torch.sum(self.motif_nodes_features[motif_id],dim=0))

                motifs.append(torch.tensor([motif_id]))
            # generate mm_KG
            for ii in range(len(all_motifs)):
                for jj in range(ii + 1, len(all_motifs)):
                    motif_a = all_motifs[ii]
                    motif_b = all_motifs[jj]
                    motif_ids = []
                    for motif in [motif_a,motif_b]:
                        if len(motif) > 2: 
                            weight = tuple(self.find_ring_weights(motif, g))
                            ring = []
                            for i in range(len(motif)):
                                ring.append(node_labels[motif[i]])
                            cycle = (tuple(ring), weight)
                            motif_id = self.add_to_vocab(cycle)
                        elif len(motif) == 2: 
                            weight = g.get_edge_data(motif[0], motif[1])['weight']
                            edge = ((node_labels[motif[0]], node_labels[motif[1]]), weight)
                            motif_id = self.add_to_vocab(edge)
                        else:
                            edge = ((node_labels[motif[0]],), (0,))
                            motif_id = self.add_to_vocab(edge)
                        motif_ids.append(motif_id)
                    if motif_ids not in mm_edge_index:

                        mm_edge_index.append(motif_ids)
                        re_motif_ids = list(reversed(motif_ids))
                        if re_motif_ids != motif_ids:
                            mm_edge_index.append(re_motif_ids)

            num_edges = h_g.number_of_edges()
            if num_edges == 0:
                h_g.add_edge(0,0)
            motif_features = torch.cat((torch.stack(motif_node_features),torch.stack(motif_edge_features)),dim=1)
            motifs = torch.stack(motifs)
            adj_matrix = nx.to_scipy_sparse_matrix(h_g,weight = 'weight')
            edge_index,edge_type = from_scipy_sparse_matrix(adj_matrix)
            h_data = HeteroData()
            h_data['atom'].x = data.node_features
            h_data['motif'].x = motif_features
            h_data['motif'].type = motifs
            h_data['atom','link','atom'].edge_index = node_graph_edge_index
            h_data['motif','link','motif'].edge_index = edge_index
            h_data['atom','link','motif'].edge_index = node_to_motif_edge_index
            h_data['atom', 'link', 'atom'].edge_attr = node_graph_edge_attr
            h_data['atom','line','atom'].line_graph_edge_index = line_graph_edge_index
            node_Motif_datas[data.key] = h_data

            g_lists[data.key] = h_g

            Motif_datas[data.key] = MotifData(edge_index=edge_index,x= motif_features)

        mm_g_x = []
        for id in range(len(self.motif_nodes_features)):
            mm_g_x.append(torch.cat((torch.sum(self.motif_nodes_features[id],dim=0),torch.sum(self.motif_edges_features[id],dim=0))))
        mm_g_x = torch.stack(mm_g_x)
        mm_edge_index = torch.tensor(mm_edge_index).T

        kg_edge_index=mm_edge_index.T

        kg_edge_weight=torch.zeros(1,kg_edge_index.size(0))
        special_edges_set=set(list(self.edge_count.keys()))
        corresponding_edge_indices = []

        for idx, edge_index in enumerate(kg_edge_index.numpy()):
            if (tuple(edge_index) in special_edges_set) or (tuple(reversed(edge_index)) in special_edges_set):
                corresponding_edge_indices.append((idx, tuple(edge_index)))

        for index, edge in corresponding_edge_indices:
            if edge[0]>edge[1]:
                edge=tuple(reversed(edge))
            kg_edge_weight[0,index]=math.exp(math.log(self.edge_count[edge]/math.sqrt(self.whole_node_count[edge[0]]*self.whole_node_count[edge[1]])))

        mm_edge_attr=kg_edge_weight.T
        ####==============================================================####
        mm_graph = Data(x = mm_g_x,edge_index=mm_edge_index,edge_weight=mm_edge_attr)
        

        self.save_data(dir,'MM_graph.pkl',mm_graph)
        self.save_data(dir,'motif_graph_data.pkl',Motif_datas)
        self.save_data(dir,'node_motif_graph.pkl',node_Motif_datas)
        self.save_data(dir,'motif_vocab.pkl',self.vocab)
        self.save_data(dir,'motif_vocab_reverse.pkl',self.vocab_reverse)
        self.save_data(dir,'motif_graph.pkl',g_lists)
        self.save_data(dir,'motifs.pkl',self.motifs)
    def save_data(self,dir,name,data):
        filename = dir + '/' + name
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f'\nData saved as {name}!')
    def add_to_vocab(self, clique):
        c = copy.deepcopy(clique[0])
        weight = copy.deepcopy(clique[1])
        for i in range(len(c)):
            if (c, weight) in self.vocab:

                return self.vocab[(c, weight)]
            else:
                c = self.shift_right(c)
                weight = self.shift_right(weight)
        self.vocab[(c, weight)] = len(list(self.vocab.keys()))
        if self.vocab[(c, weight)] not in self.vocab_reverse.keys():
            self.vocab_reverse[self.vocab[(c, weight)]] = (c, weight)
        return self.vocab[(c, weight)]

    def create_motif_node_graph(self,g,all_motifs):

        node_to_motif_edge_index = []

        edge_list = [edge for edge in g.edges()] 
        edge_list_ = torch.LongTensor(edge_list)
        line_graph_edge_index = torch.LongTensor([])
        if edge_list_.nelement() != 0:
            conn = (edge_list_[:, 1].unsqueeze(1) == edge_list_[:, 0].unsqueeze(0)) & (
                        edge_list_[:, 0].unsqueeze(1) != edge_list_[:, 1].unsqueeze(0))
            line_graph_edge_index = conn.nonzero(as_tuple=False).T
        node_graph_edge_index = torch.tensor(edge_list,dtype=int).T

        node_graph_edge_attr = torch.stack([g.get_edge_data(edge[0],edge[1])['feature'].float() for edge in g.edges()])
        for id,data in enumerate(all_motifs):
            for node in data:
                node_to_motif_edge_index.append([node,id])
        node_to_motif_edge_index = torch.tensor(node_to_motif_edge_index,dtype=int).T
        return node_to_motif_edge_index,node_graph_edge_index,node_graph_edge_attr,line_graph_edge_index

    def add_motif_features(self,data,id,atoms,g):
        if id in self.motif_edges_features.keys():
            return
        edge_index = data.edge_index
        all_edge_features = data.edge_features
        all_nodes_features = data.node_features
        motif_edge_features = []
        motif_nodes_features = []
        weights = []
        edge_index_ = []
        node_count = {}
        i= 0
        for node in atoms:
            motif_nodes_features.append(all_nodes_features[node,:])
            node_count[node] = i
            i += 1
        for i in range(len(atoms)):
            for j in range(i+1,len(atoms)):
                if (atoms[i],atoms[j]) in edge_index:
                    index = edge_index.index((atoms[i],atoms[j]))
                    weight = Edgetype[g.get_edge_data(atoms[i], atoms[j])['weight']]
                    weights.append(weight)
                    weights.append(weight)
                    edge_index_.append([node_count[atoms[i]],node_count[atoms[j]]])
                    edge_index_.append([node_count[atoms[j]],node_count[atoms[i]]])
                    motif_edge_features.append(all_edge_features[index,:])
                    motif_edge_features.append(all_edge_features[index,:])
        if len(motif_edge_features) == 0:
            motif_edge_features.append(torch.tensor([0,0,0,0,0,0]))
            motif_edge_features.append(torch.tensor([0,0,0,0,0,0]))
            edge_index_.append([0, 0])
            edge_index_.append([0, 0])
            weights.append(0)
            weights.append(0)
        motif_edge_features = torch.stack(motif_edge_features)
        motif_nodes_features = torch.stack(motif_nodes_features)
        self.motif_nodes_features[id] = motif_nodes_features
        self.motif_edges_features[id] = motif_edge_features
        edge_index_ = torch.tensor(edge_index_,dtype=int).T
        weights = torch.tensor(weights)
        self.motifs [id] = Data(x=motif_nodes_features,edge_index=edge_index_,edge_attr=motif_edge_features,edge_types= weights)


    @staticmethod
    def shift_right(l):
        if type(l) == int or type(l) == float:
            return l
        elif type(l) == tuple:
            l = list(l)
            return tuple([l[-1]] + l[:-1])
        elif type(l) == list:
            return tuple([l[-1]] + l[:-1])
        else:
            print('ERROR!')

    @staticmethod
    def find_ring_weights(ring, g):
        weight_list = []
        for i in range(len(ring)-1):
            weight = g.get_edge_data(ring[i], ring[i+1])['weight']
            weight_list.append(weight)
        weight = g.get_edge_data(ring[-1], ring[0])['weight']
        weight_list.append(weight)
        return weight_list

if __name__ == '__main__':
    data_dir = 'drugbank/data/preprocessed/drugbank/drug_data.pkl'
    with open(data_dir, 'rb') as f:
        datas = pickle.load(f)
    g_list = []
    for key in datas:
        data = datas[key]
        g = nx.Graph()
        edge_index = data.edge_index.numpy()
        mol_graph = data.mol_graph
        if len(edge_index)>0:
            edge_index_ = [(edge_index[0,i],edge_index[1,i]) for i in range(edge_index.shape[1])]
            edge_index_tuple = [(edge_index[0,i],edge_index[1,i],{'weight':mol_graph.GetBondBetweenAtoms(int(edge_index[0,i]),int(edge_index[1,i]),).GetBondTypeAsDouble(),
            'feature':data.edge_attr[edge_index_.index((edge_index[0,i],edge_index[1,i])),:]                                                      }) for i in range(edge_index.shape[1])]

        else:
            edge_index_tuple = [(0,0,{'weight':0,'feature':torch.tensor([0,0,0,0,0,0])})]
            edge_types = [0]
            edge_index_ = [(0,0)]

        g.add_edges_from(edge_index_tuple)
        atom_types = {atom.GetIdx():atom.GetSymbol() for atom in data.mol_graph.GetAtoms()}
        g_list.append(GraphMessage(g,key,data.x,data.edge_attr,atom_types,edge_index_))
    tools = GenMotifGraph(g_list)
    tools.gen_Motif('drugbank/data/preprocessed/drugbank/')