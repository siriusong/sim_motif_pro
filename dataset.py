import os
import torch
from torch_geometric.data import Batch, Data,HeteroData
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from data_preprocessing_warm_start import CustomData
from ops import MotifData


def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

class DrugDataset(Dataset):
    def __init__(self, data_df, drug_graph,node_motif_graph):
        self.data_df = data_df
        self.drug_graph = drug_graph
        self.node_motif_graph = node_motif_graph
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.data_df.iloc[index]

    def collate_fn(self, batch):
        head_list = []
        tail_list = []
        label_list = []
        rel_list = []

        for row in batch:
            Drug1_ID, Drug2_ID, Y, Neg_samples = row['Drug1_ID'], row['Drug2_ID'], row['Y'], row['Neg samples']
            Neg_ID, Ntype = Neg_samples.split('$')
            h_graph = self.drug_graph.get(Drug1_ID)

            t_graph = self.drug_graph.get(Drug2_ID)

            n_graph = self.drug_graph.get(Neg_ID)
            pos_pair_h = h_graph

            pos_pair_t = t_graph

            if Ntype == 'h':
                neg_pair_h = n_graph
                neg_pair_t = t_graph

            else:
                neg_pair_h = h_graph

                neg_pair_t = n_graph  

            head_list.append(pos_pair_h)
            head_list.append(neg_pair_h)

            tail_list.append(pos_pair_t)
            tail_list.append(neg_pair_t)
            rel_list.append(torch.LongTensor([Y]))
            rel_list.append(torch.LongTensor([Y]))

            label_list.append(torch.FloatTensor([1]))
            label_list.append(torch.FloatTensor([0]))

        head_pairs = Batch.from_data_list(head_list, follow_batch=['edge_index'])
        tail_pairs = Batch.from_data_list(tail_list, follow_batch=['edge_index'])
        rel = torch.cat(rel_list, dim=0)
        label = torch.cat(label_list, dim=0)
        return head_pairs, tail_pairs, rel, label

class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


def split_train_valid(data_df, fold, val_ratio=0.0001):
        cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
        train_index, val_index = next(iter(cv_split.split(X=range(len(data_df)), y = data_df['Y'])))

        train_df = data_df.iloc[train_index]
        val_df = data_df.iloc[val_index]

        return train_df, val_df

def load_ddi_dataset(root, batch_size, fold=0):
    drug_graph = read_pickle(os.path.join(root, 'node_motif_graph.pkl'))
    node_motif_graph = read_pickle(os.path.join(root, 'node_motif_graph.pkl'))
    motif_graph = read_pickle(os.path.join(root,'motifs.pkl'))
    mm_graph = read_pickle(os.path.join(root,'MM_graph.pkl'))
    motifs = []
    for data in motif_graph.values():
        motifs.append(data)
    motifs = Batch.from_data_list(motifs)

    ## warm start
    train_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triples_train_fold{fold}.csv'))
    test_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triples_test_fold{fold}.csv'))
    
    ## cold start
    # train_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triples-fold0-train.csv'))
    # val_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triples-fold0-s1.csv'))
    # test_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triples-fold0-s2.csv'))

    train_set = DrugDataset(train_df, drug_graph,node_motif_graph)
    # val_set = DrugDataset(val_df, drug_graph,node_motif_graph)
    test_set = DrugDataset(test_df, drug_graph,node_motif_graph)

    train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    # val_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    print("Number of samples in the train set: ", len(train_set))
    # print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))
        
    return train_loader, test_loader, test_loader,motifs,mm_graph

if __name__ == "__main__":

    train_loader, val_loader, test_loader,motifs = load_ddi_dataset(root='data/preprocessed/drugbank', batch_size=256, fold=0)