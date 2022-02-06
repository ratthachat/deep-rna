%%writefile ./deep_rna/dataset.py

import numpy as np
import pandas as pd
from deep_rna.spektral.data import Graph, Dataset
from glob import glob
from tqdm.notebook import tqdm

class RNADataset(Dataset):
    """
    An upgraded Spektral's Batch Dataset which fully support multi-length RNA sequences
    This RNADataset class assumes the user processes the data using Tutorial's notebook
    https://www.kaggle.com/ratthachat/preprocessing-deep-learning-input-from-rna-string

    Essential input:
    rna_seq_id_list: list of a unique rna-sequence id, must be the same used in the notebook above
    node_dir: directory name containing node features generated from the notebook
    edge_dir: directory name containing edge features generated from the notebook

    Optional input:
    label_df: a pandas dataframe containing label information for each rna id in arbitrary format
              if label_df is not None, user must implement extract_label() to extract a label from
              seq_id and label_df
    pred_len: length of each rna string to be predicted (i.e. have labels); just in case that
              pred_len < seq_len (as in OpenVaccine) for technical difficulties in bio-experiments
    """

    def __init__(self, 
                rna_seq_id_list,
                node_dir,
                edge_dir,
                label_df=None,
                pred_len=None,
                edge_thresh = 1e-1, 
                num_examples=None,
                **kwargs):
        self.rna_seq_id_list = rna_seq_id_list
        self.node_dir = node_dir
        self.edge_dir = edge_dir
        self.label_df = label_df
        self.edge_thresh = edge_thresh
        self.pred_len = pred_len
        self.num_examples = num_examples # optional: if you want to test only small data
        self.__dict__.update(kwargs)
        super().__init__(**kwargs)
    
    def extract_label(self, seq_id, **kwarg):
        raise NotImplementedError
    
    def read(self):        
        graph_list = []

        for seq_id in tqdm(self.rna_seq_id_list):	
            node_feature = pd.read_csv(self.node_dir+f'{seq_id}_node_features.csv').values # (seq_len, feat)
            
            package_dirs = glob(f"{self.edge_dir}/*/")
            edge_matrix_list = []
            for package_dir in package_dirs:
                edge_matrix = np.load(package_dir + f"/{seq_id}.npy")
                edge_matrix = np.where(edge_matrix > self.edge_thresh, edge_matrix, 0)
                edge_matrix_list.append(edge_matrix)

            if self.pred_len is not None:
                node_feature = node_feature[:self.pred_len]
                edge_matrix_list = [m[:self.pred_len, :self.pred_len] for m in edge_matrix_list]
            
            if self.label_df is not None:
                labels = self.extract_label(seq_id, node_feature=node_feature)
                if self.pred_len:
                    labels = labels[:self.pred_len] # TOFIX, allowing different len
            else:
                labels = None
            
           
            graph_list.append(Graph(x=node_feature, 
                                a=None, 
                                e=np.stack(edge_matrix_list, axis=-1), 
                                y=labels))
            
            if self.num_examples is not None and len(graph_list) == self.num_examples: 
                break
        
        return graph_list

class RNAAutoEncoderDataset(RNADataset):        
    def extract_label(self, seq_id, **kwarg):
        return kwarg['node_feature']
