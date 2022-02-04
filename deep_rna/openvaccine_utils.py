'''
Utility Functions which are frequently used in Kaggle's OpenVaccine mRNA data format
https://www.kaggle.com/c/stanford-covid-vaccine
'''

import pandas as pd
import numpy as np

def pandas_list_to_array(df):
  """
  Input: dataframe of shape (num_examples, num_cols), each cell contains a list of length "seq_len"
  Return: np.array of shape (num_examples, seq_len, num_cols)
  """
    
  return np.transpose(
          np.array(df.values.tolist()),
          (0, 2, 1)
         )
  
def extract_seq_pseudo_label_from_submission(submission_df, rna_seq_id, pred_cols=None):
  """
  Each Kaggle's OpenVaccine submission csv file can be used to make a pseudo label.
  OpenVaccine submission format is explained here: 
  https://www.kaggle.com/c/stanford-covid-vaccine/overview/evaluation
  
  Example of high-quality submission is here:
  https://www.kaggle.com/group16/covid-19-mrna-4th-place-solution/data?select=submission_lstm_lstm.csv
  
  This function reads a submission dataframe and the specified rna sequence id 
  to extract the pseudo-label of that sequence in the shape (seq_len, n_labels)
  
  Example:
  submission_df = pd.read_csv('submission.csv')
  pseudo_label = extract_seq_pseudo_label_from_submission(submission_df, "id_00073f8be")
  """
  
  if pred_cols is None:
    # be careful about the order of these labels
    pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
  
  sel_row = submission_df.id_seqpos.str.startswith(rna_seq_id)
  sub_sel_id = submission_df[sel_row]
    
  sub_sel_id[['dummy','seq_name', 'seq_pos']] = sub_sel_id.id_seqpos.str.split('_',expand=True)
  sub_sel_id['seq_pos'] = sub_sel_id['seq_pos'].astype(int)
  
  return sub_sel_id.sort_values(by='seq_pos')[self.pred_cols].values
