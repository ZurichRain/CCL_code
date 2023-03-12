
import sys
import os

from transformers import BertTokenizer,AutoTokenizer

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import json
from torch.utils.data import Dataset

from sklearn.metrics import recall_score,precision_score,f1_score,confusion_matrix,roc_curve,accuracy_score

model_path = 'hfl/chinese-roberta-wwm-ext'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Task2TypeCDataset(Dataset):
    def __init__(self, seqlis, labellis=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.is_train =  labellis is not None
        self.dataset = self.preprocess(seqlis,labellis)
    
    def ref_label(self,seqlis,tokenizer,label_lis):
        if self.is_train:
            seq_lab_lis=[]
            seqtokidslis=[]
            for idx,lab in enumerate(label_lis):
                # print(lab)
                seq=seqlis[idx]
                curseqtok=tokenizer.tokenize(seq)
                
                char2tok_span=self.get_char2tok_spanlis_one_seq(seq,tokenizer)
                num_tok = len(tokenizer.convert_tokens_to_ids(curseqtok))
                
                for tr in lab:
                    seq_lab=[0 for _ in range(num_tok)]
                    Sf = None
                    Pf = None
                    Ef = None
                    for f in tr["fragments"]:
                        if f['role']=='S':
                            Sf=f
                        if f['role']=='P':
                            Pf=f
                        if f['role']=='E':
                            Ef=f
                    if Sf != None:
                        for text1_idx in Sf["idxes"]:
                            seq_lab[char2tok_span[text1_idx][0]]=1
                    if Pf != None:
                        for text2_idx in Pf["idxes"]:
                            seq_lab[char2tok_span[text2_idx][0]]=2
                    if Ef != None:
                        for text2_idx in Ef["idxes"]:
                            seq_lab[char2tok_span[text2_idx][0]]=3
                    seqtokidslis.append(tokenizer.convert_tokens_to_ids(curseqtok))
                    seq_lab_lis.append(seq_lab)

            return seqtokidslis,seq_lab_lis
        seqtokidslis=[]
        for seq in seqlis:
            curseqtok=tokenizer.tokenize(seq)
            seqtokidslis.append(tokenizer.convert_tokens_to_ids(curseqtok))
        return seqtokidslis
    def preprocess(self,seqlis,labellis):
        if self.is_train:
            seqtokidslis,seq_lab_lis=self.ref_label(seqlis,self.tokenizer,labellis)
            data=[]
            for seq,lab in zip(seqtokidslis,seq_lab_lis):
                data.append((seq,lab))
            return data
        seqtokidslis = self.ref_label(seqlis,self.tokenizer,labellis)
        data=[]
        for seq in seqtokidslis:
            data.append((seq,))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        seq = self.dataset[idx][0]
        if self.is_train:
            lab = self.dataset[idx][1]
            return [seq,lab]
        return [seq]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)
    def get_seq_tok(self,seqlis,tokenizer):
        curseqtoklis=[]
        curseqtokidslis=[]
        doc_max_seq_len=0
        for seq in seqlis:
            curseqtok=tokenizer.tokenize(seq)
            doc_max_seq_len=max(doc_max_seq_len,len(curseqtok))
            curseqtoklis.append(curseqtok)
            curseqtokidslis.append(tokenizer.convert_tokens_to_ids(curseqtok))

        return curseqtoklis,curseqtokidslis
    
    def get_char2tok_spanlis_one_seq(self,seq,tokenizer):
        token_span = tokenizer.encode_plus(seq, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        char_num = None
        for tok_ind in range(len(token_span) - 1, -1, -1):
            if token_span[tok_ind][1] != 0:
                char_num = token_span[tok_ind][1]
                break
        char2tok_span = [[-1, -1] for _ in range(char_num)] 
        for tok_ind, char_sp in enumerate(token_span):
            for char_ind in range(char_sp[0], char_sp[1]):
                tok_sp = char2tok_span[char_ind]
                
                if tok_sp[0] == -1:
                    tok_sp[0] = tok_ind
                tok_sp[1] = tok_ind + 1 
        return char2tok_span
    
    def collate_fn(self, batch):
        
        seqs = [x[0] for x in batch]
        if self.is_train:
            labs = [x[1] for x in batch]

        
        batch_len = len(seqs)
        max_len = max([len(s) for s in seqs])
        batch_data=[[0 for i in range(max_len)]for j in range(batch_len)]
        if self.is_train:
            batch_data_lab=[[0 for i in range(max_len)]for j in range(batch_len)]
        batch_data_mask=[[0 for i in range(max_len)]for j in range(batch_len)]
        for j in range(batch_len):
            cur_len = len(seqs[j])
            batch_data[j][:cur_len] = seqs[j]
            if self.is_train:
                batch_data_lab[j][:cur_len] = labs[j]
            batch_data_mask[j][:cur_len] = [1 for _ in range(len(seqs[j]))]
        
        batch_data = torch.tensor(batch_data, dtype=torch.long).to(self.device)
        batch_data_mask = torch.tensor(batch_data_mask, dtype=torch.long).to(self.device)
        if self.is_train:
            batch_data_lab = torch.tensor(batch_data_lab, dtype=torch.long).to(self.device)
            return {
                'datas':batch_data,
                'y_labels':batch_data_lab,
                'datas_mask':batch_data_mask,
            }
        return {
            'datas':batch_data,
            'datas_mask':batch_data_mask,
        }
    
if __name__ == '__main__':
    pass





