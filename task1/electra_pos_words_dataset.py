
import sys
import os

from transformers import BertTokenizer,BertTokenizerFast,AutoTokenizer
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import json
from torch.utils.data import Dataset
import re

from sklearn.metrics import recall_score,precision_score,f1_score,confusion_matrix,roc_curve,accuracy_score

bert_pathname = 'hfl/chinese-electra-base-discriminator'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class ElectraPosWordDataset(Dataset):
    def __init__(self, seqlis, labellis=None):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_pathname)
        self.device = device
        self.is_test = labellis==None
        self.dataset = self.preprocess(seqlis,labellis)
        
    def preprocess(self,seqlis,labellis):
        seqtoklis , seqtokidslis, pos_mask = self.get_seq_tok(seqlis,self.tokenizer)
        data=[]
        if(self.is_test):
            for seq, pos in zip(seqtokidslis,pos_mask):
                data.append((seq,pos))
        else:
            for seq,pos,lab in zip(seqtokidslis,pos_mask,labellis):
                data.append((seq,pos,lab))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        seq = self.dataset[idx][0]
        pos = self.dataset[idx][1]
        if (not self.is_test):
            lab = self.dataset[idx][2]
            return [seq,pos,lab]
        return [seq,pos]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)
    def get_char2tok_span_one_seq(slef,seq,tokenizer):
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

        return token_span,char2tok_span
    def get_seq_tok(self,seqlis,tokenizer):
        pos_words='[上下左右东南西北里外前后旁回中边内来去出入进侧顶底面背口]'
        regex = re.compile(pos_words)
        pos_mask = []
        curseqtoklis=[]
        curseqtokidslis=[]
        doc_max_seq_len=0
        for seq in seqlis:
            curseqtok=tokenizer.tokenize(seq)
            doc_max_seq_len=max(doc_max_seq_len,len(curseqtok))
            curseqtoklis.append(curseqtok)
            curseqtokidslis.append(tokenizer.convert_tokens_to_ids(curseqtok))
            c_pos_mask=[0 for _ in range(len(curseqtok))]
            token_span,char2tok_span=self.get_char2tok_span_one_seq(seq,self.tokenizer)
            pos_idx_lis = [char2tok_span[r.span()[0]][0] for r in regex.finditer(seq)]
            if(len(pos_idx_lis)>0):
                for pos_idx in pos_idx_lis:
                    c_pos_mask[pos_idx]=1
            else:
                c_pos_mask[0]=1
            pos_mask.append(c_pos_mask)

        
        return curseqtoklis,curseqtokidslis,pos_mask
    
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
        pos_mask = [x[1] for x in batch]
        labs = [x[2] for x in batch]

        
        batch_len = len(seqs)
        max_len = max([len(s) for s in seqs])
        batch_data=[[0 for i in range(max_len)]for j in range(batch_len)]
        batch_data_mask = [[0 for i in range(max_len)]for j in range(batch_len)]
        bacth_data_pos_mask = [[0 for i in range(max_len)]for j in range(batch_len)]
        for j in range(batch_len):
            cur_len = len(seqs[j])
            batch_data[j][:cur_len] = seqs[j]
            batch_data_mask[j][:cur_len] = [1 for _ in range(len(seqs[j]))]
            bacth_data_pos_mask[j][:cur_len] = pos_mask[j]
        
        
        batch_data = torch.tensor(batch_data, dtype=torch.long).to(self.device)
        batch_data_mask = torch.tensor(batch_data_mask,dtype=torch.long).to(self.device)
        bacth_data_pos_mask = torch.tensor(bacth_data_pos_mask,dtype=torch.long).to(self.device)
        batch_labs=torch.tensor(labs,dtype=torch.float).to(self.device)
        return {
            'datas':batch_data,
            'data_mask':batch_data_mask,
            'y_labels':batch_labs,
            'data_pos_mask':bacth_data_pos_mask,
        }
    def collate_fn_test(self, batch):
        seqs = [x[0] for x in batch]
        pos_mask = [x[1] for x in batch]
        batch_len = len(seqs)
        max_len = max([len(s) for s in seqs])
        batch_data=[[0 for i in range(max_len)]for j in range(batch_len)]
        batch_data_mask = [[0 for i in range(max_len)]for j in range(batch_len)]
        bacth_data_pos_mask = [[0 for i in range(max_len)]for j in range(batch_len)]
        for j in range(batch_len):
            cur_len = len(seqs[j])
            batch_data[j][:cur_len] = seqs[j]
            batch_data_mask[j][:cur_len] = [1 for _ in range(len(seqs[j]))]
            bacth_data_pos_mask[j][:cur_len] = pos_mask[j]
        
        
        batch_data = torch.tensor(batch_data, dtype=torch.long).to(self.device)
        bacth_data_pos_mask = torch.tensor(bacth_data_pos_mask,dtype=torch.long).to(self.device)
        batch_data_mask = torch.tensor(batch_data_mask,dtype=torch.long).to(self.device)
        return {
            'datas':batch_data,
            'data_mask':batch_data_mask,
            'data_pos_mask':bacth_data_pos_mask
        }
    
if __name__ == '__main__':
    pass





