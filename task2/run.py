import sys
import os
sys.path.append('./task1/')
from transformers import (BertTokenizer,AutoModel,AutoTokenizer,
                        get_cosine_schedule_with_warmup)
import torch.nn as nn
import torch
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
import torch.optim as optim
from dataset import Task2TypeCDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score,precision_score,f1_score
import json
import random
import numpy as np
from awp import AWP

do_train = False
do_test = True
use_awp = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_path = 'hfl/chinese-roberta-wwm-ext-large'
epoch_num = 20
learning_rate = 1e-5
batch_size = 16
save_train_model_file = './model/model_task2.pkl'
save_pred_result = './pred/pred.json'
patience = 0.0002 
patience_num = 100 
min_epoch_num = 3


tokenizer = AutoTokenizer.from_pretrained(model_path)

def f1_score_1(y_true,y_pred):
    return f1_score(y_true,y_pred,labels=[1,2,3],average='macro')


class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
class Task2Model(nn.Module):
    def __init__(self):
        super(Task2Model,self).__init__()
        self.bert=AutoModel.from_pretrained(model_path)
        self.span_extractor=SelfAttentiveSpanExtractor(input_dim=768)
        self.tanh=nn.Tanh()
        self.lab_mlp=nn.Linear(1024,4)
        torch.nn.init.xavier_uniform_(self.lab_mlp.weight)
        self.drop1=nn.Dropout(p=0.15)
        self.ce_loss = nn.CrossEntropyLoss()



    def forward(self, datas,datas_mask,y_labels=None):
        bert_out = self.bert(input_ids=datas,attention_mask=datas_mask)
        wemb = bert_out[0]
        lab1_logits = self.lab_mlp(wemb)

        if y_labels !=None:
            datas_mask=datas_mask.view(-1) == 1

            loss = self.ce_loss(lab1_logits.view(-1,4)[datas_mask],y_labels.view(-1)[datas_mask])
            return loss, lab1_logits
        return lab1_logits


def train_one_epoch(model,train_loader,optimizer,scheduler):
    model.train()
    train_losses = 0
    prey_lab_lis = []
    true_lab_lis = []
    for batchdata in tqdm(train_loader):
        l1,lab1_logits = model(**batchdata)
        prey_lab = torch.argmax(lab1_logits,dim=-1)
        datas_mask=batchdata['datas_mask'].view(-1) == 1
        prey_lab_lis += prey_lab.view(-1)[datas_mask].to('cpu').detach().numpy().tolist()
        true_lab_lis += batchdata['y_labels'].view(-1)[datas_mask].to('cpu').detach().numpy().tolist()
        train_losses += l1.item()
        optimizer.zero_grad()
        l1.backward()
        if use_awp:
            loss = awp.attack_backward(batchdata, batchdata['y_labels'].view(-1)[datas_mask])
            loss.backward()
            awp._restore()
        optimizer.step()
        scheduler.step()
    train_epoch_lab_f1= f1_score_1(prey_lab_lis,true_lab_lis)
    print('train_losses: ',train_losses/len(train_loader))
    print('-'*30,'train_epoch_lab1_f1: {}'.format(train_epoch_lab_f1))
def dev_one_epoch(model,dev_loader):
    res=dict()
    model.eval()
    dev_losses = 0
    prey_lab_lis = []
    true_lab_lis = []
    for batchdata in tqdm(dev_loader):
        l1,lab1_logits = model(**batchdata)
        prey_lab = torch.argmax(lab1_logits,dim=-1)
        datas_mask=batchdata['datas_mask'].view(-1) == 1
        prey_lab_lis += prey_lab.view(-1)[datas_mask].to('cpu').detach().numpy().tolist()
        true_lab_lis += batchdata['y_labels'].view(-1)[datas_mask].to('cpu').detach().numpy().tolist()
        dev_losses += l1.item()

    dev_epoch_lab_f1= f1_score_1(prey_lab_lis,true_lab_lis)
    res['lab_f1'] = dev_epoch_lab_f1
    res['loss'] = dev_losses
    print('dev_losses: ',dev_losses/len(dev_loader))
    print('-'*30,'dev_epoch_lab1_f1: {}'.format(dev_epoch_lab_f1))
    return res

def eval(test_loader,qid_lis):
    model = torch.load(save_train_model_file)
    model.eval()
    prey_lab_lis = []
    seq_lis = []
    for batchdata in tqdm(test_loader):
        lab1_logits = model(**batchdata)
        prey_lab = torch.argmax(lab1_logits,dim=-1)
        for idx in range(batchdata['datas'].size()[0]):
            de_seq = tokenizer.decode(batchdata['datas'][idx][batchdata['datas_mask'][idx]==1].to('cpu').detach().numpy().tolist())
            tr_seq = ''
            for c in de_seq:
                if c!=' ':
                    tr_seq+=c
            seq_lis.append(tr_seq)
            prey_lab_lis.append(prey_lab[idx][batchdata['datas_mask'][idx]==1].to('cpu').detach().numpy().tolist())
    pred_lis=[]
    for qid,seq, lab in zip(qid_lis,seq_lis,prey_lab_lis):
        text1 = ''
        text1_idx = []
        text2 = ''
        text2_idx = []

        text3 = ''
        text3_idx = []
        token_span,char2tok_span = get_char2tok_spanlis_one_seq(seq,tokenizer)
        for idx in range(len(lab)):
            if(lab[idx]==1):
                for cidx in range(token_span[idx][0],token_span[idx][1]):
                    text1+=seq[cidx]
                    text1_idx.append(cidx)
            if(lab[idx]==2):
                for cidx in range(token_span[idx][0],token_span[idx][1]):
                    text2+=seq[cidx]
                    text2_idx.append(cidx)

            if(lab[idx]==3):
                for cidx in range(token_span[idx][0],token_span[idx][1]):
                    text3+=seq[cidx]
                    text3_idx.append(cidx)
        if(len(text1)>0 or len(text2)>0 or len(text3)>0):
            pred_lis.append(
                {
                    'qid':qid,
                    'context': seq,
                    'reasons': [
                        { 
                            "fragments": [],
                            "type": "C" 
                        }
                    ]
                }
            )
            if (len(text1)>0):
                pred_lis[-1]['reasons'][0]["fragments"].append({ "role": "S", "text": text1, "idxes": text1_idx})
            if (len(text2)>0):
                pred_lis[-1]['reasons'][0]["fragments"].append({ "role": "P", "text": text2, "idxes": text2_idx})
            if (len(text3)>0):
                pred_lis[-1]['reasons'][0]["fragments"].append({ "role": "E", "text": text3, "idxes": text3_idx})
        else:
            pred_lis.append(
                {
                    'qid':qid,
                    'context': seq,
                    'reasons': [],
                }
            )
    
    with open(save_pred_result,'w',encoding='utf-8') as f:
        for p in pred_lis:
            json_str = json.dumps(p,ensure_ascii=False)
            f.write(json_str+'\n')

def get_char2tok_spanlis_one_seq(seq,tokenizer):
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

def write_result(seqtokenlis, prelabel, truelabel , filename):
    with open(filename, 'w') as f:
        for idx in range(len(seqtokenlis)):
            seq=tokenizer.decode(seqtokenlis[idx],skip_special_tokens= True)
            f.writelines(str(seq)+';'+ str(prelabel[idx]) +';'+ str(truelabel[idx])+'\n')

def read_data(data_path):
    with open(data_path,'r') as f:
        res = json.load(f)
    return res


def seed_everything(seed=1226):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True 

def Ain_reason(example):
    A_lis = []
    for r in example['reasons']:
        if r["type"]=='C':
            A_lis.append(r)
    return A_lis

def get_task2_input(data,mod='train'):
    if mod =='train':
        seq=[]
        lab=[]
        for example in data:
            A_lis = Ain_reason(example)
            if(len(A_lis)>0):
                seq.append(example['context'])
                lab.append(A_lis)
        return seq, lab
    seq=[]
    for e in data:
        seq.append(e['context'])
    return seq

if __name__ == '__main__':
    seed_everything()
    if do_train:
        task2_train_data_path='./data/task2_train.json'
        task2_dev_data_path='./data/task2_dev.json'
        task2_train_data = read_data(task2_train_data_path)
        task2_dev_data = read_data(task2_dev_data_path)
        train_sentences,train_labels = get_task2_input(task2_train_data)
        dev_sentences,dev_labels = get_task2_input(task2_dev_data)
        train_dataset = Task2TypeCDataset(train_sentences,train_labels)
        dev_dataset = Task2TypeCDataset(dev_sentences,dev_labels)
        train_size = len(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn=train_dataset.collate_fn)

        dev_loader = DataLoader(dev_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn=dev_dataset.collate_fn)

        model = Task2Model().to(device)
        optimizer = optim.AdamW(model.parameters(), lr= learning_rate)
        if use_awp:
            loss_fn = nn.CrossEntropyLoss()
            awp = AWP(model,loss_fn,optimizer)
        train_steps_per_epoch = train_size // batch_size
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=train_steps_per_epoch,
                                                    num_training_steps=epoch_num * train_steps_per_epoch)
        best_val_f1 = 0.0
        patience_counter = 0
        for epoch in range(epoch_num):
            print('epoch : {}'.format(epoch))
            train_one_epoch(model,train_loader,optimizer,scheduler)
            val_metrics=dev_one_epoch(model,dev_loader)
            val_f1 = val_metrics['lab_f1']
            improve_f1 = val_f1 - best_val_f1
            
            if improve_f1 > 1e-5:
                best_val_f1 = val_f1
                torch.save(model,save_train_model_file)
                if improve_f1 < patience: 
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1
            # Early stopping and logging best f1
            if (patience_counter >= patience_num and epoch > min_epoch_num) or epoch == epoch_num:
                break
        print('best_f1: {}'.format(best_val_f1))

    if do_test:
        task2_test_data_path='./data/task2_test_input.json'
        task2_test_data = read_data(task2_test_data_path)
        test_sentences = get_task2_input(task2_test_data,mod='test')
        test_dataset = Task2TypeCDataset(test_sentences)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=test_dataset.collate_fn)
        qid_lis = []
        for e in task2_test_data:
            qid_lis.append(e['qid'])
        eval(test_loader,qid_lis)