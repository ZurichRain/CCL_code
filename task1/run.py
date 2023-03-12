import sys
import os
from transformers import AutoTokenizer,BertModel,BertTokenizerFast,AutoModel,get_cosine_schedule_with_warmup
from transformers import ElectraPreTrainedModel,ElectraModel
import torch.nn as nn
import torch
from torch import Tensor
from packaging import version
import math
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
import torch.optim as optim
from electra_pos_words_dataset import ElectraPosWordDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score,precision_score,f1_score
import json
import random
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
do_train = False
do_test = True
bert_pathname = 'hfl/chinese-electra-large-discriminator'
electra_pathname = 'hfl/chinese-electra-180g-large-discriminator'
epoch_num = 30
learning_rate = 1e-5
batch_size = 16
save_train_model_file = './model/model_task1.pkl'
save_pred_result = './pred/pred.jsonl'
patience = 0.0002 
patience_num = 10 
min_epoch_num = 3

def f1_score_1(y_true,y_pred):
    return f1_score(y_true,y_pred,labels=[1])

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.4") or use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class FastGELUActivation(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.
    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)


class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.7"):
            self.act = self._silu_python
        else:
            self.act = nn.functional.silu

    def _silu_python(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(input)

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class MishActivation(nn.Module):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9"):
            self.act = self._mish_python
        else:
            self.act = nn.functional.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        return input * torch.tanh(nn.functional.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input: Tensor) -> Tensor:
        return input
ACT2FN = {
    "gelu": GELUActivation(),
    "gelu_10": ClippedGELUActivation(-10, 10),
    "gelu_fast": FastGELUActivation(),
    "gelu_new": NewGELUActivation(),
    "gelu_python": GELUActivation(use_gelu_python=True),
    "linear": LinearActivation(),
    "mish": MishActivation(),
    "quick_gelu": QuickGELUActivation(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "silu": SiLUActivation(),
    "swish": SiLUActivation(),
    "tanh": nn.Tanh(),
}

def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")

class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits


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
class BertCls(nn.Module):
    def __init__(self):
        super(BertCls,self).__init__()
        self.num_labels1=2
        self.bertmodel=AutoModel.from_pretrained(bert_pathname)
        self.tanh=nn.Tanh()
        self.outlin1=nn.Linear(768,self.num_labels1)
        torch.nn.init.xavier_uniform_(self.outlin1.weight)
        self.drop1=nn.Dropout(p=0.15) 
        self.loss1 = nn.CrossEntropyLoss()
        self.span_extractor=SelfAttentiveSpanExtractor(input_dim=768)



    def forward(self, datas,y_labels=None):
        bertout=self.bertmodel(datas)
        wemb=bertout[0]
        activate_emb = wemb[:,0,:]
        activate_emb = self.outlin1(activate_emb)
        wemb1=activate_emb.view(-1,self.num_labels1)
        if y_labels!=None:
            ctrain_y=y_labels.view(-1)
            l1=self.loss1(wemb1,ctrain_y)
            return l1,wemb1
        return wemb1


class Electra_pos_token_cls(ElectraPreTrainedModel):

    def __init__(self,config):
        super(Electra_pos_token_cls,self).__init__(config)
        self.electra = ElectraModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
        self.tanh=nn.Tanh()
        self.sen_mlp=nn.Linear(1024,1)
        torch.nn.init.xavier_uniform_(self.sen_mlp.weight)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.span_extractor=SelfAttentiveSpanExtractor(input_dim=1024)
        self.sigd = nn.Sigmoid()
        self.tok_mlp = nn.Linear(1024,1)


    def forward(self, datas,data_mask,data_pos_mask,y_labels=None):

        electra_out=self.electra(input_ids=datas,attention_mask=data_mask)

        tokemb=electra_out[0]
        sen_activate_emb = tokemb[:,0,:]

        token_activate = self.discriminator_predictions(tokemb)
        sen_tok_logits = torch.max(token_activate.masked_fill(mask=(1.0-data_pos_mask).to(torch.bool),value=-1e9),dim=-1).values
        sen_logits = self.discriminator_predictions(sen_activate_emb)

        if y_labels!=None:
   
            y_labels = 1-y_labels
            l_bce_sen = self.bce_loss(sen_logits,y_labels)
            l_bce_tok = self.bce_loss(sen_tok_logits,y_labels)
            return l_bce_sen+l_bce_tok,torch.max(1-self.sigd(sen_tok_logits),1-self.sigd(sen_logits))
        return torch.max(1-self.sigd(sen_tok_logits),1-self.sigd(sen_logits))


def train_one_epoch(model,train_loader,optimizer,scheduler):
    model.train()
    train_losses = 0
    prey1_lis=[]
    truy1_lis=[]
    for batchdata in tqdm(train_loader):
        l1,prey = model(**batchdata)
        prey1_lab = torch.where(prey > 0.5,1,0).view(-1)
        prey1_lis += prey1_lab.to('cpu').numpy().tolist()
        truy1_lis += batchdata['y_labels'].view(-1).to('cpu').numpy().tolist()
        train_losses += l1.item()
        optimizer.zero_grad()
        l1.backward()
        optimizer.step()
        scheduler.step()
    train_epoch_f1=f1_score_1(truy1_lis,prey1_lis)
    print('train_losses: ',train_losses/len(train_loader))
    print('-'*30,'train_epoch_f1: ',train_epoch_f1)
def dev_one_epoch(model,dev_loader):
    res=dict()
    model.eval()
    dev_losses = 0
    prey1_lis=[]
    truy1_lis=[]
    with torch.no_grad():
        for batchdata in tqdm(dev_loader):
            l1,prey = model(**batchdata)
            prey1_lab = torch.where(prey > 0.5,1,0).view(-1)
            prey1_lis += prey1_lab.to('cpu').numpy().tolist()
            truy1_lis += batchdata['y_labels'].view(-1).to('cpu').numpy().tolist()
            dev_losses += l1.item()
    dev_epoch_f1=f1_score_1(truy1_lis,prey1_lis)
    res['f1'] = dev_epoch_f1
    res['loss'] = dev_losses/len(dev_loader)
    print('dev_losses: ',dev_losses/len(dev_loader))
    print('-'*30,'dev_epoch_f1: ',dev_epoch_f1)
    return res

def eval(test_loader,qid_lis):
    model = torch.load(save_train_model_file)
    model.eval()
    
    pre_seq=[]
    prey1_lis=[]
    tokenizer = AutoTokenizer.from_pretrained(bert_pathname)
    with torch.no_grad():
        for batchdata in tqdm(test_loader):
            prey = model(**batchdata)
            prey1_lab = torch.where(prey > 0.5,1,0).view(-1)
            for seq in batchdata['datas']:
                pre_seq.append(tokenizer.decode(seq,skip_special_tokens=True))
            prey1_lis += prey1_lab.to('cpu').numpy().tolist()

    pred_lis=[]
    for qid,seq,pre_lab in zip(qid_lis,pre_seq,prey1_lis):
        pred_lis.append(
            {
                'qid':qid,
                'context': seq,
                'judge': pre_lab,
            }
        )
    with open(save_pred_result,'w',encoding='utf-8') as f:
        for p in pred_lis:
            json_str = json.dumps(p)
            f.write(json_str+'\n')
    return pred_lis

def write_result(seqtokenlis, prelabel, truelabel , filename):
    tokenizer = BertTokenizerFast.from_pretrained(bert_pathname)
    with open(filename, 'w') as f:
        for idx in range(len(seqtokenlis)):
            seq=tokenizer.decode(seqtokenlis[idx],skip_special_tokens= True)
            f.writelines(str(seq)+';'+ str(prelabel[idx]) +';'+ str(truelabel[idx])+'\n')

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

def read_data(data_path):
    # res=dict()
    with open(data_path,'r') as f:
        res = json.load(f)
    return res

def get_task1_input(data,mod='train'):
    if mod == 'train':
        seq=[]
        lab=[]
        for example in data:
            seq.append(example['context'])
            lab.append(int(example['judge']))
        return seq, lab
    seq=[]
    qid_lis=[]
    for example in data:
        qid_lis.append(example['qid'])
        seq.append(example['context'])
    return seq,qid_lis

def get_k_idx_train_and_dev(k,k_span,data):
    train_data = data[k*k_span:(k+7)*k_span]
    dev_data = data[(k+7)*k_span:] + data[:k*k_span]
    return train_data,dev_data


if __name__ == '__main__':
    seed_everything(666)
    task1_train_data_path='./data/task1_train.json'
    task1_dev_data_path='../data/task1_dev.json'
    task1_train_data = read_data(task1_train_data_path)
    task1_dev_data = read_data(task1_dev_data_path)


    train_sentences,train_labels = get_task1_input(task1_train_data)
    dev_sentences,dev_labels = get_task1_input(task1_dev_data)

    train_dataset = ElectraPosWordDataset(train_sentences,train_labels)
    dev_dataset = ElectraPosWordDataset(dev_sentences,dev_labels)
    train_size = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=train_dataset.collate_fn)

    dev_loader = DataLoader(dev_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=dev_dataset.collate_fn)

    model = Electra_pos_token_cls.from_pretrained(electra_pathname).to(device)
    optimizer = optim.AdamW(model.parameters(), lr= learning_rate)
    train_steps_per_epoch = train_size // batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=epoch_num * train_steps_per_epoch)
    if do_train:
        best_val_f1 = 0.0
        patience_counter = 0
        for epoch in range(epoch_num):
            print('train epoch: {}'.format(epoch+1))
            train_one_epoch(model,train_loader,optimizer,scheduler)
            val_metrics=dev_one_epoch(model,dev_loader)
            val_f1 = val_metrics['f1']
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
            if (patience_counter >= patience_num and epoch > min_epoch_num) or epoch == epoch_num:
                break
        print('best_f1: {}'.format(best_val_f1))
    if do_test:
        
        task1_test_data_path = '../data/task1_test_input.json'
        task1_test_data = read_data(task1_test_data_path)
        test_sentences,qid_lis = get_task1_input(task1_test_data,'test')
        test_dataset = ElectraPosWordDataset(test_sentences)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=test_dataset.collate_fn_test) 
  
        eval(test_loader,qid_lis)       
        predictions=[]
        with open(save_pred_result,'r',encoding='utf-8') as f:
            for line in f:
                predictions.append(json.loads(line))
        print(predictions[0])
    