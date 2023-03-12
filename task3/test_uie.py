from pprint import pprint
from paddlenlp import Taskflow
import json
import re
from zhon import hanzi

def read_data(data_path):
    with open(data_path,'r') as f:
        res = json.load(f)
    return res

role_name = ['空间实体1','空间实体2','事件','事实性','时间（文本）','时间（标签）的参照事件','时间（标签）','处所','起点','终点','方向','朝向'\
    ,'部件处所','部位','形状','路径','距离（文本）','距离（标签）']

role_idx = {name:idx for idx,name in enumerate(role_name)}



def get_sen_and_sen_sted(context):
    sen_lis=re.findall(hanzi.sentence, context)
    sen_sted = []
    st = 0
    ed = -1 
    for sen in sen_lis:
        ed = st+len(sen)
        sen_sted.append((st,ed))
        st = ed
    return sen_lis, sen_sted

def get_one_sen_pred(result,st,ed):
    cursen_dic = dict()
    for k,v in result.items():
        cur_sen_v_lis = []
        for span in v:
            if(span['start']>=st and span['start']<ed):
                cur_sen_v_lis.append({'text':span['text'],'idxes':[i for i in range(span['start'],span['end'])]})
        cursen_dic[k]=cur_sen_v_lis
    if(cursen_dic.get('空间实体1') == None):
        return []
    pred=[[None for _ in range(len(role_name))] for i in range(len(cursen_dic['空间实体1']))]
    for i in range(len(cursen_dic['空间实体1'])):
        for k,v in cursen_dic.items():
            idx = role_idx[k]
            if(i<len(v)):
                pred[i][idx] = v[i]
    return pred
            

def get_eval_result(data,ie):
    ans = []
    for e in data:
        ce = dict()
        ce['qid']=e['qid']
        ce['context']=e['context']
        sen_lis, sen_sted=get_sen_and_sen_sted(e['context'])
        result = ie(e['context'])[0]

        for k,v in result.items():
            cv = sorted(v, key=lambda x: x['start'])
            result[k]=cv
        preds=[]
        for sted in sen_sted:
            st,ed = sted
            pred=get_one_sen_pred(result,st,ed)
            preds+=pred
        ce['outputs']=preds
        ans.append(ce)
        
    with open('./pred/pred.json','w') as f:
        for lin in ans:
            f.write(json.dumps(lin,ensure_ascii=False)+'\n')
     



if __name__ == '__main__':
    task3_dev_data_path='./ori_data/task3_test_input.json'
    task3_dev_data = read_data(task3_dev_data_path)
    schema = role_name
    ie = Taskflow('information_extraction', schema=schema,task_path='./checkpoint/model_best')
    get_eval_result(task3_dev_data,ie)

