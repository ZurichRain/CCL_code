import json
def read_data(data_path):
    with open(data_path,'r') as f:
        res = json.load(f)
    return res

role_name = ['空间实体1','空间实体2','事件','事实性[假]','时间（文本）','时间（标签）的参照事件','时间（标签）[说话时，过去，将来，之时，之前，之后]','处所','起点','终点','方向','朝向'\
    ,'部件处所','部位','形状','路径','距离（文本）','距离（标签）[远，近，变远，变近]']
def get_uie_data(data):
    uie_data=[]
    uie_data_idx=0
    e_idx=0
    for e in data:
        cuie_e=dict()
        cuie_e['id'] = uie_data_idx
        cuie_e['text'] = e['context']
        cuie_e['relations'] = []
        cuie_e['entities'] = []
        for r in e['outputs']:
            for sidx,span in enumerate(r):
                if span is not None:
                    if(sidx in [3,6,17]):
                        continue
                    cuie_e['entities'].append(
                        {'id':e_idx,"start_offset": span['idxes'][0],"end_offset": span['idxes'][-1]+1, "label": role_name[sidx]}
                    )
                    e_idx+=1
        uie_data_idx+=1
        uie_data.append(cuie_e)
    with open('./data/doccano_dev_ext.json','w') as f:
        for d in uie_data:
            f.write(json.dumps(d,ensure_ascii=False)+'\n')

task3_train_data_path='./ori_data/task3_train.json'
task3_dev_data_path='./ori_data/task3_dev.json'

task3_train_data = read_data(task3_train_data_path)
task3_dev_data = read_data(task3_dev_data_path)

print(len(task3_train_data))
print(len(task3_dev_data))
print(task3_train_data[0])
get_uie_data(task3_dev_data)