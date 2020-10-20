import pandas as pd
train = pd.read_table('train.txt', header=None, names=['sub', 'pred', 'obj', 'time_begin'])
test = pd.read_table('test.txt', header=None, names=['sub', 'pred', 'obj', 'time_begin'])
valid = pd.read_table('valid.txt', header=None, names=['sub', 'pred', 'obj', 'time_begin'])

train['time_end'] = train['time_begin']
test['time_end'] = test['time_begin']
valid['time_end'] = valid['time_begin']

triple2id = pd.concat([train,test,valid])

ents = {x : i for i, x in enumerate(triple2id['sub'].append(triple2id['obj']).unique())}
preds = {x : i for i, x in enumerate(triple2id['pred'].unique()) }

items = ['train', 'test', 'valid', 'triple2id']
for i, split in enumerate([train,test,valid, triple2id]):
    split["sub"] = split["sub"].map(ents)
    split["pred"] = split["pred"].map(preds)
    split["obj"] = split["obj"].map(ents)
    split.to_csv(items[i] + '.txt', sep="\t", header=False, columns=['sub', 'pred', 'obj','time_begin','time_end'], index=False)


items = ['entity2id', 'relation2id']
for i, elem in enumerate([ents, preds]):
    with open(items[i] + '.txt', "w", encoding="utf-8") as f:
        for name, id in elem.items():
            f.write(f'{id}\t{name}\n')