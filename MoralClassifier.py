import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import random
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

import wandb
print(wandb.login())

print("Loading models and files ... ")

sweep_config = {
    'method': 'random', #grid, random, bayesian
    'metric': {
      'name': 'auc_score',
      'goal': 'maximize'
    },
    'parameters': {

        'learning_rate': {
            'values': [2e-5] #best_value --> 3e-5
        },
        'batch_size': {
            'values': [64], #,128] #best_value --> 64
        },
        'epochs':{'value': 10},
        'dropout':{'values': [0.4]}, #[0.1,0.2,0.3]}, #, 0.2] #best_value --> 0.3

        'tokenizer_max_len': {'value': 40}, #70
    }
}

sweep_id = wandb.sweep(sweep_config, project='BERT_MoralValues') #

device = torch.device('cuda:0') # if torch.cuda.is_available() else 'cpu')

mapping = {0: 'fairness',
           1:'cheating',
           2: 'care',
           3:'harm',
           4: 'loyalty',
           5:'betrayal',
           6: 'authority',
           7:'subversion',
           8: 'purity',
           9:'degradation',
           10: 'non-moral'
           }

print('Models loaded')

GOLD_LABEL = 'tweet_label'
TEXT = 'tweet_text'

path = "/home/luana/NLPGAT/moral_values/data/"
train =  pd.read_csv(f'{path}Twitter_train.csv')
validation = pd.read_csv(f'{path}Twitter_test.csv')

n_labels = len(mapping)

def seed_everything(seed=73):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def mapping_dataset (dataset,mapping):
  for index,row in dataset.iterrows():
    number_label = [k for label in row[GOLD_LABEL].split(',') for k,v in mapping.items() if label.strip() == v ]
    dataset.loc[index,GOLD_LABEL] = str(number_label)
  return dataset

def one_hot_encoder(df):
    one_hot_encoding = []
    for i in tqdm(range(len(df)), desc='Loading:',disable=True):
        temp = [0]*n_labels
        label_indices = list(df.iloc[i][GOLD_LABEL][1:-1].split(', '))
        for index in label_indices:
            temp[int(index)] = 1
        one_hot_encoding.append(temp)
    return pd.DataFrame(one_hot_encoding)

class TwitterDataset:
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        inputs = self.tokenizer.__call__(text,
                                         None,
                                         add_special_tokens=True,
                                         max_length=self.max_len,
                                         padding="max_length",
                                         truncation=True,
                                         )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }

class ValuesClassifier(nn.Module):
    def __init__(self, n_classes, do_prob, bert_model):
        super(ValuesClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(do_prob)
        self.out = nn.Linear(768, n_classes)

    def forward(self, ids, mask):
        output_1 = self.bert(ids, attention_mask=mask)["pooler_output"]
        output_2 = self.dropout(output_1)
        output = self.out(output_2)
        return output

def build_dataset(tokenizer_max_len,train,valid):
    train_dataset = TwitterDataset(list(train[TEXT]), train[range(n_labels)].values.tolist(), tokenizer,
                                   tokenizer_max_len)
    valid_dataset = TwitterDataset(list(valid[TEXT]), valid[range(n_labels)].values.tolist(), tokenizer,
                                   tokenizer_max_len)

    return train_dataset, valid_dataset

def build_dataloader(train_dataset, valid_dataset, batch_size):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_data_loader, valid_data_loader

def loss_function (outputs, labels):
    return nn.MultiLabelSoftMarginLoss()(outputs, labels.float())

def log_metrics(preds, labels):
    preds = torch.stack(preds)
    preds = preds.cpu().detach().numpy()
    labels = torch.stack(labels)
    labels = labels.cpu().detach().numpy()
    threshold = 0.5

    fpr_micro, tpr_micro, _ = metrics.roc_curve(labels.ravel(), preds.ravel())
    pred_f1 = [[1 if score >= threshold else 0 for score in ele] for ele in preds]

    return {"auc_micro": metrics.auc(fpr_micro, tpr_micro),
            "precision": metrics.precision_score(labels, pred_f1, average='weighted'),
            "recall": metrics.recall_score(labels, pred_f1, average='weighted'),
            "f1": metrics.f1_score(labels, pred_f1, average='weighted')}

def train_fn(data_loader, model, optimizer, device): #, scheduler):
    train_loss = 0.0
    model.train()
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader), desc='Loading:',disable=True):
        ids = d["ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)
        targets = d["labels"].to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask)

        loss = loss_function(outputs, targets)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # scheduler.step()
    return train_loss

def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader)):
            ids = d["ids"].to(device, dtype=torch.long)
            mask = d["mask"].to(device, dtype=torch.long)
            targets = d["labels"].to(device, dtype=torch.long)

            outputs = model(ids=ids, mask=mask)
            fin_targets.extend(targets)
            fin_outputs.extend(torch.sigmoid(outputs))
    return fin_outputs, fin_targets


seed_everything(1234)

map_train = mapping_dataset(train,mapping)
map_validation = mapping_dataset(validation,mapping)
train = pd.concat([map_train, one_hot_encoder(map_train)], axis=1)
valid = pd.concat([map_validation, one_hot_encoder(map_validation)], axis=1)

model_name = "squeezebert/squeezebert-uncased"
tokenizer = transformers.SqueezeBertTokenizer.from_pretrained(model_name, do_lower_case=True)
bert_model =  transformers.SqueezeBertModel.from_pretrained(model_name)

def trainer(config=None):
    with wandb.init(config=config):
        config = wandb.config

        train_dataset, valid_dataset = build_dataset(config.tokenizer_max_len,train,valid)
        train_data_loader, valid_data_loader = build_dataloader(train_dataset, valid_dataset, config.batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ValuesClassifier(n_labels, config.dropout, bert_model=bert_model)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=wandb.config.learning_rate)
        wandb.watch(model)

        n_epochs = config.epochs

        for epoch in tqdm(range(n_epochs),desc='Loading:',disable=True):

            train_loss = train_fn(train_data_loader, model, optimizer, device) #, scheduler)
            preds, labels = eval_fn(valid_data_loader, model, device)


            res_metrics = log_metrics(preds, labels)
            auc_score = res_metrics["auc_micro"]
            precision = res_metrics["precision"]
            recall = res_metrics["recall"]
            f1 = res_metrics["f1"]
            avg_train_loss = train_loss / len(train_data_loader)

            print("AUC score: ", auc_score, "\nF1-score: ", f1, "Average Train loss: ", avg_train_loss)

            if epoch +1 == 10:
                torch.save(model.state_dict(), f'{path}-{epoch}_{f1}.pt' )

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "auc_score": auc_score,
                "precisio":precision,
                "recall":recall,
                "F1-score":f1
            })


wandb.agent(sweep_id, function=trainer, count=1)

