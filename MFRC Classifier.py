import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import random
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaModel
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

import wandb
print(wandb.login())


def seed_everything(seed=73):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def mapping_dataset (dataset,mapping,gold_label):
  for index,row in dataset.iterrows():
    number_label = [k for label in row[gold_label].split(',') for k,v in mapping.items() if label.strip() == v ]
    dataset.loc[index,gold_label] = str(number_label)
  return dataset

def one_hot_encoder(df,gold_label):
    one_hot_encoding = []
    for i in tqdm(range(len(df)), desc='Loading:',disable=True):
        temp = [0]*n_labels
        label_indices = list(df.iloc[i][gold_label][1:-1].split(', '))
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

        inputs = self.tokenizer.__call__(f"{text}",
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

class Classifier(nn.Module):
    def __init__(self, num_labels, do_prob, bert_model):
        super(Classifier, self).__init__()
        self.roberta = bert_model #RobertaModel.from_pretrained("Jiva/xlm-roberta-large-it-mnli")
        self.dropout = nn.Dropout(do_prob)
        self.linear = nn.Linear(self.roberta.config.hidden_size, num_labels)  

    def forward(self, ids, mask):
        #per mnli
        outputs = self.roberta(input_ids=ids, attention_mask=mask)#['logits']
        #per roberta - large
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits

def build_dataset(tokenizer_max_len,df,text,n_labels, batch_size):
    dataset =  TwitterDataset(list(df[text]), df[range(n_labels)].values.tolist(), tokenizer, tokenizer_max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

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
            fin_outputs.extend(torch.sigmoid(outputs)) #sigmoide ?
    return fin_outputs, fin_targets

print("Loading models and files ... ")

sweep_config = {
    'method': 'random', #grid, random, bayesian
    'metric': {
      'name': 'auc_score',
      'goal': 'maximize'
    },
    'parameters': {

        'learning_rate': {
            'values': [3e-5]
        },
        'batch_size': {
            'values': [64], #,128] 
        },
        'epochs':{'value': 10},
        'dropout':{'values': [0.1]}, 

        'tokenizer_max_len': {'value': 70}, 
    }
}

sweep_id = wandb.sweep(sweep_config, project='MFRC-RoBERTaLarge') 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mapping = {0: 'Fairness',
           1: 'Care',
           2: 'Loyalty',
           3: 'Authority',
           4: 'Purity',
           5: 'non-moral'
           }

print('Models loaded')


path = "/home/luana/NLPGAT/moral_gpt/data/"

#models
model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name,do_lower_case=True)
bert_model = RobertaModel.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_test = pd.read_csv(f'{path}MFRC_Fr.csv')
train, test = train_test_split(train_test)

#cross-domain
odd1 = pd.read_csv(f'{path}MFRC_EveryDay.csv')
odd2 = pd.read_csv(f'{path}MFRC_USA.csv')

n_labels = len(mapping)
name_text_column = 'text'
name_label_column = 'label'

map_train = mapping_dataset(train,mapping, gold_label = name_label_column )
map_test = mapping_dataset(test,mapping, gold_label = name_label_column )
train = pd.concat([map_train, one_hot_encoder(map_train, gold_label = name_label_column )], axis=1)
test = pd.concat([map_test, one_hot_encoder(map_test, gold_label = name_label_column )], axis=1)

map_odd1 = mapping_dataset(odd1,mapping, gold_label = name_label_column )
map_odd2 = mapping_dataset(odd2,mapping, gold_label = name_label_column )
odd1 = pd.concat([map_odd1, one_hot_encoder(map_odd1, gold_label = name_label_column )], axis=1)
odd2 = pd.concat([map_odd2, one_hot_encoder(map_odd2, gold_label = name_label_column )], axis=1)

seed_everything(1234)

def metrics_execution (data_loader,model,device):
    preds, labels = eval_fn(data_loader, model, device)
    res_metrics = log_metrics(preds, labels)
    auc_score = res_metrics["auc_micro"]
    precision = res_metrics["precision"]
    recall = res_metrics["recall"]
    f1 = res_metrics["f1"]
    return auc_score, precision, recall, f1


def trainer(config=None):
    with wandb.init(config=config):
        config = wandb.config

        train_data_loader = build_dataset(config.tokenizer_max_len,train,name_text_column,n_labels,config.batch_size) #tokenizer_max_len,df,text,n_labels, batch_size
        test_data_loader =  build_dataset(config.tokenizer_max_len,test,name_text_column,n_labels,config.batch_size)
        odd1_data_loader = build_dataset(config.tokenizer_max_len,odd1,name_text_column,n_labels,config.batch_size)
        odd2_data_loader = build_dataset(config.tokenizer_max_len,odd2,name_text_column,n_labels,config.batch_size)

        model = Classifier(n_labels, config.dropout, bert_model=bert_model)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=wandb.config.learning_rate)
        wandb.watch(model)

        n_epochs = config.epochs

        for epoch in tqdm(range(n_epochs),desc='Loading:',disable=True):

            #in-domain
            train_loss = train_fn(train_data_loader, model, optimizer, device) #, scheduler)
            avg_train_loss = train_loss / len(train_data_loader)

            auc_score, precision, recall, f1 = metrics_execution(test_data_loader,model,device)
            auc_score_odd1, precision_odd1, recall_odd1, f1_odd1 = metrics_execution(odd1_data_loader,model,device)
            auc_score_odd2, precision_odd2, recall_odd2, f1_odd2 = metrics_execution(odd2_data_loader,model,device)

            print("AUC score: ", auc_score, "\nF1-score: ", f1, "Average Train loss: ", avg_train_loss)

            if epoch +1 == 10:
                torch.save(model.state_dict(), f'{path}-{epoch}_{f1}.pt' )

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "auc_score": auc_score,
                "auc_score_ood1": auc_score_odd1,
                "auc_score_ood2": auc_score_odd2,
                "precision_in":precision,
                "precision_ood1": precision_odd1,
                "precision_ood2": precision_odd2,
                "recall_in":recall,
                "recall_ood1": recall_odd1,
                "recall_ood2": recall_odd2,
                "F1-score_in":f1,
                "F1-score_ood1": f1_odd1,
                "F1-score_ood2": f1_odd2
            })


wandb.agent(sweep_id, function=trainer, count=1)

