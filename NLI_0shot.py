import pandas as ps
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
model.to('cuda')

def nli (premise,hypothesis):
  sent = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                     truncation_strategy='only_first')
  logits = model(sent.to('cuda'))[0]
  entail_contradiction_logits = logits[:,[1,2]]
  probs = entail_contradiction_logits.softmax(dim=1)
  prob_label_is_true = probs[:,1]
  return prob_label_is_true

name_subcorpora = 'MFRC_USA' #name of the dataset
df = pd.read_csv(f'{name_subcorpora}.csv')

mapping = {0: 'fairness',
           1:'cheating',
           2: 'care',
           3:'harm',
           4: 'loyalty',
           5:'betrayal',
           6: 'authority',
           7:'subversion',
           8: 'Purity',
           9:'degradation'
           }

prompt = "This text conveys the moral values of {}."
out = {'text':[], 'label':[], 'gpt_pred':[], 'nli_pred':[]}
for i,r in df.iterrows():
  results = []
  for value in mapping.values():
    res = nli(r['text'],prompt.replace('{}',value))
    if res.tolist()[0] >= 0.5:

      results.append(value)
  if len(results) == 0:
    results  = 'non-moral'
  else:
    results = ', '.join(results)

  out['text'].append(r['text'])
  out['gpt_pred'].append(r['gpt_pred'])
  out['label'].append(r['label'])
  out['nli_pred'].append(results)


pd.DataFrame(out).to_csv('f{name_subcorpora}_nli.csv',index=False)