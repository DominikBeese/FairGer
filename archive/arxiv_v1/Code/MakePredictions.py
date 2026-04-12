import numpy as np
import pandas as pd
from tensorflow.data import Dataset
from transformers import BertTokenizer

from Helper.TensorflowPlus import TFCustomBertForSequenceClassification

# Configuration
model_name = 'bert-base-german-cased'
encoding_type = 'c-1'
model_file = r'../Model/best-model.h5'
data_file = r'../Data/Datasets/Frau.json'
output_file = r'Frau-Predictions.json'

# Load data, tokenizer, model, and weights
dataset = pd.read_json(data_file)
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFCustomBertForSequenceClassification.from_pretrained(model_name, num_labels=3, output_act='softmax')
model.load_weights(model_file)

# Tokenize papers
if encoding_type == '1': data = dataset.apply(lambda s: s['sent'], axis=1).to_list()
elif encoding_type == '1+1+1': data = dataset.apply(lambda s: ' '.join(s['prev_sents'][-1:] + [s['sent']] + s['next_sents'][:1]), axis=1).to_list()
elif encoding_type == '3+1+3': data = dataset.apply(lambda s: ' '.join(s['prev_sents'] + [s['sent']] + s['next_sents']), axis=1).to_list()
elif encoding_type == 'c-1': data = dataset.apply(lambda s: s['category'] + '[SEP]' + s['sent'], axis=1).to_list()
elif encoding_type == 'c-1+1+1': data = dataset.apply(lambda s: s['category'] + '[SEP]' + ' '.join(s['prev_sents'][-1:] + [s['sent']] + s['next_sents'][:1]), axis=1).to_list()
elif encoding_type == 'c-3+1+3': data = dataset.apply(lambda s: s['category'] + '[SEP]' + ' '.join(s['prev_sents'] + [s['sent']] + s['next_sents']), axis=1).to_list()
else: raise Exception('Unknown encoding_type: %s' % encoding_type)
encodings = tokenizer(data, max_length=350, padding='max_length', truncation=True)
encoded_dataset = Dataset.from_tensor_slices(dict(encodings))

# Make predictions
outputs = model.predict(encoded_dataset.batch(16), verbose=1).logits
predictions = np.argmax(outputs, axis=1).tolist()

# Save predictions
dataset['output'] = outputs.tolist()
dataset['label'] = predictions
dataset.to_json(output_file, orient='records', indent=2, force_ascii=False)
