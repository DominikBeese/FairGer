import numpy as np
import pandas as pd
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.data import Dataset
from tensorflow_addons.optimizers import AdamW
from transformers import BertTokenizer
from time import strftime

from Helper.TensorflowPlus import TFCustomBertForSequenceClassification
from Helper.TensorflowPlus import SlantedTriangularLearningRate
from Helper.HyperparameterTuner import GridTuner

### Configuration ###

# File Configuration
train_file = r'../Data/Human Annotated Data/Migrant.json'
dev_file = r'../Data/Human Annotated Data/Frau.json'

# Model Configuration
model_name = 'bert-base-german-cased'
encoding_type = 'c-1'

# Hyperparameter Configuration
hyperparameters = {
	'batch_size': [8, 16],
	'epochs': [2, 3, 4],
	'learning_rate': [1e-5, 2e-5, 5e-5],
	'warmup_ratio': [0.06],
}
executions_per_trial = 10


### Setup ###

# Load data and tokenizer
dataset = {
	'train': pd.read_json(train_file),
	'dev': pd.read_json(dev_file),
}
tokenizer = BertTokenizer.from_pretrained(model_name)

# Tokenize papers
encoded_dataset = dict()
for t in dataset:
	if encoding_type == '1': data = dataset[t].apply(lambda s: s['sent'], axis=1).to_list()
	elif encoding_type == '1+1+1': data = dataset[t].apply(lambda s: ' '.join(s['prev_sents'][-1:] + [s['sent']] + s['next_sents'][:1]), axis=1).to_list()
	elif encoding_type == '3+1+3': data = dataset[t].apply(lambda s: ' '.join(s['prev_sents'] + [s['sent']] + s['next_sents']), axis=1).to_list()
	elif encoding_type == 'c-1': data = dataset[t].apply(lambda s: s['category'] + '[SEP]' + s['sent'], axis=1).to_list()
	elif encoding_type == 'c-1+1+1': data = dataset[t].apply(lambda s: s['category'] + '[SEP]' + ' '.join(s['prev_sents'][-1:] + [s['sent']] + s['next_sents'][:1]), axis=1).to_list()
	elif encoding_type == 'c-3+1+3': data = dataset[t].apply(lambda s: s['category'] + '[SEP]' + ' '.join(s['prev_sents'] + [s['sent']] + s['next_sents']), axis=1).to_list()
	else: raise Exception('Unknown encoding_type: %s' % encoding_type)
	encodings = tokenizer(data, max_length=350, padding='max_length', truncation=True)
	encoded_dataset[t] = Dataset.from_tensor_slices((dict(encodings), to_categorical(dataset[t]['label'])))


### Model Builder ###

def build_model(hp):
	# load model
	model = TFCustomBertForSequenceClassification.from_pretrained(model_name, num_labels=3, output_act='softmax')
	
	# get parameters
	batch_size = hp.get('batch_size', values=hyperparameters['batch_size'])
	epochs = hp.get('epochs', values=hyperparameters['epochs'])
	learning_rate = hp.get('learning_rate', values=hyperparameters['learning_rate'])
	warmup_ratio = hp.get('warmup_ratio', values=hyperparameters.get('warmup_ratio', [0.0]))
	weight_decay = hp.get('weight_decay', values=hyperparameters.get('weight_decay', [0.0]))
	adam_epsilon = hp.get('adam_epsilon', values=hyperparameters.get('adam_epsilon', [1e-6]))
	adam_beta_1 = hp.get('adam_beta_1', values=hyperparameters.get('adam_beta_1', [0.9]))
	adam_beta_2 = hp.get('adam_beta_2', values=hyperparameters.get('adam_beta_2', [0.999]))
	
	# build model
	lr_schedule = SlantedTriangularLearningRate(
		maximum_learning_rate=learning_rate,
		number_of_iterations=int(epochs*len(encoded_dataset['train'])/batch_size),
		cut_frac=warmup_ratio,
	)
	optimizer = AdamW(
		learning_rate=lr_schedule,
		weight_decay=weight_decay,
		epsilon=adam_epsilon,
		beta_1=adam_beta_1,
		beta_2=adam_beta_2,
	)
	model.compile(
		optimizer=optimizer,
		loss=CategoricalCrossentropy(),
		metrics=[CategoricalAccuracy()],
	)
	return model


### Train Model ###

# Create grid tuner
tuner = GridTuner(
	build_model,
	objective='val_loss',
	direction='minimize',
	executions_per_trial=executions_per_trial,
	save_best_model_weights=True,
	output_dir='TrainModel/%s' % strftime('%Y-%m-%d_%H-%M-%S'),
)

# Start grid search
tuner.search(
	data=encoded_dataset['train'].shuffle(100000, reshuffle_each_iteration=True).batch(16),
	validation_data=encoded_dataset['dev'].batch(16)
)
