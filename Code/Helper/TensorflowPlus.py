import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike
import transformers as trf

from dataclasses import dataclass
from typing import Optional, Tuple

class CustomBertConfig(trf.PretrainedConfig):
	model_type = 'bert'
	
	def __init__(
		self,
		vocab_size=30522,
		hidden_size=768,
		num_hidden_layers=12,
		num_attention_heads=12,
		intermediate_size=3072,
		hidden_act='gelu',
		output_act='linear',
		hidden_dropout_prob=0.1,
		attention_probs_dropout_prob=0.1,
		max_position_embeddings=512,
		type_vocab_size=2,
		initializer_range=0.02,
		layer_norm_eps=1e-12,
		pad_token_id=0,
		position_embedding_type='absolute',
		use_cache=True,
		classifier_dropout=None,
		**kwargs
	):
		super().__init__(pad_token_id=pad_token_id, **kwargs)
		
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.hidden_act = hidden_act
		self.output_act = output_act
		self.intermediate_size = intermediate_size
		self.hidden_dropout_prob = hidden_dropout_prob
		self.attention_probs_dropout_prob = attention_probs_dropout_prob
		self.max_position_embeddings = max_position_embeddings
		self.type_vocab_size = type_vocab_size
		self.initializer_range = initializer_range
		self.layer_norm_eps = layer_norm_eps
		self.position_embedding_type = position_embedding_type
		self.use_cache = use_cache
		self.classifier_dropout = classifier_dropout

class TFCustomBertForSequenceClassification(trf.TFPreTrainedModel):
	_keys_to_ignore_on_load_unexpected = [r'mlm___cls', r'nsp___cls', r'cls.predictions', r'cls.seq_relationship']
	_keys_to_ignore_on_load_missing = [r'dropout']
	
	config_class = CustomBertConfig
	base_model_prefix = 'bert'
    
	def __init__(self, config, *inputs, **kwargs):
		super().__init__(config, *inputs, **kwargs)
		
		self.num_labels = config.num_labels
		
		self.bert = trf.TFBertMainLayer(config, add_pooling_layer=True, name='bert')
		
		classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
		self.dropout = tf.keras.layers.Dropout(rate=classifier_dropout)
		self.classifier = tf.keras.layers.Dense(
			units=self.num_labels,
			activation=config.output_act,
			kernel_initializer=trf.modeling_tf_utils.get_initializer(config.initializer_range),
			name='classifier',
		)
	
	def call(
		self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
		head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None,
		return_dict=None, labels=None, training=False, **kwargs,
	):
		outputs = self.bert(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			training=training,
		)
		
		pooled_output = outputs.pooler_output
		pooled_output = self.dropout(inputs=pooled_output, training=training)
		logits = self.classifier(inputs=pooled_output)
		
		return trf.modeling_tf_outputs.TFSequenceClassifierOutput(
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

class SlantedTriangularLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
	""" https://www.aclweb.org/anthology/P18-1031/ """

	def __init__(
		self,
		maximum_learning_rate: FloatTensorLike,
		number_of_iterations: FloatTensorLike,
		cut_frac: FloatTensorLike = 0.1,
		ratio: FloatTensorLike = 32,
		name: str = None
	):
		super(SlantedTriangularLearningRate, self).__init__()
		self.maximum_learning_rate = maximum_learning_rate
		self.number_of_iterations = number_of_iterations
		self.cut_frac = cut_frac
		self.ratio = ratio
		self.name = name
	
	def __call__(self, step):
		with tf.name_scope(self.name or 'SlantedTriangularLearningRate') as name:
			lr_max = tf.convert_to_tensor(
				self.maximum_learning_rate, name='maximum_learning_rate'
			)
			dtype = lr_max.dtype
			cut_frac = tf.cast(self.cut_frac, dtype)
			ratio = tf.cast(self.ratio, dtype)
			T = tf.cast(self.number_of_iterations, dtype)
			t = tf.cast(step, dtype)
			cut = tf.math.floor(T * cut_frac)
			p = tf.cond(
				t < cut,
				lambda: t / cut,
				lambda: 1 - (t - cut) / (cut * (1 / cut_frac - 1))
			)
			return lr_max * (1 + p * (ratio - 1)) / ratio
	
	def get_config(self):
		return {
			'maximum_learning_rate': self.maximum_learning_rate,
			'number_of_iterations': self.number_of_iterations,
			'cut_frac': self.cut_frac,
			'ratio': self.ratio,
			'name': self.name,
		}
