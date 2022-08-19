# Code
This folder contains [(a)](#code-to-apply-the-model) the code to apply a model to our datasets and [(b)](#code-to-train-the-model) the code to train a model on our dataset.
 * 📜 [Code to apply the model](MakePredictions.py)
 * 📜 [Code to train the model](TrainModel.py)
 * 📂 [Helper](Helper)
   * 📜 Code for performing a grid search
   * 📜 Implementation of a _slanted triangular learning rate_ ([Howard and Ruder, 2018](https://aclanthology.org/P18-1031/)) and a three class accuracy metric

We tested the code using Python 3.10.6. The required libraries are listed in the [`requirements.txt`](requirements.txt).


## Code to apply the model
You can apply a model of your choice to a dataset of your choice by changing the confiugration in line 11-13:
```python
model_file = r'../Model/best-model.h5'
data_file = r'../Data/Datasets/Frau.json'
output_file = r'Frau-Predictions.json'
```

You can change the model and encoding type to use by chancing the configuration in line 9-10:
```python
model_name = 'bert-base-german-cased'
encoding_type = 'c-1'
```

Options for the `encoding_type` are:
| Option    | Description                                                                                  |
|-----------|----------------------------------------------------------------------------------------------|
| `1`       | only feed the target sentence                                                                |
| `1+1+1`   | feed the target sentence preceded and followed by one surrounding sentence                   |
| `3+1+3`   | feed the target sentence preceded and followed by three surrounding sentences                |
| `c-1`     | feed the target and the target sentence                                                      |
| `c-1+1+1` | feed the target and the target sentence preceded and followed by one surrounding sentence    |
| `c-3+1+3` | feed the target and the target sentence preceded and followed by three surrounding sentences |


## Code to train the model
You can train a model on a dataset of your choice by changing the train and dev datasets in line 17-19:
```python
# File Configuration
train_file = r'../Data/Human Annotated Data/all.json'
dev_file = r'../Data/Human Annotated Data/Frau.json'
```

You can change the pretrained weights and encoding type by chancing the configuration in lines 21-23:
```python
# Model Configuration
model_name = 'bert-base-german-cased'
encoding_type = 'c-1'
```

You can also specify the hyperparameters to use by modifying the dict in line 26-31 and you can set the number of models to train for each combination of hyperparameters in line 32:
```python
# Hyperparameter Configuration
hyperparameters = {
	'batch_size': [8, 16],
	'epochs': [2, 3, 4],
	'learning_rate': [1e-5, 2e-5, 5e-5],
	'warmup_ratio': [0.06],
}
executions_per_trial = 10
```

The following keys in the `hyperparameters` dict can be set to a list of values to try:

| Hyperparameter  | Description                                                                                                                                         |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `batch_size`    | number of samples per gradient update, passed to [Model.fit(...)](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)                    |
| `epochs`        | number of epochs to train the model, passed to [Model.fit(...)](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)                      |
| `learning_rate` | maximum learning rate for the slanted triangular learning rate                                                                                      |
| `warmup_ratio`  | fraction of iterations the slanted triangular learning rate increases                                                                               |
| `weight_decay`  | weight decay, passed to [AdamW(...)](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW)                                        |
| `adam_epsilon`  | small constant for numerical stability, passed to [AdamW(...)](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW)              |
| `adam_beta_1`   | exponential decay rate for the 1st moment estimates, passed to [AdamW(...)](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW) |
| `adam_beta_2`   | exponential decay rate for the 2nd moment estimates, passed to [AdamW(...)](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW) |

After running the script the following contents will be in a subfolder of the `TrainModel` folder:

| File                      | Description                                                                                                                                |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `best-model.h5`           | weights of the best model based on the MSE on the dev set (you can use the weights with the [code to apply the model](MakePredictions.py)) |
| `best-configuration.json` | hyperparameters of the best model                                                                                                          |
| `log.json`                | detailed log of hyperparameters and metrics for all epochs and trials                                                                      |
| `predictions.json`        | predictions of all models for the train and dev sets                                                                                       |
