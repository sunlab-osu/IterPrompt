---
title: "Old Documentation"
permalink: /docs/docs-old/
excerpt: "Setup and installation instructions for Minimal Mistakes 2.2 (deprecated)."
last_modified_at: 2020-05-02 17:59:15
toc: true
---


[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Downloads](https://pepy.tech/badge/simpletransformers)](https://pepy.tech/project/simpletransformers)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-25-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

# Simple Transformers

This library is based on the [Transformers](https://github.com/huggingface/transformers) library by HuggingFace. Simple Transformers lets you quickly train and evaluate Transformer models. Only 3 lines of code are needed to initialize a model, train the model, and evaluate a model.

Supports
- Sequence Classification
- Token Classification (NER)
- Question Answering
- Language Model Fine-Tuning
- Language Model Training
- Language Generation
- Multi-Modal Classification
- Conversational AI.

### Latest

ELECTRA models can now be used with Language Model Training, Token Classification, Sequence Classification, and Question Answering.

# Table of contents

<!--ts-->
- [Simple Transformers](#simple-transformers)
    - [Latest](#latest)
- [Table of contents](#table-of-contents)
  - [Setup](#setup)
    - [With Conda](#with-conda)
      - [Optional](#optional)
  - [Usage](#usage)
    - [Structure](#structure)
  - [Text Classification](#text-classification)
    - [Task Specific Notes](#task-specific-notes)
      - [Minimal Start for Binary Classification](#minimal-start-for-binary-classification)
      - [Minimal Start for Multiclass Classification](#minimal-start-for-multiclass-classification)
      - [Minimal Start for Multilabel Classification](#minimal-start-for-multilabel-classification)
        - [Special Attributes](#special-attributes)
      - [Minimal Start for Sentence Pair Classification](#minimal-start-for-sentence-pair-classification)
      - [Real Dataset Examples](#real-dataset-examples)
      - [ClassificationModel](#classificationmodel)
  - [Named Entity Recognition](#named-entity-recognition)
      - [Minimal Start](#minimal-start)
      - [Real Dataset Examples](#real-dataset-examples-1)
      - [NERModel](#nermodel)
  - [Question Answering](#question-answering)
    - [Data format](#data-format)
    - [Minimal Example](#minimal-example)
    - [Real Dataset Examples](#real-dataset-examples-2)
    - [QuestionAnsweringModel](#questionansweringmodel)
    - [Additional attributes for Question Answering tasks](#additional-attributes-for-question-answering-tasks)
      - [*doc_stride: int*](#docstride-int)
      - [*max_query_length: int*](#maxquerylength-int)
      - [*n_best_size: int*](#nbestsize-int)
      - [*max_answer_length: int*](#maxanswerlength-int)
      - [*null_score_diff_threshold: float*](#nullscorediffthreshold-float)
  - [Language Model Training](#language-model-training)
    - [Data format](#data-format-1)
    - [Minimal Example For Language Model Fine Tuning](#minimal-example-for-language-model-fine-tuning)
      - [Example (Medium Article)](#example-medium-article)
    - [Minimal Example For Language Model Training From Scratch](#minimal-example-for-language-model-training-from-scratch)
    - [Minimal Example For Language Model Training With ELECTRA](#minimal-example-for-language-model-training-with-electra)
    - [Real Dataset Example For Training a Language Model](#real-dataset-example-for-training-a-language-model)
    - [LanguageModelingModel](#languagemodelingmodel)
    - [Additional attributes for Language Modeling tasks](#additional-attributes-for-language-modeling-tasks)
      - [*dataset_type: str*](#datasettype-str)
      - [*dataset_class: Subclass of Pytorch Dataset*](#datasetclass-subclass-of-pytorch-dataset)
      - [*block_size: int*](#blocksize-int)
      - [*mlm: bool*](#mlm-bool)
      - [*mlm_probability: float*](#mlmprobability-float)
      - [*max_steps: int*](#maxsteps-int)
      - [*config_name: str*](#configname-str)
      - [*tokenizer_name: str*](#tokenizername-str)
      - [*min_frequencey: int*](#minfrequencey-int)
      - [*special_tokens: list*](#specialtokens-list)
      - [*sliding_window: bool*](#slidingwindow-bool)
      - [*stride: float*](#stride-float)
    - [*config: dict*](#config-dict)
    - [*generator_config: dict*](#generatorconfig-dict)
    - [*discriminator_config: dict*](#discriminatorconfig-dict)
  - [Language Generation](#language-generation)
      - [Minimal Start](#minimal-start-1)
      - [LanguageGenerationModel](#languagegenerationmodel)
    - [Additional attributes for Language Generation tasks](#additional-attributes-for-language-generation-tasks)
      - [*prompt: str*](#prompt-str)
      - [*length: int*](#length-int)
      - [*stop_token: str*](#stoptoken-str)
      - [*temperature: float*](#temperature-float)
      - [*repetition_penalty: float*](#repetitionpenalty-float)
      - [*k: int*](#k-int)
      - [*p: float*](#p-float)
      - [*padding_text: str*](#paddingtext-str)
      - [*xlm_language: str*](#xlmlanguage-str)
      - [*num_return_sequences: int*](#numreturnsequences-int)
      - [*config: dict*](#config-dict-1)
  - [Conversational AI](#conversational-ai)
    - [Data format](#data-format-2)
    - [Minimal Example](#minimal-example-1)
    - [Real Dataset Example](#real-dataset-example)
    - [ConvAIModel](#convaimodel)
    - [Additional attributes for Conversational AI](#additional-attributes-for-conversational-ai)
      - [*num_candidates: int*](#numcandidates-int)
      - [*personality_permutations: int*](#personalitypermutations-int)
      - [*max_history: int*](#maxhistory-int)
      - [*lm_coef: int*](#lmcoef-int)
      - [*mc_coef: int*](#mccoef-int)
      - [*no_sample: bool*](#nosample-bool)
      - [*max_length: int*](#maxlength-int)
      - [*min_length: int*](#minlength-int)
      - [*temperature: float*](#temperature-float-1)
      - [*top_k: float*](#topk-float)
      - [*top_p: float*](#topp-float)
  - [Multi-Modal Classification](#multi-modal-classification)
    - [Data format](#data-format-3)
      - [1 - Directory based](#1---directory-based)
      - [2 - Directory and file list](#2---directory-and-file-list)
      - [3 - Pandas DataFrame](#3---pandas-dataframe)
    - [Using custom names for column names or fields in JSON files](#using-custom-names-for-column-names-or-fields-in-json-files)
    - [Specifying the file type extension for image and text files](#specifying-the-file-type-extension-for-image-and-text-files)
    - [Label formats](#label-formats)
    - [Creating a Model](#creating-a-model)
    - [Training a Model](#training-a-model)
    - [Evaluating a Model](#evaluating-a-model)
    - [Predicting from a trained Model](#predicting-from-a-trained-model)
  - [Regression](#regression)
      - [Minimal Start for Regression](#minimal-start-for-regression)
  - [Visualization Support](#visualization-support)
  - [Experimental Features](#experimental-features)
    - [Sliding Window For Long Sequences](#sliding-window-for-long-sequences)
  - [Loading Saved Models](#loading-saved-models)
  - [Default Settings](#default-settings)
    - [Args Explained](#args-explained)
      - [*output_dir: str*](#outputdir-str)
      - [*cache_dir: str*](#cachedir-str)
      - [*best_model_dir: str*](#bestmodeldir-str)
      - [*fp16: bool*](#fp16-bool)
      - [*max_seq_length: int*](#maxseqlength-int)
      - [*train_batch_size: int*](#trainbatchsize-int)
      - [*gradient_accumulation_steps: int*](#gradientaccumulationsteps-int)
      - [*eval_batch_size: int*](#evalbatchsize-int)
      - [*num_train_epochs: int*](#numtrainepochs-int)
      - [*weight_decay: float*](#weightdecay-float)
      - [*learning_rate: float*](#learningrate-float)
      - [*adam_epsilon: float*](#adamepsilon-float)
      - [*max_grad_norm: float*](#maxgradnorm-float)
      - [*do_lower_case: bool*](#dolowercase-bool)
      - [*evaluate_during_training*](#evaluateduringtraining)
      - [*evaluate_during_training_steps*](#evaluateduringtrainingsteps)
      - [*evaluate_during_training_verbose*](#evaluateduringtrainingverbose)
      - [*use_cached_eval_features*](#usecachedevalfeatures)
      - [*save_eval_checkpoints*](#saveevalcheckpoints)
      - [*logging_steps: int*](#loggingsteps-int)
      - [*save_steps: int*](#savesteps-int)
      - [*no_cache: bool*](#nocache-bool)
      - [*save_model_every_epoch: bool*](#savemodeleveryepoch-bool)
      - [*tensorboard_dir: str*](#tensorboarddir-str)
      - [*overwrite_output_dir: bool*](#overwriteoutputdir-bool)
      - [*reprocess_input_data: bool*](#reprocessinputdata-bool)
      - [*process_count: int*](#processcount-int)
      - [*n_gpu: int*](#ngpu-int)
      - [*silent: bool*](#silent-bool)
      - [*use_multiprocessing: bool*](#usemultiprocessing-bool)
      - [*wandb_project: str*](#wandbproject-str)
      - [*wandb_kwargs: dict*](#wandbkwargs-dict)
      - [*use_early_stopping*](#useearlystopping)
      - [*early_stopping_patience*](#earlystoppingpatience)
      - [*early_stopping_delta*](#earlystoppingdelta)
      - [*early_stopping_metric*](#earlystoppingmetric)
      - [*early_stopping_metric_minimize*](#earlystoppingmetricminimize)
      - [*manual_seed*](#manualseed)
      - [*encoding*](#encoding)
      - [*config*](#config)
  - [Current Pretrained Models](#current-pretrained-models)
  - [Acknowledgements](#acknowledgements)
  - [Contributors ✨](#contributors-%e2%9c%a8)
<!--te-->

## Setup

### With Conda

1. Install Anaconda or Miniconda Package Manager from [here](https://www.anaconda.com/distribution/)
2. Create a new virtual environment and install packages.  
`conda create -n transformers python pandas tqdm`  
`conda activate transformers`  
If using cuda:  
&nbsp;&nbsp;&nbsp;&nbsp;`conda install pytorch cudatoolkit=10.1 -c pytorch`  
else:  
&nbsp;&nbsp;&nbsp;&nbsp;`conda install pytorch cpuonly -c pytorch`  

3. Install simpletransformers.  
`pip install simpletransformers` 

#### Optional

1. Install Weights and Biases (wandb) for tracking and visualizing training in a web browser.  
`pip install wandb`

## Usage

Most available hyperparameters are common for all tasks. Any special hyperparameters will be listed in the docs section for the corresponding class. See [Default Settings](#default-settings) and [Args Explained](#args-explained) sections for more information.

Example scripts can be found in the `examples` directory.

See the [Changelog](https://github.com/ThilinaRajapakse/simpletransformers/blob/master/CHANGELOG.md) for up-to-date changes to the project.

### Structure

_The file structure has been updated starting with version 0.6.0. This should only affect import statements. The old import paths should still be functional although it is recommended to use the updated paths given below and in the minimal start examples_.

* `simpletransformers.classification` - Includes all Classification models.
  * `ClassificationModel`
  * `MultiLabelClassificationModel`
* `simpletransformers.ner` - Includes all Named Entity Recognition models.
  * `NERModel`
* `simpletransformers.question_answering` - Includes all Question Answering models.
  * `QuestionAnsweringModel`


_[Back to Table of Contents](#table-of-contents)_

---

## Text Classification

Supports Binary Classification, Multiclass Classification, and Multilabel Classification.

Supported model types:

* ALBERT
* BERT
* CamemBERT
* RoBERTa
* DistilBERT
* ELECTRA
* FlauBERT
* XLM
* XLM-RoBERTa
* XLNet

### Task Specific Notes

* Set `'sliding_window': True` in `args` to prevent text being truncated. The default *stride* is `'stride': 0.8` which is `0.8 * max_seq_length`. Training text will be split using a sliding window and each window will be assigned the label from the original text. During evaluation and prediction, the mode of the predictions for each window will be the final prediction on each sample. The `tie_value` (default `1`) will be used in the case of a tie.  
*Currently not available for Multilabel Classification*

#### Minimal Start for Binary Classification

```python
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0]]
train_df = pd.DataFrame(train_data)

eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0]]
eval_df = pd.DataFrame(eval_data)

# Create a ClassificationModel
model = ClassificationModel('roberta', 'roberta-base') # You can set class weights by using the optional weight argument

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
```

If you wish to add any custom metrics, simply pass them as additional keyword arguments. The keyword is the name to be given to the metric, and the value is the function that will calculate the metric. Make sure that the function expects two parameters with the first one being the true label, and the second being the predictions. (This is the default for sklearn metrics)

```python
import sklearn


result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
```


To make predictions on arbitary data, the `predict(to_predict)` function can be used. For a list of text, it returns the model predictions and the raw model outputs.

```python
predictions, raw_outputs = model.predict(['Some arbitary sentence'])
```

#### Minimal Start for Multiclass Classification

For multiclass classification, simply pass in the number of classes to the `num_labels` optional parameter of `ClassificationModel`.

```python
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column is the text with type str, and the second column in the label with type int.
train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0], ['Example eval senntence belonging to class 2', 2]]
train_df = pd.DataFrame(train_data)

eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0], ['Example eval senntence belonging to class 2', 2]]
eval_df = pd.DataFrame(eval_data)

# Create a ClassificationModel
model = ClassificationModel('bert', 'bert-base-cased', num_labels=3, args={'reprocess_input_data': True, 'overwrite_output_dir': True}) 
# You can set class weights by using the optional weight argument

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

predictions, raw_outputs = model.predict(["Some arbitary sentence"])
```

#### Minimal Start for Multilabel Classification

For Multi-Label Classification, the labels should be multi-hot encoded. The number of classes can be specified (default is 2) by passing it to the `num_labels` optional parameter of `MultiLabelClassificationModel`.

_Warning: Pandas can cause issues when saving and loading lists stored in a column. Check whether your list has been converted to a String!_

The default evaluation metric used is Label Ranking Average Precision ([LRAP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html)) Score.

```python
from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
train_data = [['Example sentence 1 for multilabel classification.', [1, 1, 1, 1, 0, 1]]], [['This is another example sentence. ', [0, 1, 1, 0, 0, 0]]]
train_df = pd.DataFrame(train_data, columns=['text', 'labels'])

eval_data = [['Example eval sentence for multilabel classification.', [1, 1, 1, 1, 0, 1]], ['Example eval senntence belonging to class 2', [0, 1, 1, 0, 0, 0]]]
eval_df = pd.DataFrame(eval_data)

# Create a MultiLabelClassificationModel
model = MultiLabelClassificationModel('roberta', 'roberta-base', num_labels=6, args={'reprocess_input_data': True, 'overwrite_output_dir': True, 'num_train_epochs': 5})
# You can set class weights by using the optional weight argument
print(train_df.head())

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print(result)
print(model_outputs)

predictions, raw_outputs = model.predict(['This thing is entirely different from the other thing. '])
print(predictions)
print(raw_outputs)
```

##### Special Attributes

* The args dict of `MultiLabelClassificationModel` has an additional `threshold` parameter with default value 0.5. The threshold is the value at which a given label flips from 0 to 1 when predicting. The `threshold` may be a single value or a list of value with the same length as the number of labels. This enables the use of seperate threshold values for each label.
* `MultiLabelClassificationModel` takes in an additional optional argument `pos_weight`. This should be a list with the same length as the number of labels. This enables using different weights for each label when calculating loss during training and evaluation.

#### Minimal Start for Sentence Pair Classification

* Training and evaluation Dataframes must contain a `text_a`, `text_b`, and a `labels` column.
* The `predict()` function expects a list of lists in the format below. A single sample input should also be a list of lists like `[[text_a, text_b]]`.

```python
[
    [sample_1_text_a, sample_1_text_b],
    [sample_2_text_a, sample_2_text_b],
    [sample_3_text_a, sample_3_text_b],
    # More samples
]
```

```python
from simpletransformers.classification import ClassificationModel
import pandas as pd
import sklearn
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data = [
    ['Example sentence belonging to class 1', 'Yep, this is 1', 1],
    ['Example sentence belonging to class 0', 'Yep, this is 0', 0],
    ['Example  2 sentence belonging to class 0', 'Yep, this is 0', 0]
]

train_df = pd.DataFrame(train_data, columns=['text_a', 'text_b', 'labels'])

eval_data = [
    ['Example sentence belonging to class 1', 'Yep, this is 1', 1],
    ['Example sentence belonging to class 0', 'Yep, this is 0', 0],
    ['Example  2 sentence belonging to class 0', 'Yep, this is 0', 0]
]

eval_df = pd.DataFrame(eval_data, columns=['text_a', 'text_b', 'labels'])

train_args={
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs': 3,
}

# Create a ClassificationModel
model = ClassificationModel('roberta', 'roberta-base', num_labels=2, use_cuda=True, cuda_device=0, args=train_args)
print(train_df.head())

# Train the model
model.train_model(train_df, eval_df=eval_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)

predictions, raw_outputs = model.predict([["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
print(predictions)
print(raw_outputs)
```

#### Real Dataset Examples

* [Yelp Reviews Dataset - Binary Classification](https://towardsdatascience.com/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3?source=friends_link&sk=40726ceeadf99e1120abc9521a10a55c)
* [AG News Dataset - Multiclass Classification](https://medium.com/swlh/simple-transformers-multi-class-text-classification-with-bert-roberta-xlnet-xlm-and-8b585000ce3a?source=friends_link&sk=90e1c97255b65cedf4910a99041d9dfc)
* [Toxic Comments Dataset - Multilabel Classification](https://towardsdatascience.com/multi-label-classification-using-bert-roberta-xlnet-xlm-and-distilbert-with-simple-transformers-b3e0cda12ce5?source=friends_link&sk=354e688fe238bfb43e9a575216816219)
* [Semantic Textual Similarity Benchmark - Sentence Pair](https://medium.com/@chaturangarajapakshe/solving-sentence-pair-tasks-using-simple-transformers-2496fe79d616?source=friends_link&sk=fbf7439e9c31f7aefa1613d423a0fd40)
* [AG News Dataset - BERT (base and distilled), RoBERTa (base and distilled), and XLNet compared](https://towardsdatascience.com/to-distil-or-not-to-distil-bert-roberta-and-xlnet-c777ad92f8?source=friends_link&sk=6a3c7940b18066ded94aeee95e354ed1)


#### ClassificationModel

`class simpletransformers.classification.ClassificationModel (model_type, model_name, args=None, use_cuda=True)`  
This class  is used for Text Classification tasks.

`Class attributes`
* `tokenizer`: The tokenizer to be used.
* `model`: The model to be used.
* `model_name`: model_name: Default Transformer model name or path to Transformer model file (pytorch_model.bin).
* `device`: The device on which the model will be trained and evaluated.
* `results`: A python dict of past evaluation results for the TransformerModel object.
* `args`: A python dict of arguments used for training and evaluation.
- `cuda_device`: (optional) int - Default = -1. Used to specify which GPU should be used.

`Parameters`
* `model_type`: (required) str - The type of model to use. Currently, BERT, XLNet, XLM, and RoBERTa models are available.
* `model_name`: (required) str - The exact model to use. Could be a pretrained model name or path to a directory containing a model. See [Current Pretrained Models](#current-pretrained-models) for all available models.
* `num_labels` (optional): The number of labels or classes in the dataset.
* `weight` (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
* `args`: (optional) python dict - A dictionary containing any settings that should be overwritten from the default values.
* `use_cuda`: (optional) bool - Default = True. Flag used to indicate whether CUDA should be used.

`class methods`  
**`train_model(self, train_df, output_dir=None, show_running_loss=True, args=None, eval_df=None)`**

Trains the model using 'train_df'

Args:  
* `train_df`: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.

* `output_dir` (optional): The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

* `args` (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.

* show_running_loss (optional): Set to False to disable printing running training loss to the terminal.

* `eval_df` (optional): A DataFrame against which evaluation will be performed when `evaluate_during_training` is enabled. Is required if `evaluate_during_training` is enabled.

* `**kwargs`: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

Returns:  
* None

**`eval_model(self, eval_df, output_dir=None, verbose=False)`**

Evaluates the model on eval_df. Saves results to output_dir.

Args:  
* eval_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.

* output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.  

* verbose: If verbose, results will be printed to the console on completion of evaluation.  

* silent: If silent, tqdm progress bars will be hidden.

* `**kwargs`: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

Returns:  
* result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)  

* model_outputs: List of model outputs for each row in eval_df  

* wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model  

**`predict(self, to_predict)`**

Performs predictions on a list of text.

Args:
* to_predict: A python list of text (str) to be sent to the model for prediction.

Returns:
* preds: A python list of the predictions (0 or 1) for each text.  
* model_outputs: A python list of the raw model outputs for each text.

If `config: {"output_hidden_states": True}`, two additional values will be returned.
* all_embedding_outputs: Numpy array of shape *(batch_size, sequence_length, hidden_size)*
* all_layer_hidden_states: Numpy array of shape *(num_hidden_layers, batch_size, sequence_length, hidden_size)*


**`train(self, train_dataset, output_dir)`**

Trains the model on train_dataset.
*Utility function to be used by the train_model() method. Not intended to be used directly.*

**`evaluate(self, eval_df, output_dir, prefix="")`**

Evaluates the model on eval_df.
*Utility function to be used by the eval_model() method. Not intended to be used directly*

**`load_and_cache_examples(self, examples, evaluate=False)`**

Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
*Utility function for train() and eval() methods. Not intended to be used directly*

**`compute_metrics(self, preds, labels, eval_examples, **kwargs):`**

Computes the evaluation metrics for the model predictions.

Args:
* preds: Model predictions  

* labels: Ground truth labels  

* eval_examples: List of examples on which evaluation was performed  

Returns:
* result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)  

* wrong: List of InputExample objects corresponding to each incorrect prediction by the model  

_[Back to Table of Contents](#table-of-contents)_

---

## Named Entity Recognition

This section describes how to use Simple Transformers for Named Entity Recognition. (If you are updating from a Simple Transformers before 0.5.0, note that `seqeval` needs to be installed to perform NER.)

*This model can also be used for any other NLP task involving token level classification. Make sure you pass in your list of labels to the model if they are different from the defaults.*

Supported model types:
* BERT
* CamemBERT
* DistilBERT
* ELECTRA
* RoBERTa
* XLM-RoBERTa

```python
model = NERModel('bert', 'bert-base-cased', labels=["LABEL_1", "LABEL_2", "LABEL_3"])
```

#### Minimal Start

```python
from simpletransformers.ner import NERModel
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Creating train_df  and eval_df for demonstration
train_data = [
    [0, 'Simple', 'B-MISC'], [0, 'Transformers', 'I-MISC'], [0, 'started', 'O'], [1, 'with', 'O'], [0, 'text', 'O'], [0, 'classification', 'B-MISC'],
    [1, 'Simple', 'B-MISC'], [1, 'Transformers', 'I-MISC'], [1, 'can', 'O'], [1, 'now', 'O'], [1, 'perform', 'O'], [1, 'NER', 'B-MISC']
]
train_df = pd.DataFrame(train_data, columns=['sentence_id', 'words', 'labels'])

eval_data = [
    [0, 'Simple', 'B-MISC'], [0, 'Transformers', 'I-MISC'], [0, 'was', 'O'], [1, 'built', 'O'], [1, 'for', 'O'], [0, 'text', 'O'], [0, 'classification', 'B-MISC'],
    [1, 'Simple', 'B-MISC'], [1, 'Transformers', 'I-MISC'], [1, 'then', 'O'], [1, 'expanded', 'O'], [1, 'to', 'O'], [1, 'perform', 'O'], [1, 'NER', 'B-MISC']
]
eval_df = pd.DataFrame(eval_data, columns=['sentence_id', 'words', 'labels'])

# Create a NERModel
model = NERModel('bert', 'bert-base-cased', args={'overwrite_output_dir': True, 'reprocess_input_data': True})

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, predictions = model.eval_model(eval_df)

# Predictions on arbitary text strings
predictions, raw_outputs = model.predict(["Some arbitary sentence"])

print(predictions)
```

#### Real Dataset Examples

* [CoNLL Dataset Example](https://towardsdatascience.com/simple-transformers-named-entity-recognition-with-transformer-models-c04b9242a2a0?source=friends_link&sk=e8b98c994173cd5219f01e075727b096)

#### NERModel

`class simpletransformers.ner.ner_model.NERModel (model_type, model_name, labels=None, args=None, use_cuda=True)`  
This class  is used for Named Entity Recognition.

`Class attributes`
* `tokenizer`: The tokenizer to be used.
* `model`: The model to be used.
            model_name: Default Transformer model name or path to Transformer model file (pytorch_model.bin).
* `device`: The device on which the model will be trained and evaluated.
* `results`: A python dict of past evaluation results for the TransformerModel object.
* `args`: A python dict of arguments used for training and evaluation.
- `cuda_device`: (optional) int - Default = -1. Used to specify which GPU should be used.

`Parameters`
* `model_type`: (required) str - The type of model to use. Currently, BERT, XLNet, XLM, and RoBERTa models are available.
* `model_name`: (required) str - The exact model to use. Could be a pretrained model name or path to a directory containing a model. See [Current Pretrained Models](#current-pretrained-models) for all available models.
* `labels` (optional): A list of all Named Entity labels.  If not given, ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"] will be used.
* `args`: (optional) python dict - A dictionary containing any settings that should be overwritten from the default values.
* `use_cuda`: (optional) bool - Default = True. Flag used to indicate whether CUDA should be used.

`class methods`  
**`train_model(self, train_data, output_dir=None, args=None, eval_df=None)`**

Trains the model using 'train_data'

Args:  
* train_data: train_data should be the path to a .txt file containing the training data OR a pandas DataFrame with 3 columns.
If a text file is used the data should be in the CoNLL format. i.e. One word per line, with sentences seperated by an empty line. 
The first word of the line should be a word, and the last should be a Name Entity Tag.
If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.

* output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

* show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.

* args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.

* eval_df (optional): A DataFrame against which evaluation will be performed when `evaluate_during_training` is enabled. Is required if `evaluate_during_training` is enabled.


Returns:  
* None

**`eval_model(self, eval_data, output_dir=None, verbose=True)`**

Evaluates the model on eval_data. Saves results to output_dir.

Args:  
* eval_data: eval_data should be the path to a .txt file containing the evaluation data or a pandas DataFrame. If a text file is used the data should be in the CoNLL format. I.e. One word per line, with sentences seperated by an empty line. The first word of the line should be a word, and the last should be a Name Entity Tag. If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.

* output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.  

* verbose: If verbose, results will be printed to the console on completion of evaluation.  

Returns:  
* result: Dictionary containing evaluation results. (eval_loss, precision, recall, f1_score)

* model_outputs: List of raw model outputs

* preds_list: List of predicted tags

**`predict(self, to_predict)`**

Performs predictions on a list of text.

Args:
* to_predict: A python list of text (str) to be sent to the model for prediction.

Returns:
* preds: A Python list of lists with dicts containg each word mapped to its NER tag.
* model_outputs: A python list of the raw model outputs for each text.


**`train(self, train_dataset, output_dir)`**

Trains the model on train_dataset.
*Utility function to be used by the train_model() method. Not intended to be used directly.*

**`evaluate(self, eval_dataset, output_dir, prefix="")`**

Evaluates the model on eval_dataset.
*Utility function to be used by the eval_model() method. Not intended to be used directly*

**`load_and_cache_examples(self, data, evaluate=False, no_cache=False, to_predict=None)`**

Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
*Utility function for train() and eval() methods. Not intended to be used directly*

_[Back to Table of Contents](#table-of-contents)_

---

## Question Answering

Supported model types:

* ALBERT
* BERT
* DistilBERT
* ELECTRA
* XLM
* XLNet

### Data format

For question answering tasks, the input data can be in JSON files or in a Python list of dicts in the correct format.

The file should contain a single list of dictionaries. A dictionary represents a single context and its associated questions.

Each such dictionary contains two attributes, the `"context"` and `"qas"`.
* `context`: The paragraph or text from which the question is asked.
* `qas`: A list of questions and answers.

Questions and answers are represented as dictionaries. Each dictionary in `qas` has the following format.
* `id`: (string) A unique ID for the question. Should be unique across the entire dataset.
* `question`: (string) A question. 
* `is_impossible`: (bool) Indicates whether the question can be answered correctly from the context. 
* `answers`: (list) The list of correct answers to the question.

A single answer is represented by a dictionary with the following attributes.
* `answer`: (string) The answer to the question. Must be a substring of the context.
* `answer_start`: (int) Starting index of the answer in the context.

### Minimal Example

```python
from simpletransformers.question_answering import QuestionAnsweringModel
import json
import os
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Create dummy data to use for training.
train_data = [
    {
        'context': "This is the first context",
        'qas': [
            {
                'id': "00001",
                'is_impossible': False,
                'question': "Which context is this?",
                'answers': [
                    {
                        'text': "the first",
                        'answer_start': 8
                    }
                ]
            }
        ]
    },
    {
        'context': "Other legislation followed, including the Migratory Bird Conservation Act of 1929, a 1937 treaty prohibiting the hunting of right and gray whales,
            and the Bald Eagle Protection Act of 1940. These later laws had a low cost to society—the species were relatively rare—and little opposition was raised",
        'qas': [
            {
                'id': "00002",
                'is_impossible': False,
                'question': "What was the cost to society?",
                'answers': [
                    {
                        'text': "low cost",
                        'answer_start': 225
                    }
                ]
            },
            {
                'id': "00003",
                'is_impossible': False,
                'question': "What was the name of the 1937 treaty?",
                'answers': [
                    {
                        'text': "Bald Eagle Protection Act",
                        'answer_start': 167
                    }
                ]
            }
        ]
    }
]

# Save as a JSON file
os.makedirs('data', exist_ok=True)
with open('data/train.json', 'w') as f:
    json.dump(train_data, f)


# Create the QuestionAnsweringModel
model = QuestionAnsweringModel('distilbert', 'distilbert-base-uncased-distilled-squad', args={'reprocess_input_data': True, 'overwrite_output_dir': True})

# Train the model with JSON file
model.train_model('data/train.json')

# The list can also be used directly
# model.train_model(train_data)

# Evaluate the model. (Being lazy and evaluating on the train data itself)
result, text = model.eval_model('data/train.json')

print(result)
print(text)

print('-------------------')

# Making predictions using the model.
to_predict = [{'context': 'This is the context used for demonstrating predictions.', 'qas': [{'question': 'What is this context?', 'id': '0'}]}]

print(model.predict(to_predict))
```

### Real Dataset Examples

* [SQuAD 2.0 - Question Answering](https://towardsdatascience.com/question-answering-with-bert-xlnet-xlm-and-distilbert-using-simple-transformers-4d8785ee762a?source=friends_link&sk=e8e6f9a39f20b5aaf08bbcf8b0a0e1c2)

### QuestionAnsweringModel

`class simpletransformers.question_answering.QuestionAnsweringModel (model_type, model_name, args=None, use_cuda=True)`  
This class is used for Question Answering tasks.

`Class attributes`
- `tokenizer`: The tokenizer to be used.
- `model`: The model to be used.
            model_name: Default Transformer model name or path to Transformer model file (pytorch_model.bin).
- `device`: The device on which the model will be trained and evaluated.
- `results`: A python dict of past evaluation results for the TransformerModel object.
- `args`: A python dict of arguments used for training and evaluation.
- `cuda_device`: (optional) int - Default = -1. Used to specify which GPU should be used.

`Parameters`
- `model_type`: (required) str - The type of model to use.
- `model_name`: (required) str - The exact model to use. Could be a pretrained model name or path to a directory containing a model. See [Current Pretrained Models](#current-pretrained-models) for all available models.
- `args`: (optional) python dict - A dictionary containing any settings that should be overwritten from the default values.
- `use_cuda`: (optional) bool - Default = True. Flag used to indicate whether CUDA should be used.

`class methods`  
**`train_model(self, train_df, output_dir=None, args=None, eval_df=None)`**

Trains the model using 'train_file'

Args:  

- `train_df`: Path to JSON file containing training data. The model will be trained on this file.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

- `output_dir` (optional): The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

- `show_running_loss` (Optional): Set to False to prevent training loss being printed.

- `args` (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.

- `eval_file` (optional): Path to JSON file containing evaluation data against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.

- `**kwargs`: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
    A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

Returns:

- None

**`eval_model(self, eval_df, output_dir=None, verbose=False)`**

Evaluates the model on eval_file. Saves results to output_dir.

Args:  

- `eval_file`: Path to JSON file containing evaluation data. The model will be evaluated on this file.

- `output_dir`: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.  

- `verbose`: If verbose, results will be printed to the console on completion of evaluation.  

- `**kwargs`: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
    A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

Returns:  

- `result`: Dictionary containing evaluation results. (correct, similar, incorrect)

- `text`: A dictionary containing the 3 dictionaries correct_text, similar_text (the predicted answer is a substring of the correct answer or vise versa), incorrect_text.

**`predict(self, to_predict)`**

Performs predictions on a list of text.

Args:

- `to_predict`: A python list of python dicts containing contexts and questions to be sent to the model for prediction.

```python
E.g: predict([
    {
        'context': "Some context as a demo",
        'qas': [
            {'id': '0', 'question': 'What is the context here?'},
            {'id': '1', 'question': 'What is this for?'}
        ]
    }
])
```

- `n_best_size` (Optional): Number of predictions to return. args['n_best_size'] will be used if not specified.

Returns:

- `preds`: A python list containg the predicted answer, and id for each question in to_predict.

**`train(self, train_dataset, output_dir, show_running_loss=True, eval_file=None)`**

Trains the model on train_dataset.
*Utility function to be used by the train_model() method. Not intended to be used directly.*

**`evaluate(self, eval_df, output_dir, , verbose=False)`**

Evaluates the model on eval_df.
*Utility function to be used by the eval_model() method. Not intended to be used directly*

**`load_and_cache_examples(self, examples, evaluate=False, no_cache=False, output_examples=False)`**

Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
*Utility function for train() and eval() methods. Not intended to be used directly*

### Additional attributes for Question Answering tasks

QuestionAnsweringModel has a few additional attributes in its `args` dictionary, given below with their default values.

```python
  'doc_stride': 384,
  'max_query_length': 64,
  'n_best_size': 20,
  'max_answer_length': 100,
  'null_score_diff_threshold': 0.0
```

#### *doc_stride: int*

When splitting up a long document into chunks, how much stride to take between chunks.

#### *max_query_length: int*

Maximum token length for questions. Any questions longer than this will be truncated to this length.

#### *n_best_size: int*

The number of predictions given per question.

#### *max_answer_length: int*

The maximum token length of an answer that can be generated.

#### *null_score_diff_threshold: float*

If null_score - best_non_null is greater than the threshold predict null.

_[Back to Table of Contents](#table-of-contents)_

---

## Language Model Training

Supported model types:

- BERT
- CamemBERT
- DistilBERT
- ELECTRA
- GPT-2
- OpenAI-GPT
- RoBERTa

### Data format

The data should simply be placed in a text file. E.g.: [WikiText-2](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)

### Minimal Example For Language Model Fine Tuning

The minimal example given below assumes that you have downloaded the WikiText-2 dataset.

```python
from simpletransformers.language_modeling import LanguageModelingModel
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
}

model = LanguageModelingModel('bert', 'bert-base-cased', args=train_args)

model.train_model("wikitext-2/wiki.train.tokens", eval_file="wikitext-2/wiki.test.tokens")

model.eval_model("wikitext-2/wiki.test.tokens")

```

#### Example (Medium Article)

- [Language Model Fine-tuning](https://medium.com/skilai/language-model-fine-tuning-for-pre-trained-transformers-b7262774a7ee?source=friends_link&sk=1f9f834447db7e748ae333c6490064fa)

### Minimal Example For Language Model Training From Scratch

You can use any text file/files for training a new language model. Setting `model_name` to `None` will indicate that the language model should be trained from scratch.

Required for Language Model Training From Scratch:

- `train_files` must be specifief when creating the `LanguagModelingModel`. This may be a path to a single file or a list of paths to multiple files.
- `vocab_size` (in args dictionary)

```python
from simpletransformers.language_modeling import LanguageModelingModel
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "vocab_size": 52000,
}

model = LanguageModelingModel('roberta', None, args=train_args)

model.train_model("wikitext-2/wiki.train.tokens", eval_file="wikitext-2/wiki.test.tokens")

model.eval_model("wikitext-2/wiki.test.tokens")

```

### Minimal Example For Language Model Training With ELECTRA

[ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) is a new approach to pretraining Transformer Language Models. This method is comparatively less compute-intensive.

You can use the `save_discriminator()` and `save_generator()` methods to extract the pretrained models. The two models will be saved to `<output_dir>/discriminator_model` and `<output_dir>/generator_model` by default.

```python
from simpletransformers.language_modeling import LanguageModelingModel
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "vocab_size": 52000,
}

model = LanguageModelingModel('electra', None, args=train_args, train_files="wikitext-2/wiki.train.tokens")

# Mixing standard ELECTRA architectures example
# model = LanguageModelingModel(
#     "electra",
#     None,
#     generator_name="google/electra-small-generator",
#     discriminator_name="google/electra-large-discriminator",
#     args=train_args,
#     train_files="wikitext-2/wiki.train.tokens",
# )

model.train_model("wikitext-2/wiki.train.tokens", eval_file="wikitext-2/wiki.test.tokens")

model.eval_model("wikitext-2/wiki.test.tokens")

```

### Real Dataset Example For Training a Language Model

- [Esparanto Model trained with ELECTRA](https://medium.com/@chaturangarajapakshe/understanding-electra-and-training-an-electra-language-model-3d33e3a9660d?source=friends_link&sk=2b4b4a79954e3d7c84ab863efaea8c65)

### LanguageModelingModel

`class simpletransformers.language_modeling.LanguageModelingModel (model_type, model_name, generator_name=None, discriminator_name=None, args=None, use_cuda=True, cuda_device=-1)`  
This class is used for Question Answering tasks.

`Class attributes`

- `tokenizer`: The tokenizer to be used.
- `model`: The model to be used.
            model_name: Default Transformer model name or path to Transformer model file (pytorch_model.bin).
- `device`: The device on which the model will be trained and evaluated.
- `results`: A python dict of past evaluation results for the TransformerModel object.
- `args`: A python dict of arguments used for training and evaluation.
- `cuda_device`: (optional) int - Default = -1. Used to specify which GPU should be used.

`Parameters`

- `model_type`: (required) str - The type of model to use.
- `model_name`: (required) str - The exact model to use. Could be a pretrained model name or path to a directory containing a model. See [Current Pretrained Models](#current-pretrained-models) for all available models. Set to `None` for language model training from scratch.
- `generator_name`: (optional) A pretrained model name or path to a directory containing an ELECTRA generator model.
- `discriminator_name`: (optional) A pretrained model name or path to a directory containing an ELECTRA discriminator model.
- `args`: (optional) python dict - A dictionary containing any settings that should be overwritten from the default values.
- `train_files`: (optional) List of files to be used when training the tokenizer.
- `use_cuda`: (optional) bool - Default = True. Flag used to indicate whether CUDA should be used.
- `cuda_device`: (optional) Specific GPU that should be used. Will use the first available GPU by default.

`class methods`  
**`train_model(self, train_file, output_dir=None, show_running_loss=True, args=None, eval_file=None, verbose=True,)`**

Trains the model using 'train_file'

Args:  

- `train_file`: Path to text file containing the text to train the language model on.

- `output_dir` (optional): The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

- `show_running_loss` (Optional): Set to False to prevent training loss being printed.

- `args` (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.

- `eval_file` (optional): Path to eval file containing the text to evaluate the language model on. Is required if evaluate_during_training is enabled.

Returns:

- None

**`eval_model(self, eval_file, output_dir=None, verbose=True, silent=False,)`**

Evaluates the model on eval_file. Saves results to output_dir.

Args:  

- `eval_file`: Path to eval file containing the text to evaluate the language model on.

- `output_dir` (optional): The directory where model files will be saved. If not given, self.args['output_dir'] will be used.  

- `verbose`: If verbose, results will be printed to the console on completion of evaluation.  

- `silent`: If silent, tqdm progress bars will be hidden.

Returns:  

- `result`: Dictionary containing evaluation results. (correct, similar, incorrect)

- `text`: A dictionary containing the 3 dictionaries correct_text, similar_text (the predicted answer is a substring of the correct answer or vise versa), incorrect_text.

**`train_tokenizer(self, train_files, tokenizer_name=None, output_dir=None, use_trained_tokenizer=True)`

Train a new tokenizer on `train_files`.

Args:

- `train_files`: List of files to be used when training the tokenizer.

- `tokenizer_name`: Name of a pretrained tokenizer or a path to a directory containing a tokenizer.

- `output_dir` (optional): The directory where model files will be saved. If not given, self.args['output_dir'] will be used.  

- `use_trained_tokenizer` (optional): Load the trained tokenizer once training completes.

Returns: None

**`train(self, train_dataset, output_dir, show_running_loss=True, eval_file=None)`**

Trains the model on train_dataset.
*Utility function to be used by the train_model() method. Not intended to be used directly.*

**`evaluate(self, eval_dataset, output_dir, , verbose=False)`**

Evaluates the model on eval_dataset.
*Utility function to be used by the eval_model() method. Not intended to be used directly*

**`load_and_cache_examples(self, examples, evaluate=False, no_cache=False, output_examples=False)`**

Reads a text file from file_path and creates training features.
*Utility function for train() and eval() methods. Not intended to be used directly*

### Additional attributes for Language Modeling tasks

LanguageModelingModel has a few additional attributes in its `args` dictionary, given below with their default values.

```python
    "dataset_type": "None",
    "dataset_class": None,
    "custom_tokenizer": None,
    "block_size": 512,
    "mlm": True,
    "mlm_probability": 0.15,
    "max_steps": -1,
    "config_name": None,
    "tokenizer_name": None,
    "min_frequency": 2,
    "special_tokens": ["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    "sliding_window": False,
    "stride": 0.8
    "config": {},
    "generator_config": {},
    "discriminator_config": {},
```

#### *dataset_type: str*

Used to specify the Dataset type to be used. The choices are given below.

- `simple` (or None) - Each line in the train files are considered to be a single, separate sample. `sliding_window` can be set to True to 
automatically split longer sequences into samples of length `max_seq_length`. Uses multiprocessing for significantly improved performance on multicore systems.

- `line_by_line` - Treats each line in the train files as a seperate sample.
  
- `text` - Treats each file in `train_files` as a seperate sample.

*Using `simple` is recommended.*

#### *dataset_class: Subclass of Pytorch Dataset*

A custom dataset class to use.

#### *block_size: int*

Optional input sequence length after tokenization.
The training dataset will be truncated in block of this size for training.
Default to the model max input length for single sentence inputs (take into account special tokens).

#### *mlm: bool*

Train with masked-language modeling loss instead of language modeling

#### *mlm_probability: float*

Ratio of tokens to mask for masked language modeling loss

#### *max_steps: int*

If > 0: set total number of training steps to perform. Override num_train_epochs.

#### *config_name: str*

Name of pretrained config or path to a directory containing a `config.json` file.

#### *tokenizer_name: str*

Name of pretrained tokenizer or path to a directory containing tokenizer files.

#### *min_frequencey: int*

Minimum frequency required for a word to be added to the vocabulary.

#### *special_tokens: list*

List of special tokens to be used when training a new tokenizer.

#### *sliding_window: bool*

Whether sliding window technique should be used when preparing data. *Only works with SimpleDataset.*

#### *stride: float*

A fraction of the `max_seq_length` to use as the stride when using a sliding window

### *config: dict*

Key-values given here will override the default values used in a model `Config`.

### *generator_config: dict*

Key-values given here will override the default values used in an Electra generator model `Config`.

### *discriminator_config: dict*

Key-values given here will override the default values used in an Electra discriminator model `Config`.

_[Back to Table of Contents](#table-of-contents)_

---

## Language Generation

This section describes how to use Simple Transformers for Langauge Generation.

Supported model types:

* CTRL
* GPT-2
* OpenAI-GPT
* Transformer-XL
* XLM
* XLNet

#### Minimal Start

```python
import logging
from simpletransformers.language_generation import LanguageGenerationModel


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model = LanguageGenerationModel("gpt2", "gpt2")
model.generate("Let's give a minimal start to the model like")
```

#### LanguageGenerationModel

`class simpletransformers.language_generation.language_generation_model.LanguageGenerationModel (self, model_type, model_name, args=None, use_cuda=True, cuda_device=-1, **kwargs)`  
This class  is used for Named Entity Recognition.

`Class attributes`
* `tokenizer`: The tokenizer to be used.
* `model`: The model to be used.
            model_name: Default Transformer model name or path to Transformer model file (pytorch_model.bin).
* `device`: The device on which the model will be trained and evaluated.
* `args`: A python dict of arguments used for training and evaluation.
- `cuda_device`: (optional) int - Default = -1. Used to specify which GPU should be used.

`Parameters`
* `model_type`: (required) str - The type of model to use.

* `model_name`: (required) str - The exact model to use. Could be a pretrained model name or path to a directory containing a model. See [Current Pretrained Models](#current-pretrained-models) for all available models.

* `args`: (optional) python dict - A dictionary containing any settings that should be overwritten from the default values.

* `use_cuda`: (optional) bool - Default = True. Flag used to indicate whether CUDA should be used.

* `cuda_device` (optional): Specific GPU that should be used. Will use the first available GPU by default.

* `**kwargs` (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.


`class methods`  

**`generate(self, prompt=None, args=None, verbose=True)`**

Generate text using a `LanguageGenerationModel`

Args:

* `prompt` (optional): A prompt text for the model. If given, will override args["prompt"]

* `args` (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.

* `verbose` (optional): If verbose, generated text will be logged to the console.


Returns:

* `generated_sequences`: Sequences of text generated by the model.

### Additional attributes for Language Generation tasks

LanguageGenerationModel has a few additional attributes in its `args` dictionary, given below with their default values.

```python
    "prompt": "",
    "length": 20,
    "stop_token": None,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "k": 0,
    "p": 0.9,
    "padding_text": "",
    "xlm_language": "",
    "num_return_sequences": 1,
    "config_name": None,
    "tokenizer_name": None,
```

#### *prompt: str*

A prompt text for the model.

#### *length: int*

Length of the text to generate

#### *stop_token: str*

Token at which text generation is stopped

#### *temperature: float*

Temperature of 1.0 is the default. Lowering this makes the sampling *greedier*

#### *repetition_penalty: float*

Primarily useful for CTRL model; in that case, use 1.2

#### *k: int*

*k* value for top-k sampling

#### *p: float*

*p* value for top-p (nucleus) sampling

#### *padding_text: str*

Padding text for Transfo-XL and XLNet.

#### *xlm_language: str*

Optional language when used with the XLM model.

#### *num_return_sequences: int*

The number of samples to generate.

#### *config: dict*
Key-values given here will override the default values used in a model Config.


_[Back to Table of Contents](#table-of-contents)_

---

## Conversational AI

Chatbot creation based on the Hugging Face [State-of-the-Art Conversational AI](https://github.com/huggingface/transfer-learning-conv-ai).

Supported model types:

- GPT
- GPT2

### Data format

Data format follows the [Facebook Persona-Chat](http://arxiv.org/abs/1801.07243) format. A JSON formatted version by Hugging Face is found [here](https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json). The JSON file is directly compatible with this library (and it will be automatically downloaded and used if no dataset is specified).

Each entry in personachat is a **dict** with two keys `personality` and `utterances`, the dataset is a list of entries.

- `personality`:  **list of strings** containing the personality of the agent
- `utterances`: **list of dictionaries**, each of which has two keys which are **lists of strings**.
  - `candidates`: [next_utterance_candidate_1, ..., next_utterance_candidate_19]  
        The last candidate is the ground truth response observed in the conversational data
  - `history`: [dialog_turn_0, ... dialog_turn N], where N is an odd number since the other user starts every conversation.

Preprocessing:  

- Spaces before periods at end of sentences
- everything lowercase

Example train data:

```json
[
    {
        "personality": [
            "i like computers .",
            "i like reading books .",
            "i like talking to chatbots .",
            "i love listening to classical music ."
        ],
        "utterances": [
            {
                "candidates": [
                    "i try to wear all black every day . it makes me feel comfortable .",
                    "well nursing stresses you out so i wish luck with sister",
                    "yeah just want to pick up nba nfl getting old",
                    "i really like celine dion . what about you ?",
                    "no . i live near farms .",
                    "mother taught me to cook ! we are looking for an exterminator .",
                    "i enjoy romantic movie . what is your favorite season ? mine is summer .",
                    "editing photos takes a lot of work .",
                    "you must be very fast . hunting is one of my favorite hobbies .",
                    "hi there . i'm feeling great! how about you ?"
                ],
                "history": [
                    "hi , how are you ?"
                ]
            },
            {
                "candidates": [
                    "i have trouble getting along with family .",
                    "i live in texas , what kind of stuff do you do in ",
                    "toronto ?",
                    "that's so unique ! veganism and line dancing usually don't mix !",
                    "no , it isn't that big . do you travel a lot",
                    "that's because they are real ; what do you do for work ?",
                    "i am lazy all day lol . my mom wants me to get a job and move out",
                    "i was born on arbor day , so plant a tree in my name",
                    "okay , i should not tell you , its against the rules ",
                    "i like to talk to chatbots too ! do you know why ? ."
                ],
                "history": [
                    "hi , how are you ?",
                    "hi there . i'm feeling great! how about you ?",
                    "not bad ! i am trying out this chatbot ."
                ]
            },
            {
                "candidates": [
                    "ll something like that . do you play games ?",
                    "does anything give you relief ? i hate taking medicine for mine .",
                    "i decorate cakes at a local bakery ! and you ?",
                    "do you eat lots of meat",
                    "i am so weird that i like to collect people and cats",
                    "how are your typing skills ?",
                    "yeah . i am headed to the gym in a bit to weight lift .",
                    "yeah you have plenty of time",
                    "metal is my favorite , but i can accept that people listen to country . haha",
                    "that's why you desire to be controlled . let me control you person one .",
                    "two dogs they are the best , how about you ?",
                    "you do art ? what kind of art do you do ?",
                    "i love watching baseball outdoors on sunny days .",
                    "oh i see . do you ever think about moving ? i do , it is what i want .",
                    "because i am a chatbot too, silly !"
                ],
                "history": [
                    "hi , how are you ?",
                    "hi there . i'm feeling great! how about you ?",
                    "not bad ! i am trying out this chatbot .",
                    "i like to talk to chatbots too ! do you know why ? .",
                    "no clue, why don't you tell me ?"
                ]
            }
        ]
    }
]
```

### Minimal Example

You can download the pretrained (OpenAI GPT based) Conversation AI model open-sourced by Hugging Face [here](https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz).

For the minimal example given below, you can download the model and extract it to `gpt_personachat_cache`. Note that you can use any of the other GPT or GPT-2 models but they will require more training.

You will also need to create the JSON file given in the Data Format section and save it as `data/minimal_train.json`.

```python
from simpletransformers.conv_ai import ConvAIModel


train_args = {
    "num_train_epochs": 50,
    "save_model_every_epoch": False,
}

# Create a ConvAIModel
model = ConvAIModel("gpt", "gpt_personachat_cache", use_cuda=True, args=train_args)

# Train the model
model.train_model("data/minimal_train.json")

# Evaluate the model
model.eval_model()

# Interact with the trained model.
model.interact()

```

The `interact()` method can be given a list of Strings which will be used to build a personality. If a list of Strings is not given, a random personality will be chosen from PERSONA-CHAT instead.

### Real Dataset Example

- [Persona-Chat Conversational AI](https://medium.com/@chaturangarajapakshe/how-to-train-your-chatbot-with-simple-transformers-da25160859f4?sk=edd04e406e9a3523fcfc46102529e775)

### ConvAIModel

`class simpletransformers.conv_ai.ConvAIModel ( model_type, model_name, args=None, use_cuda=True, cuda_device=-1, **kwargs)`  
This class is used to build Conversational AI.

`Class attributes`

- `tokenizer`: The tokenizer to be used.
- `model`: The model to be used.
            model_name: Default Transformer model name or path to Transformer model file (pytorch_model.bin).
- `device`: The device on which the model will be trained and evaluated.
- `results`: A python dict of past evaluation results for the TransformerModel object.
- `args`: A python dict of arguments used for training and evaluation.

`Parameters`

- `model_type`: (required) str - The type of model to use.
- `model_name`: (required) str - The exact model to use. Could be a pretrained model name or path to a directory containing a model. See [Current Pretrained Models](#current-pretrained-models) for all available models.
- `args`: (optional) python dict - A dictionary containing any settings that should be overwritten from the default values.
- `use_cuda`: (optional) bool - Default = True. Flag used to indicate whether CUDA should be used.
- `cuda_device`: (optional) int - Default = -1. Used to specify which GPU should be used.

`class methods`  
**`train_model(self, train_file=None, output_dir=None, show_running_loss=True, args=None, eval_file=None)`**

Trains the model using 'train_file'

Args:  

- train_df: ath to JSON file containing training data. The model will be trained on this file.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

- output_dir (optional): The directory where model files will be saved. If not given, self.args['output_dir'] will be used.

- show_running_loss (Optional): Set to False to prevent training loss being printed.

- args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.

- eval_file (optional): Evaluation data against which evaluation will be performed when evaluate_during_training is enabled. If not given when evaluate_during_training is enabled, the evaluation data from PERSONA-CHAT will be used.

Returns:

- None

**`eval_model(self, eval_file, output_dir=None, verbose=True, silent=False)`**

Evaluates the model on eval_file. Saves results to output_dir.

Args:  

- eval_file: Path to JSON file containing evaluation data. The model will be evaluated on this file.  
If not given, eval dataset from PERSONA-CHAT will be used.

- output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.  

- verbose: If verbose, results will be printed to the console on completion of evaluation.

- silent: If silent, tqdm progress bars will be hidden.

Returns:  

- result: Dictionary containing evaluation results. (correct, similar, incorrect)

- text: A dictionary containing the 3 dictionaries correct_text, similar_text (the predicted answer is a substring of the correct answer or vise versa), incorrect_text.

**`interact(self, personality=None)`**

Interact with a model in the terminal.

Args:

- personality (optional): A list of sentences that the model will use to build a personality.  
If not given, a random personality from PERSONA-CHAT will be picked.
  
```python
model.interact(
    personality=[
        "i like computers .",
        "i like reading books .",
        "i love classical music .",
        "i am very social ."
    ]
)
```

Returns:

- None

**`train(self, train_dataloader, output_dir, show_running_loss=True, eval_dataloader=None, verbose=verbose)`**

Trains the model on train_dataset.
*Utility function to be used by the train_model() method. Not intended to be used directly.*

**`evaluate(self, eval_file, output_dir, verbose=True, silent=False)`**

Evaluates the model on eval_file.
*Utility function to be used by the eval_model() method. Not intended to be used directly*

**`load_and_cache_examples(self, dataset_path=None, evaluate=False, no_cache=False, verbose=True, silent=False)`**

Loads, tokenizes, and prepares data for training and/or evaluation.
*Utility function for train() and eval() methods. Not intended to be used directly*

### Additional attributes for Conversational AI

ConvAIModel has a few additional attributes in its `args` dictionary, given below with their default values.

```python
    "num_candidates": 2,
    "personality_permutations": 1,
    "max_history": 2,
    "lm_coef": 2.0,
    "mc_coef": 1.0,
    "no_sample": False,
    "max_length": 20,
    "min_length": 1,
    "temperature": 0.7,
    "top_k": 0,
    "top_p": 0.9,
```

#### *num_candidates: int*

Number of candidates for training

#### *personality_permutations: int*

Number of permutations of personality sentences".

#### *max_history: int*

Number of previous exchanges to keep in history

#### *lm_coef: int*

LM loss coefficient

#### *mc_coef: int*

Multiple-choice loss coefficient

#### *no_sample: bool*

Set to use greedy decoding instead of sampling

#### *max_length: int*

Maximum length of the output utterances

#### *min_length: int*

Minimum length of the output utterances

#### *temperature: float*

Sampling softmax temperature

#### *top_k: float*

Filter top-k tokens before sampling (<=0: no filtering)

#### *top_p: float*

Nucleus filtering (top-p) before sampling (<=0.0: no filtering)

_[Back to Table of Contents](#table-of-contents)_

---

## Multi-Modal Classification

Multi-Modal Classification fuses text and image data. This is performed using multi-modal bitransformer models
introduced in the paper [Supervised Multimodal Bitransformers for Classifying Images and Text](https://arxiv.org/abs/1909.02950).

Supported model types:

- BERT

### Data format

There are several possible input formats you may use. The input formats are inspired by the [MM-IMDb](http://lisi1.unal.edu.co/mmimdb/) format.
Note that several options for data preprocessing have been added for convenience and flexibility when dealing with
complex datasets which can be found after the input format definitions.

#### 1 - Directory based

Each subset of data (E.g: train and test) should be in its own directory. The path to the directory can then be given
directly to either `train_model()` or `eval_model()`.

Each data sample should have a text portion and an image associated with it (and a label/labels for training and evaluation data).
The text for each sample should be in a separate JSON file. The JSON file may contain other fields in addition to the text
itself but they will be ignored. The image associated with each sample should be in the same directory and both the text
and the image must have the same identifier except for the file extension (E.g: 000001.json and 000001.jpg).

#### 2 - Directory and file list

All data (including both train and test data) should be in the same directory. The path to this directory should be given
to both `train_model()` and `eval_model()`. A second argument, `files_list` specifies which files should be taken from
the directory. `files_list` can be a Python list or the path to a JSON file containing the list of files.

Each data sample should have a text portion and an image associated with it (and a label/labels for training and evaluation data).
The text for each sample should be in a separate JSON file. The JSON file may contain other fields in addition to the text
itself but they will be ignored. The image associated with each sample should be in the same directory and both the text
and the image must have the same identifier except for the file extension (E.g: 000001.json and 000001.jpg).

#### 3 - Pandas DataFrame

Data can also be given in a Pandas DataFrame. When using this format, the `image_path` argument must be specified and
it should be a String of the path to the directory containing the images. The DataFrame should contain at least 3
columns as detailed below.

- `text` (str) - The text associated with the sample.
- `images` (str) - The relative path to the image file from `image_path` directory.
- `labels` (str) - The label (or list of labels for multilabel tasks) associated with the sample.

### Using custom names for column names or fields in JSON files

By default, Simple Transformers will look for column/field names `text`, `images`, and `labels`. However, you can define
your own names to use in place of these names. This behaviour is controlled using the three attributes `text_label`, `labels_label`,
 and `images_label` in the `args` dictionary.

You can set your custom names when creating the model by assigning the custom name to the corresponding attribute in the
`args` dictionary.

You can also change these values at training and/or evaluation time (but not with the `predict()` method) by passing the
names to the arguments `text_label`, `labels_label`, and `images_label`. Note that the change will persist even after
the method call terminates. That is, the `args` dictionary of the model itself will be modified.

### Specifying the file type extension for image and text files

By default, Simple Transformers will assume that any paths will also include the file type extension (E.g: .json or .jpg).
Alternatively, you can specify the extensions using the `image_type_extension` and `data_type_extension` attributes (for
image file extensions and text file extensions respectively) in the `args` dictionary.

This too can be done when creating the model or when running the `train_model()` or `eval_model()` methods. The changes
will persist in the `args` dictionary when using these methods.

The `image_type_extension` can be specified when using the `predict()` method but the change WILL NOT persist.

### Label formats

With Multi-Modal Classification, labels are always given as strings. You may specify a list of labels by passing in the
list to `label_list` argument when creating the model. If `label_list` is given, `num_labels` is not required.

If `label_list` is not given, `num_labels` is required and the labels should be Strings starting from `"0"` up to
`"<num_labels>"`.

### Creating a Model

Create a `MultiModalClassificationModel`.

```python
from simpletransformers.classification.multi_modal_classification_model import MultiModalClassificationModel


model = MultiModalClassificationModel("bert", "bert-base-uncased")
```

Available arguments:

```python
"""
Args:
    model_type: The type of model (bert, xlnet, xlm, roberta, distilbert, albert)
    model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_model.bin).
    multi_label (optional): Set to True for multi label tasks.
    label_list (optional) : A list of all the labels (str) in the dataset.
    num_labels (optional): The number of labels or classes in the dataset.
    pos_weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
    args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
    use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
    cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
    **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
"""
```

### Training a Model

Use the `train_model()` method to train. You can use the `auto_weights` feature to balance out unbalanced datasets.

Available arguments:

```python
"""
Args:
    data: Path to data directory containing text files (JSON) and image files OR a Pandas DataFrame.
        If a DataFrame is given, it should contain the columns [text, labels, images]. When using a DataFrame,
        image_path MUST be specified. The image column of the DataFrame should contain the relative path from
        image_path to the image.
        E.g:
            For an image file 1.jpeg located in "data/train/";
                image_path = "data/train/"
                images = "1.jpeg"
    files_list (optional): If given, only the files specified in this list will be taken from data directory.
        files_list can be a Python list or the path (str) to a JSON file containing a list of files.
    image_path (optional): Must be specified when using DataFrame as input. Path to the directory containing the
        images.
    text_label (optional): Column name to look for instead of the default "text"
    labels_label (optional): Column name to look for instead of the default "labels"
    images_label (optional): Column name to look for instead of the default "images"
    image_type_extension (optional): If given, this will be added to the end of each value in "images".
    data_type_extension (optional): If given, this will be added to the end of each value in "files_list".
    auto_weights (optional): If True, weights will be used to balance the classes.
    output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
    show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
    args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
    eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
    **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
"""
```

### Evaluating a Model

Use the `eval_model()` method to evaluate. You can load a saved model by giving the path to the model directory as
`model_name`. Note that you need to provide the same arguments when loading a saved model as you did when creating the
original model.

```python
model = MultiModalClassificationModel("bert", "outputs")
results, _ = model.eval_model("data/dataset/", "data/dev.json")
```

Available arguments:

```python
"""
Args:
    data: Path to data directory containing text files (JSON) and image files OR a Pandas DataFrame.
        If a DataFrame is given, it should contain the columns [text, labels, images]. When using a DataFrame,
        image_path MUST be specified. The image column of the DataFrame should contain the relative path from
        image_path to the image.
        E.g:
            For an image file 1.jpeg located in "data/train/";
                image_path = "data/train/"
                images = "1.jpeg"
    files_list (optional): If given, only the files specified in this list will be taken from data directory.
        files_list can be a Python list or the path (str) to a JSON file containing a list of files.
    image_path (optional): Must be specified when using DataFrame as input. Path to the directory containing the
        images.
    text_label (optional): Column name to look for instead of the default "text"
    labels_label (optional): Column name to look for instead of the default "labels"
    images_label (optional): Column name to look for instead of the default "images"
    image_type_extension (optional): If given, this will be added to the end of each value in "images".
    data_type_extension (optional): If given, this will be added to the end of each value in "files_list".
    output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
    verbose: If verbose, results will be printed to the console on completion of evaluation.
    silent: If silent, tqdm progress bars will be hidden.
    **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

"""
```

### Predicting from a trained Model

Use the `predict()` method to make predictions. You can load a saved model by giving the path to the model directory as
`model_name`. Note that you need to provide the same arguments when loading a saved model as you did when creating the
original model.

```python
model = MultiModalClassificationModel("bert", "outputs")
model.predict(
    {
        "text": [
            "A lawyer is forced to defend a guilty judge, while defending other innocent clients, and trying to find punishment for the guilty and provide justice for the innocent."
        ],
        "labels": ["Crime"],
        "images": ["0078718"]
    },
    image_path="data/dataset",
    image_type_extension=".jpeg"
)
```

_[Back to Table of Contents](#table-of-contents)_

---

## Regression

Regression tasks also use the ClassificationModel with 2 caveats.

1. `num_labels` should be 1.
2. `regression` should be `True` in `args` dict.

Regression can be used with either single sentence or sentence pair tasks.

#### Minimal Start for Regression

```python
from simpletransformers.classification import ClassificationModel
import pandas as pd


train_data = [
    ['Example sentence belonging to class 1', 'Yep, this is 1', 1.8],
    ['Example sentence belonging to class 0', 'Yep, this is 0', 0.2],
    ['Example  2 sentence belonging to class 0', 'Yep, this is 0', 4.5]
]

train_df = pd.DataFrame(train_data, columns=['text_a', 'text_b', 'labels'])

eval_data = [
    ['Example sentence belonging to class 1', 'Yep, this is 1', 1.9],
    ['Example sentence belonging to class 0', 'Yep, this is 0', 0.1],
    ['Example  2 sentence belonging to class 0', 'Yep, this is 0', 5]
]

eval_df = pd.DataFrame(eval_data, columns=['text_a', 'text_b', 'labels'])

train_args={
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs': 3,

    'regression': True,
}

# Create a ClassificationModel
model = ClassificationModel('roberta', 'roberta-base', num_labels=1, use_cuda=True, cuda_device=0, args=train_args)
print(train_df.head())

# Train the model
model.train_model(train_df, eval_df=eval_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

predictions, raw_outputs = model.predict([["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
print(predictions)
print(raw_outputs)
```

---

## Visualization Support

The [Weights & Biases](https://www.wandb.com/) framework is supported for visualizing model training.

To use this, simply set a project name for W&B in the `wandb_project` attribute of the `args` dictionary. This will log all hyperparameter values, training losses, and evaluation metrics to the given project.

```python
model = ClassificationModel('roberta', 'roberta-base', args={'wandb_project': 'project-name'})
```

For a complete example, see [here](https://medium.com/skilai/to-see-is-to-believe-visualizing-the-training-of-machine-learning-models-664ef3fe4f49).

_[Back to Table of Contents](#table-of-contents)_

---

## Experimental Features

To use experimental features, import from `simpletransformers.experimental.X`

```python
from simpletransformers.experimental.classification import ClassificationModel
```

### Sliding Window For Long Sequences

Normally, sequences longer than `max_seq_length` are unceremoniously truncated.

This experimental feature moves a sliding window over each sequence and generates sub-sequences with length `max_seq_length`. The model output for each sub-sequence is averaged into a single output before being sent to the linear classifier.

Currently available on binary and multiclass classification models of the following types:

* BERT
* DistilBERT
* RoBERTa
* AlBERT
* XLNet
* CamemBERT

Set `sliding_window` to `True` for the ClassificationModel to enable this feature.

```python
from simpletransformers.classification import ClassificationModel
import pandas as pd
import sklearn

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column in the label with type int.
train_data = [['Example sentence belonging to class 1' * 50, 1], ['Example sentence belonging to class 0', 0], ['Example  2 sentence belonging to class 0', 0]] + [['Example sentence belonging to class 0', 0] for i in range(12)]
train_df = pd.DataFrame(train_data, columns=['text', 'labels'])


eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0]]
eval_df = pd.DataFrame(eval_data)

train_args={
    'sliding_window': True,
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'evaluate_during_training': True,
    'logging_steps': 5,
    'stride': 0.8,
    'max_seq_length': 128
}

# Create a TransformerModel
model = ClassificationModel('camembert', 'camembert-base', args=train_args, use_cuda=False)
print(train_df.head())

# Train the model
model.train_model(train_df, eval_df=eval_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)

predictions, raw_outputs = model.predict(["I'd like to puts some CD-ROMS on my iPad, is that possible?' — Yes, but wouldn't that block the screen?" * 25])
print(predictions)
print(raw_outputs)
```

_[Back to Table of Contents](#table-of-contents)_

---
## Loading Saved Models

To load a saved model, provide the path to the directory containing the saved model as the `model_name`.
_Note that you will need to specify the correct (usually the same used in training) `args` when loading the model_

```python
model = ClassificationModel('roberta', 'outputs/', args={})
```

```python
model = NERModel('bert', 'outputs/', args={})
```

_[Back to Table of Contents](#table-of-contents)_

---


## Default Settings

The default args used are given below. Any of these can be overridden by passing a dict containing the corresponding 
key: value pairs to the the init method of a Model class.

```python
self.args = {
    "output_dir": "outputs/",
    "cache_dir": "cache/",
    "best_model_dir": "outputs/best_model/",

    "fp16": True,
    "max_seq_length": 128,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 4e-5,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,

    "logging_steps": 50,
    "evaluate_during_training": False,
    "evaluate_during_training_steps": 2000,
    "evaluate_during_training_verbose": False,
    "use_cached_eval_features": False,
    "save_eval_checkpoints": True
    "save_steps": 2000,
    "no_cache": False,
    "save_model_every_epoch": True,
    "tensorboard_dir": None,

    "overwrite_output_dir": False,
    "reprocess_input_data": True,

    "process_count": cpu_count() - 2 if cpu_count() > 2 else 1
    "n_gpu": 1,
    "silent": False,
    "use_multiprocessing": True,

    "wandb_project": None,
    "wandb_kwargs": {},

    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,

    "manual_seed": None,
    "encoding": None,
    "config": {},
}
```

### Args Explained

#### *output_dir: str*
The directory where all outputs will be stored. This includes model checkpoints and evaluation results.

#### *cache_dir: str*
The directory where cached files will be saved.

#### *best_model_dir: str*
The directory where the best model (model checkpoints) will be saved if evaluate_during_training is enabled and the training loop achieves a lowest evaluation loss calculated after every evaluate_during_training_steps, or an epoch.

#### *fp16: bool*
Whether or not fp16 mode should be used. Requires NVidia Apex library.

#### *max_seq_length: int*
Maximum sequence level the model will support.

#### *train_batch_size: int*
The training batch size.

#### *gradient_accumulation_steps: int*
The number of training steps to execute before performing a `optimizer.step()`. Effectively increases the training batch size while sacrificing training time to lower memory consumption.

#### *eval_batch_size: int*
The evaluation batch size.

#### *num_train_epochs: int*
The number of epochs the model will be trained for.

#### *weight_decay: float*
Adds L2 penalty.

#### *learning_rate: float*
The learning rate for training.

#### *adam_epsilon: float*
Epsilon hyperparameter used in AdamOptimizer.

#### *max_grad_norm: float*
Maximum gradient clipping.

#### *do_lower_case: bool*
Set to True when using uncased models.

#### *evaluate_during_training*
Set to True to perform evaluation while training models. Make sure `eval_df` is passed to the training method if enabled.

#### *evaluate_during_training_steps*
Perform evaluation at every specified number of steps. A checkpoint model and the evaluation results will be saved.

#### *evaluate_during_training_verbose*
Print results from evaluation during training.

#### *use_cached_eval_features*
Evaluation during training uses cached features. Setting this to `False` will cause features to be recomputed at every evaluation step.

#### *save_eval_checkpoints*
Save a model checkpoint for every evaluation performed.

#### *logging_steps: int*
Log training loss and learning at every specified number of steps.

#### *save_steps: int*
Save a model checkpoint at every specified number of steps.

#### *no_cache: bool*
Cache features to disk.

#### *save_model_every_epoch: bool*
Save a model at the end of every epoch.

#### *tensorboard_dir: str*
The directory where Tensorboard events will be stored during training. By default, Tensorboard events will be saved in a subfolder inside `runs/`  like `runs/Dec02_09-32-58_36d9e58955b0/`.

#### *overwrite_output_dir: bool*
If True, the trained model will be saved to the ouput_dir and will overwrite existing saved models in the same directory.

#### *reprocess_input_data: bool*
If True, the input data will be reprocessed even if a cached file of the input data exists in the cache_dir.

#### *process_count: int*
Number of cpu cores (processes) to use when converting examples to features. Default is (number of cores - 2) or 1 if (number of cores <= 2)

#### *n_gpu: int*
Number of GPUs to use.

#### *silent: bool*
Disables progress bars.

#### *use_multiprocessing: bool*
If True, multiprocessing will be used when converting data into features. Disabling can reduce memory usage, but may substantially slow down processing.


#### *wandb_project: str*
Name of W&B project. This will log all hyperparameter values, training losses, and evaluation metrics to the given project.

#### *wandb_kwargs: dict*
Dictionary of keyword arguments to be passed to the W&B project.

#### *use_early_stopping*
Use early stopping to stop training when `early_stopping_metric` doesn't improve (based on `early_stopping_patience`, and `early_stopping_delta`)

#### *early_stopping_patience*
Terminate training after this many evaluations without an improvement in `eval_loss` greater then `early_stopping_delta`.

#### *early_stopping_delta*
The improvement over `best_eval_loss` necessary to count as a better checkpoint.

#### *early_stopping_metric*
The metric that should be used with early stopping. (Should be computed during `eval_during_training`).

#### *early_stopping_metric_minimize*
Whether `early_stopping_metric` should be minimized (or maximized).

#### *manual_seed*
Set a manual seed if necessary for reproducible results.

#### *encoding*
Specify an encoding to be used when reading text files.

#### *config*
A dictionary containing configuration options that should be overriden in a model's config.

---

## Current Pretrained Models

For a list of pretrained models, see [Hugging Face docs](https://huggingface.co/pytorch-transformers/pretrained_models.html).

The `model_types` available for each task can be found under their respective section. Any pretrained model of that type
found in the Hugging Face docs should work. To use any of them set the correct `model_type` and `model_name` in the `args` 
dictionary.

_[Back to Table of Contents](#table-of-contents)_

---

## Acknowledgements

None of this would have been possible without the hard work by the HuggingFace team in developing the [Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers) library.

_<div>Icon for the Social Media Preview made by <a href="https://www.flaticon.com/authors/freepik" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>_

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/hawktang"><img src="https://avatars0.githubusercontent.com/u/2004071?v=4" width="100px;" alt=""/><br /><sub><b>hawktang</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=hawktang" title="Code">💻</a></td>
    <td align="center"><a href="http://datawizzards.io"><img src="https://avatars0.githubusercontent.com/u/22409996?v=4" width="100px;" alt=""/><br /><sub><b>Mabu Manaileng</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=mabu-dev" title="Code">💻</a></td>
    <td align="center"><a href="https://www.facebook.com/aliosm97"><img src="https://avatars3.githubusercontent.com/u/7662492?v=4" width="100px;" alt=""/><br /><sub><b>Ali Hamdi Ali Fadel</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=AliOsm" title="Code">💻</a></td>
    <td align="center"><a href="http://tovly.co"><img src="https://avatars0.githubusercontent.com/u/12242351?v=4" width="100px;" alt=""/><br /><sub><b>Tovly Deutsch</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=TovlyDeutsch" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/hlo-world"><img src="https://avatars0.githubusercontent.com/u/9633055?v=4" width="100px;" alt=""/><br /><sub><b>hlo-world</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=hlo-world" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/huntertl"><img src="https://avatars1.githubusercontent.com/u/15113885?v=4" width="100px;" alt=""/><br /><sub><b>huntertl</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=huntertl" title="Code">💻</a></td>
    <td align="center"><a href="https://whattheshot.com"><img src="https://avatars2.githubusercontent.com/u/623763?v=4" width="100px;" alt=""/><br /><sub><b>Yann Defretin</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=kinoute" title="Code">💻</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=kinoute" title="Documentation">📖</a> <a href="#question-kinoute" title="Answering Questions">💬</a> <a href="#ideas-kinoute" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/mananeau"><img src="https://avatars0.githubusercontent.com/u/29440170?v=4" width="100px;" alt=""/><br /><sub><b>Manuel </b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=mananeau" title="Documentation">📖</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=mananeau" title="Code">💻</a></td>
    <td align="center"><a href="http://jacobsgill.es"><img src="https://avatars2.githubusercontent.com/u/9109832?v=4" width="100px;" alt=""/><br /><sub><b>Gilles Jacobs</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=GillesJ" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/shasha79"><img src="https://avatars2.githubusercontent.com/u/5512649?v=4" width="100px;" alt=""/><br /><sub><b>shasha79</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=shasha79" title="Code">💻</a></td>
    <td align="center"><a href="http://www-lium.univ-lemans.fr/~garcia"><img src="https://avatars2.githubusercontent.com/u/14233427?v=4" width="100px;" alt=""/><br /><sub><b>Mercedes Garcia</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=merc85garcia" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/hammad26"><img src="https://avatars1.githubusercontent.com/u/12643784?v=4" width="100px;" alt=""/><br /><sub><b>Hammad Hassan Tarar</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=hammad26" title="Code">💻</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=hammad26" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/todd-cook"><img src="https://avatars3.githubusercontent.com/u/665389?v=4" width="100px;" alt=""/><br /><sub><b>Todd Cook</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=todd-cook" title="Code">💻</a></td>
    <td align="center"><a href="http://knuthellan.com/"><img src="https://avatars2.githubusercontent.com/u/51441?v=4" width="100px;" alt=""/><br /><sub><b>Knut O. Hellan</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=khellan" title="Code">💻</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=khellan" title="Documentation">📖</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/nagenshukla"><img src="https://avatars0.githubusercontent.com/u/39196228?v=4" width="100px;" alt=""/><br /><sub><b>nagenshukla</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=nagenshukla" title="Code">💻</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/flaviussn/"><img src="https://avatars0.githubusercontent.com/u/20523032?v=4" width="100px;" alt=""/><br /><sub><b>flaviussn</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=flaviussn" title="Code">💻</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=flaviussn" title="Documentation">📖</a></td>
    <td align="center"><a href="http://marctorrellas.github.com"><img src="https://avatars1.githubusercontent.com/u/22045779?v=4" width="100px;" alt=""/><br /><sub><b>Marc Torrellas</b></sub></a><br /><a href="#maintenance-marctorrellas" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://github.com/adrienrenaud"><img src="https://avatars3.githubusercontent.com/u/6208157?v=4" width="100px;" alt=""/><br /><sub><b>Adrien Renaud</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=adrienrenaud" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/jacky18008"><img src="https://avatars0.githubusercontent.com/u/9031441?v=4" width="100px;" alt=""/><br /><sub><b>jacky18008</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=jacky18008" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/seo-95"><img src="https://avatars0.githubusercontent.com/u/38254541?v=4" width="100px;" alt=""/><br /><sub><b>Matteo Senese</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=seo-95" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/sarthakTUM"><img src="https://avatars2.githubusercontent.com/u/23062869?v=4" width="100px;" alt=""/><br /><sub><b>sarthakTUM</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=sarthakTUM" title="Documentation">📖</a> <a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=sarthakTUM" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/djstrong"><img src="https://avatars1.githubusercontent.com/u/1849959?v=4" width="100px;" alt=""/><br /><sub><b>djstrong</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=djstrong" title="Code">💻</a></td>
    <td align="center"><a href="http://kozistr.tech"><img src="https://avatars2.githubusercontent.com/u/15344796?v=4" width="100px;" alt=""/><br /><sub><b>Hyeongchan Kim</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=kozistr" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/Pradhy729"><img src="https://avatars3.githubusercontent.com/u/49659913?v=4" width="100px;" alt=""/><br /><sub><b>Pradhy729</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=Pradhy729" title="Code">💻</a></td>
    <td align="center"><a href="https://iknoorjobs.github.io/"><img src="https://avatars2.githubusercontent.com/u/22852967?v=4" width="100px;" alt=""/><br /><sub><b>Iknoor Singh</b></sub></a><br /><a href="https://github.com/ThilinaRajapakse/simpletransformers/commits?author=iknoorjobs" title="Documentation">📖</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

*If you should be on this list but you aren't, or you are on the list but don't want to be, please don't hesitate to contact me!*
