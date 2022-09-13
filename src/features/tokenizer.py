
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

from data import load_dataset

## Instantiate this from data. Should go to main .
train_examples, val_examples = load_dataset(path) 

def map_examples(train_examples):
    """
    map examples
    Lowercase, spaces around punctuation
    Not clear unicode normalization
    """
    train_en = train_examples.map(lambda lan, en: en)
    train_lan = train_examples.map(lambda lan, en: lan)
    return train_lan, train_en


def vocab_generator():
    """
    generates a wordpiece vocabulary from a dataset
    see for more params generator
    """
    bert_tokenizer_params=dict(lower_case=True)
    reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size = 8000,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=bert_tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )
    ##%%time The notebook takes a lot of time so it would be nice to benchmark this
    
    train_lan, train_en = map_examples(train_examples)
    
    lan_vocab = bert_vocab.bert_vocab_from_dataset(
        train_lan.batch(1000).prefetch(2),
        **bert_vocab_args
    )
#### Cell 57 

def write_vocab_file(filepath, vocab):
  with open(filepath, 'w') as f:
    for token in vocab:
      print(token, file=f)


## Build the tokenizer

def tokenize(vocabularypath, bert_tokenizer_params):
    """
    Build the tokenizer.
    See the difference in between Bert tokenizer and tokenizer on how
    does that affect different languages
    """
    language_tokenizer = text.BertTokenizer(vocabularypath, bert_tokenizer_params)
    return language_tokenizer


## Build Tokenizer

def tokenizer(model_name):
    """
    Process of breaking up a sequence.
    The beginning of sentences are typically marked
    by tokend

    Apparently pt example is done and other language might imply going through the 
    full tutorial https://www.tensorflow.org/text/guide/subwords_tokenizer


    Buils subword tokenizers optimized for the dataset and exports them
    into a Tensorflow saved_model format
    
    returns: tensorflow saved model

    """
    model_name = 'ted_hrlr_translate_pt_en_converter'
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True
    )

    tokenizers = tf.saved_model.load(model_name)
    #encoded = tokenizers.en.tokenize(en_examples)-> Thinking that this is not necessary
    return tokenizers









### Add param to process. Without limitng the size of sequences, the performance
# will be affected. 
MAX_TOKENS = 128



def filter_max_tokens(lan, en):
    num_tokens = tf.maximum(tf.shape(lan)[1],tf.shape(en)[1])
    return num_tokens < MAX_TOKENS


def tokenize_pairs(lan, en, model_name):

    tokenizers = tokenizer(model_name)

    lan = tokenizers.lan.tokenize(lan)
    # Convert from ragged to dense, padding with zeros.
    lan = lan.to_tensor()

    en = tokenizers.en.tokenize(en)
    # Convert from ragged to dense, padding with zeros.
    en = en.to_tensor()
    return lan, en