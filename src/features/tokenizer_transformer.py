
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
from src.data.load_dataset import load_language_dataset

#------------------------------------------------------#
## Build Tokenizer

modelname = 'ted_hrlr_translate_pt_en_converter'
train_examples, val_examples = load_language_dataset(modelname)

def load_dataset_language():
    """Load the dataset from local,
    once dvc pull has been done """


### Add param to process. Without limitng the size of sequences, the performance
# will be affected. 
MAX_TOKENS = 128

def prepare_token_batches(lan, en):
    """
    Tokenize per batches

    """
    lan = model.lan.tokenize(lan)
    lan = lan[:, :MAX_TOKENS]
    lan = lan.to_tensor()

    en = model.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    return (lan, en_inputs), en_labels

BUFFER_SIZE = 20000
BATCH_SIZE = 64

def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_token_batches, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))


    # Create training and validation set batches.
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

#------------------------------------------------------#

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