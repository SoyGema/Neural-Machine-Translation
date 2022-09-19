
import os
from pyexpat import model
import numpy as np
#import matplotlib.pyplot as plt
import pathlib
from zipfile import ZipFile
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

from src.data.load_dataset import load_language_dataset
#------------------------------------------------------#
## Build Tokenizer

model_name_zip = 'ted_hrlr_translate_az_en_converter.zip'

model_name = 'ted_hrlr_translate/az_to_en'
train_examples, val_examples = load_language_dataset(model_name)

def load_dataset_tokenized():
    """Load the model from local,
    once dvc pull has been done """

    fullPath = os.path.abspath("/Users/gema/Documents/Neural-Machine-Translation/datasets/" + model_name_zip) 
    print('the path from it reads is'+ fullPath)
   
    model_for_processing = tf.keras.utils.get_file(model_name_zip, 'file://'+ fullPath, untar=True)
    
    with ZipFile(model_for_processing, 'r') as zipObj:
        zipObj.extractall('/Users/gema/.keras/datasets/')

    print('model loaded')
    folder_name = 'ted_hrlr_translate_az_en_converter'
    tokenizer = tf.saved_model.load('/Users/gema/.keras/datasets/' + folder_name)
    print(tokenizer)
    return tokenizer

### Add param to process. Without limitng the size of sequences, the performance
# will be affected. 
tokenizer = load_dataset_tokenized()
MAX_TOKENS = 128


def prepare_token_batches(pt, en):
    """
    Tokenize per batches. Try to remove initial arguments

    """
    pt = tokenizer.pt.tokenize(pt)
    pt = pt[:, :MAX_TOKENS]
    pt = pt.to_tensor()

    en = tokenizer.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    return (pt, en_inputs), en_labels

BUFFER_SIZE = 20000
BATCH_SIZE = 64

def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_token_batches, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))


    # Create training and validation set batches. Commented for now to ensure loading.
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

if __name__ == '__main__':
    load_dataset_tokenized()
    prepare_token_batches(lan, en)
    filter_max_tokens(lan, en)
    tokenize_pairs(lan, en, model_name)