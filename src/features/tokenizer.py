
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


#### TOKENIZER FOR EVERY LANGUAGE #####

## Instantiate this from data. Should go to main . Load the dataset . the function
# should be from local
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

bert_tokenizer_params=dict(lower_case=True)

def vocab_generator():
    """
    generates a wordpiece vocabulary from a dataset
    see for more params generator
    """
    
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
    


def write_vocab_file(filepath, vocab):
  with open(filepath, 'w') as f:
    for token in vocab:
      print(token, file=f)

vocabulary = vocab_generator()
file_vocabulary_lan = write_vocab_file('vocabularylan.txt', **bert_tokenizer_params)
file_vocabulary_en = write_vocab_file('vocabularyen.txt', **bert_tokenizer_params)

## Build the tokenizer

def tokenize(vocabularypath, bert_tokenizer_params):
    """
    Build the tokenizer.
    See the difference in between Bert tokenizer and tokenizer on how
    does that affect different languages
    """
    language_tokenizer = text.BertTokenizer(vocabularypath, bert_tokenizer_params)
    return language_tokenizer


pt_tokenizer = text.BertTokenizer('vocabularylan.txt', **bert_tokenizer_params)
en_tokenizer = text.BertTokenizer('vocabularyen.txt', **bert_tokenizer_params)

### Instantiate the tokenizer 

# Tokenize the examples -> (batch, word, word-piece)
train_lan, train_en = map_examples(train_examples=train_examples)
token_batch = en_tokenizer.tokenize(train_en)
# Merge the word and word-piece axes -> (batch, tokens)
token_batch = token_batch.merge_dims(-2,-1)
# Lookup each token id in the vocabulary.
txt_tokens = tf.gather(en_vocab, token_batch)
# Join with spaces.
tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1)
words = en_tokenizer.detokenize(token_batch)
tf.strings.reduce_join(words, separator=' ', axis=-1)


#Save Model
def save_model(tokenizers, model_name):
    """Save tokenizer model"""
    model_name = 'ted_hrlr_translate_pt_en_converter'
    return tf.saved_model.save(tokenizers, model_name)












#------------------------------------------------------#
## Build Tokenizer

def tokenizer_model_load(model_name):
    """
    Process of breaking up a sequence.
    The beginning of sentences are typically marked
    by tokens

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

def prepare_token_batches(lan, en):
    """
    Tokenize per batches

    """
    model = tokenizer_model_load('ted_hrlr_translate_pt_en_converter')
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