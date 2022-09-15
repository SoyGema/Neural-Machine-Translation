import collections
import os
import pathlib
import re
import string
import sys
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

from src.data.load_dataset import load_language_dataset

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

train_examples, val_examples = load_language_dataset('ted_hrlr_translate/tr_to_en') 


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
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )
    ##%%time The notebook takes a lot of time so it would be nice to benchmark this
    time
    train_lan, train_en = map_examples(train_examples)
    print('examples mapped')
    lan_vocab = bert_vocab.bert_vocab_from_dataset(
    train_lan.batch(1000).prefetch(2),
    **bert_vocab_args
    )
    print('vocabulary language 1 created')
    en_vocab = bert_vocab.bert_vocab_from_dataset(
    train_en.batch(1000).prefetch(2),
    **bert_vocab_args
    )
    return lan_vocab, en_vocab

def write_vocab_file(filepath, vocab):
  """
  Creates vocabulary files
  """
  with open(filepath, 'w') as f:
    f.write(str(vocab))


vocabulary_lan, _ = vocab_generator()
_ , vocabulary_en = vocab_generator()
file_vocabulary_lan = write_vocab_file('vocabularylan.txt', vocabulary_lan)
print('text portuguese done')
print(vocabulary_lan)


file_vocabulary_en = write_vocab_file('vocabularyen.txt', vocabulary_en)
print('text english done')
print(vocabulary_en)

def cleanup_text(reserved_tokens, token_txt):
  # Drop the reserved tokens, except for "[UNK]".
  reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

  bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
  bad_token_re = "|".join(bad_tokens)
    
  bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
  result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

  # Join them into strings.
  result = tf.strings.reduce_join(result, separator=' ', axis=-1)

  return result

reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]


def add_start_end(ragged):

  reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]
  START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
  END = tf.argmax(tf.constant(reserved_tokens) == "[END]")
  count = ragged.bounding_shape()[0]
  starts = tf.fill([count,1], START)
  ends = tf.fill([count,1], END)
  return tf.concat([starts, ragged, ends], axis=1)

class CustomTokenizer(tf.Module):
  def __init__(self, reserved_tokens, vocab_path):
    self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
    self._reserved_tokens = reserved_tokens
    self._vocab_path = tf.saved_model.Asset(vocab_path)

    vocab = pathlib.Path(vocab_path).read_text().splitlines()
    self.vocab = tf.Variable(vocab)

    ## Create the signatures for export:   

    # Include a tokenize signature for a batch of strings. 
    self.tokenize.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string))
    
    # Include `detokenize` and `lookup` signatures for:
    #   * `Tensors` with shapes [tokens] and [batch, tokens]
    #   * `RaggedTensors` with shape [batch, tokens]
    self.detokenize.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.detokenize.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    self.lookup.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.lookup.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    # These `get_*` methods take no arguments
    self.get_vocab_size.get_concrete_function()
    self.get_vocab_path.get_concrete_function()
    self.get_reserved_tokens.get_concrete_function()
    
  @tf.function
  def tokenize(self, strings):
    enc = self.tokenizer.tokenize(strings)
    # Merge the `word` and `word-piece` axes.
    enc = enc.merge_dims(-2,-1)
    enc = add_start_end(enc)
    return enc

  @tf.function
  def detokenize(self, tokenized):
    words = self.tokenizer.detokenize(tokenized)
    return cleanup_text(self._reserved_tokens, words)

  @tf.function
  def lookup(self, token_ids):
    return tf.gather(self.vocab, token_ids)

  @tf.function
  def get_vocab_size(self):
    return tf.shape(self.vocab)[0]

  @tf.function
  def get_vocab_path(self):
    return self._vocab_path

  @tf.function
  def get_reserved_tokens(self):
    return tf.constant(self._reserved_tokens)


if __name__ == '__main__':
    tokenizers = tf.Module()
    tokenizers.pt = CustomTokenizer(reserved_tokens, 'vocabularylan.txt')
    tokenizers.en = CustomTokenizer(reserved_tokens, 'vocabularyen.txt')
    model_name = 'ted_hrlr_translate_tr_to_en_converter'
    tf.saved_model.save(tokenizers, model_name)
    reloaded_tokenizers = tf.saved_model.load(model_name)
    reloaded_tokenizers.en.get_vocab_size().numpy()