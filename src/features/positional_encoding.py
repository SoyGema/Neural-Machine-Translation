import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))


import numpy as np
import tensorflow as tf
from features.tokenizer_transformer import load_dataset_tokenized
from data.load_dataset import load_language_dataset
from features.tokenizer_transformer import make_batches
import yaml

model_name = 'ted_hrlr_translate/pt_to_en'

with open('params.yaml') as config_file:
    config = yaml.safe_load(config_file)


def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
  
  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)



class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


if __name__ == '__main__':
  tokenizer = load_dataset_tokenized()

  model_name = 'ted_hrlr_translate/pt_to_en'
  train_examples, val_examples = load_language_dataset(model_name)
  train_batches = make_batches(train_examples)
  val_batches = make_batches(val_examples)

  for (pt, en), en_labels in train_batches.take(1):
    print('----TRAIN BATCH LANGUAGE 1----', pt.shape)
    print('----TRAIN BATCH ENGLISH ----', en.shape)

  print(tokenizer.pt.get_vocab_size())

  embed_pt = PositionalEmbedding(vocab_size=config['positional_encoding']['input_vocab_size'], d_model=config['positional_encoding']['d_model'])
  embed_en = PositionalEmbedding(vocab_size=config['positional_encoding']['target_vocab_size'], d_model=config['positional_encoding']['d_model'])

  ###See lan and en first instantiations
  pt_emb = embed_pt(pt)
  en_emb = embed_en(en)
  print(pt_emb)
  print(en_emb)
