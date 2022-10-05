
## make a funcion that configurating a path allows us
## load different parts of a dataset 
import logging
import time

import tensorflow_datasets as tfds
import tensorflow as tf


def load_language_dataset(path):
    """
    Loads the language dataset, depending on path 
    it will be based on one language or the other
    
    args: string path . Right now portuguese 

    """

    examples, metadata = tfds.load(path,
                               with_info=True,
                               as_supervised=True)
    print('----LOADING DATASET----')
    train_examples, val_examples = examples['train'], examples['validation']
    return train_examples, val_examples


## different paths to make different languages. for PIPELINE dvc.yaml 
# https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate#ted_hrlr_translatept_to_en
#[#'az_to_en', #'aztr_to_en', #'be_to_en',
# #'beru_to_en', 'gl_to_en', 'glpt_to_en',
# #'pt_to_en', #'ru_to_en', 'tr_to_en']


def load_dataset_test(path):
    """
    See the language that has been loaded
    """

    train_examples, _ = load_language_dataset(path)

    for lan_examples, en_examples in train_examples.batch(3).take(1):
        print('----EXAMPLES IN LANGUAGE TO TRANSLATE----')
    for lan in lan_examples.numpy():
        print(lan.decode('utf-8'))
    print()

    print('----EXAMPLES IN ENGLISH----')
    for en in en_examples.numpy():
        print(en.decode('utf-8'))

def save_tensor(tensor, filename):
  """Saves tensor to be a stage output"""
  one_string = tf.strings.format("{}\n", (tensor))
  tf.io.write_file(filename, one_string)
  print('----TENSOR SAVED----')

def load_tensor(tensor, filename):
  """Reads a tensor for being loaded later"""
  tf.io.read_file(str(filename), tensor)
  print('----TENSOR LOADED----')



load_language_dataset('ted_hrlr_translate/pt_to_en')
