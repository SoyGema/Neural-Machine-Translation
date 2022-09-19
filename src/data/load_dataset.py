
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
    print('are you doing this?')
    train_examples, val_examples = examples['train'], examples['validation']
    return train_examples, val_examples


def load_language_dataset(path):
    """
    Loads the language dataset, depending on path 
    it will be based on one language or the other
    
    args: string path . Right now portuguese 

    """

    examples, metadata = tfds.load(path,
                               with_info=True,
                               as_supervised=True)
    print('are you doing this?')
    train_examples, val_examples = examples['train'], examples['validation']
    return train_examples, val_examples

def load_language_dataset2(path2):
    """
    Loads the language dataset, depending on path 
    it will be based on one language or the other
    
    args: string path . Right now portuguese 

    """
    #path2 = '/Users/gema/Documents/Neural-Machine-Translation/datasets/ru_to_en'
    examples, metadata = tfds.load(path2,
                               with_info=True,
                               as_supervised=True)
    print('are you doing this?')
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
        print('> Examples in Language to translate:')
    for lan in lan_examples.numpy():
        print(lan.decode('utf-8'))
    print()

    print('> Examples in English!!:')
    for en in en_examples.numpy():
        print(en.decode('utf-8'))

load_language_dataset2('/Users/gema/Documents/Neural-Machine-Translation/datasets/~/tensorflow_datasets/ted_hrlr_translate/ru_to_en/.config:1.0.0')
