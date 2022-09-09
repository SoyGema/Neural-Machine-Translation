import tensorflow as tf



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