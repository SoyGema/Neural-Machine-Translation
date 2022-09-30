### Visualize the token distribution per example in the dataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.features.tokenizer_transformer import load_dataset_tokenized
from src.features.tokenizer_language import make_batches
from src.models.train_transformer import translator , print_translation
from src.data.load_dataset import load_language_dataset


##### TOKENIZATION . unclear if this chall be appear finally on the experiments table. 

lengths = []
tokenizers = load_dataset_tokenized()

model_name = 'ted_hrlr_translate/pt_to_en'
train_examples, val_examples = load_language_dataset(model_name)

for pt_examples, en_examples in train_examples.batch(1024):
  pt_tokens = tokenizers.en.tokenize(pt_examples)
  lengths.append(pt_tokens.row_lengths())
  
  en_tokens = tokenizers.en.tokenize(en_examples)
  lengths.append(en_tokens.row_lengths())
  print('.', end='', flush=True)



all_lengths = np.concatenate(lengths)

plt.hist(all_lengths, np.linspace(0, 500, 101))
plt.ylim(plt.ylim())
max_length = max(all_lengths)
plt.plot([max_length, max_length], plt.ylim())
plt.title(f'Maximum tokens per example: {max_length}');


sentence = 'este é um problema que temos que resolver.'
ground_truth = 'this is a problem we have to solve .'

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)



##### ATTENTION HEADS. 

def plot_attention_head(in_tokens, translated_tokens, attention):
  # The model didn't generate `<START>` in the output. Skip it.
  translated_tokens = translated_tokens[1:]

  ax = plt.gca()
  ax.matshow(attention)
  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(translated_tokens)))

  labels = [label.decode('utf-8') for label in in_tokens.numpy()]
  ax.set_xticklabels(
      labels, rotation=90)

  labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
  ax.set_yticklabels(labels)



head = 0
# Shape: `(batch=1, num_attention_heads, seq_len_q, seq_len_k)`.
attention_heads = tf.squeeze(
  attention_weights['decoder_layer4_block2'], 0)
attention = attention_heads[head]
attention.shape


### Change to function
in_tokens = tf.convert_to_tensor([sentence])
in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
in_tokens = tokenizers.pt.lookup(in_tokens)[0]
in_tokens


def plot_attention_weights(sentence, translated_tokens, attention_heads):
  in_tokens = tf.convert_to_tensor([sentence])
  in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
  in_tokens = tokenizers.pt.lookup(in_tokens)[0]

  fig = plt.figure(figsize=(16, 8))

  for h, head in enumerate(attention_heads):
    ax = fig.add_subplot(2, 4, h+1)

    plot_attention_head(in_tokens, translated_tokens, head)

    ax.set_xlabel(f'Head {h+1}')

  plt.tight_layout()
  plt.show()


if __name__ == '__main__':

  plot_attention_weights(sentence,
                       translated_tokens,
                       attention_weights['decoder_layer4_block2'][0])


  sentence = 'Eu li sobre triceratops na enciclopédia.'
  ground_truth = 'I read about triceratops in the encyclopedia.'

  translated_text, translated_tokens, attention_weights = translator(
      tf.constant(sentence))
  print_translation(sentence, translated_text, ground_truth)

  plot_attention_weights(sentence, translated_tokens,
                       attention_weights['decoder_layer4_block2'][0])