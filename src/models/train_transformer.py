
from turtle import shape
import tensorflow as tf
from src.features.positional_encoding import positional_encoding
from src.models.decoder import Decoder
from src.models.encoder import Encoder
import time
##from src.features import tokenizers --> I think this is the model ?
from src.features.tokenizer_transformer import make_batches

from src.visualization.metrics import loss_function, accuracy_function
from src.data.load_dataset import load_language_dataset 
from src.features.tokenizer_transformer import load_dataset_tokenized
from dvclive import Live
import yaml

model_name = 'ted_hrlr_translate/pt_to_en'
train_examples, val_examples = load_language_dataset(model_name)

#input_vocab_size= 8000
#target_vocab_size = 8000
MAX_TOKENS=128

with open('params.yaml') as config_file:
    config = yaml.safe_load(config_file)


##Define transformer and try it out 

class Transformer(tf.keras.Model):
  def __init__(self,
               *,
               num_layers, # Number of decoder layers.
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               input_vocab_size, # Input (Portuguese) vocabulary size.
               target_vocab_size, # Target (English) vocabulary size.
               dropout_rate=0.1
               ):
    super().__init__()
    # The encoder.
    self.encoder = Encoder(
      num_layers=config['train_transformer']['num_layers'],
      d_model=config['train_transformer']['d_model'],
      num_attention_heads=config['train_transformer']['num_attention_heads'],
      dff=config['train_transformer']['dff'],
      input_vocab_size=config['positional_encoding']['input_vocab_size'],
      dropout_rate=config['train_transformer']['dropout_rate']
      )

    # The decoder.
    self.decoder = Decoder(
      num_layers=config['train_transformer']['num_layers'],
      d_model=config['train_transformer']['d_model'],
      num_attention_heads=config['train_transformer']['num_attention_heads'],
      dff=config['train_transformer']['dff'],
      target_vocab_size=config['positional_encoding']['target_vocab_size'],
      dropout_rate=config['train_transformer']['dropout_rate']
      )

    # The final linear layer.
    self.final_layer = tf.keras.layers.Dense(config['positional_encoding']['target_vocab_size'])

  def call(self, inputs, training):
    # Keras models prefer if you pass all your inputs in the first argument.
    # Portuguese is used as the input (`inp`) language.
    # English is the target (`tar`) language.
    inp, tar = inputs

    # The encoder output.
    enc_output = self.encoder(inp, training)  # `(batch_size, inp_seq_len, d_model)`
    enc_mask = self.encoder.compute_mask(inp)

    # The decoder output.
    dec_output, attention_weights = self.decoder(
        tar, enc_output, enc_mask, training)  # `(batch_size, tar_seq_len, d_model)`

    # The final linear layer output.
    final_output = self.final_layer(dec_output)  # Shape `(batch_size, tar_seq_len, target_vocab_size)`.

    # Return the final output and the attention weights.
    return final_output, attention_weights



  ## Set hyperparameters. This will go into the params.yaml file for dvc pipeline. 
#num_layers = 4
#d_model = 128
#dff = 512
#num_attention_heads = 8
#dropout_rate = 0.1


transformer = Transformer(
    num_layers=config['train_transformer']['num_layers'],
    d_model=config['train_transformer']['d_model'],
    num_attention_heads=config['train_transformer']['num_attention_heads'],
    dff=config['train_transformer']['dff'],
    input_vocab_size=config['positional_encoding']['input_vocab_size'],
    target_vocab_size=config['positional_encoding']['target_vocab_size'],
    dropout_rate=config['train_transformer']['dropout_rate'])


  ## Test
input = tf.constant([[1,2,3, 4, 0, 0, 0]])
target = tf.constant([[1,2,3, 0]])

x, attention = transformer((input, target))
print('----TEST THE TRANSFORMER----')
print(x.shape)
print(attention['decoder_layer1_block1'].shape)
print(attention['decoder_layer4_block2'].shape)
print('----TRANSFORMER TESTED----')
transformer.summary()

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)   
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



learning_rate = CustomSchedule(config['train_transformer']['d_model'])

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


temp_learning_rate_schedule = CustomSchedule(config['train_transformer']['d_model'])

#plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
#plt.ylabel('Learning Rate')
#plt.xlabel('Train Step')


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


checkpoint_path = './checkpoints/train'

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# If a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')


train_step_signature = [
    (
         tf.TensorSpec(shape=(None, None), dtype=tf.int64),
         tf.TensorSpec(shape=(None, None), dtype=tf.int64)),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

# The `@tf.function` trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

@tf.function(input_signature=train_step_signature)
def train_step(inputs, labels):
  (inp, tar_inp) = inputs
  tar_real = labels

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],
                                 training = True)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))


EPOCHS = 1
##!!!train_batches=

train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

## INITIALIZE DVC LIVE
live = Live()

for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate(train_batches):
    train_step(inp, tar)

 ### ------Add metrics to dvc live . NOT TESTED--------- FROM DOCS IM ASSUMMING THAT WE HAVE TO DEFINE IT IN THE TRAINING STAGE -----
    live.log("accuracy_train", float(train_accuracy.result()))
    live.log("loss_train", float(train_loss.result()))

    #for acc, train_accuracy in metrics.items():
      #live.log(acc, train_accuracy)

    #for loss, train_loss in metrics.items():
      #live.log(loss, train_loss)

    live.next_step()

 ### ------NOT TESTED--------- 

    if batch % 50 == 0:
      print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

   
  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
  print(f'Training finished')



tokenizers = load_dataset_tokenized()

##### EXPORT MODEL

##### TRANSLATE THE MODEL. PROBABLY GOING FOR predict_transformer.py

class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length=config['tokenizer_transformer']['MAX_TOKENS']):
    # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

    encoder_input = sentence

    # As the output language is English, initialize the output with the
    # English `[START]` token.
    start_end = self.tokenizers.en.tokenize([''])[0]
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions, _ = self.transformer([encoder_input, output], training=False)

      # Select the last token from the `seq_len` dimension.
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

      predicted_id = tf.argmax(predictions, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to the
      # decoder as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # The output shape is `(1, tokens)`.
    text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.

    tokens = tokenizers.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop.
    # Therefore, recalculate them outside the loop.
    _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)

    return text, tokens, attention_weights




class ExportTranslator(tf.Module):
  def __init__(self, translator):
    self.translator = translator

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def __call__(self, sentence):
    (result,
     tokens,
     attention_weights) = self.translator(sentence, max_length=config['train_transformer']['MAX_TOKENS'])

    return result


def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
  print(f'{"Ground truth":15s}: {ground_truth}')


sentence = 'este é um problema que temos que resolver.'
ground_truth = 'this is a problem we have to solve .'

translator = Translator(tokenizers, transformer)
translator_exported = ExportTranslator(translator)


#translated_text, translated_tokens, attention_weights = translator(
    #tf.constant(sentence))
#print_translation(sentence, translated_text, ground_truth)