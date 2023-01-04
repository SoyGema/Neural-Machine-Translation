import yaml
from src.models.decoder import Decoder
from src.models.encoder import Encoder
import tensorflow as tf


with open('params.yaml') as config_file:
    config = yaml.safe_load(config_file)

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
      num_layers=config['train_transformer']['num_layers_encoder'],
      d_model=config['train_transformer']['d_model'],
      num_attention_heads=config['train_transformer']['num_attention_heads_encoder'],
      dff=config['train_transformer']['dff'],
      input_vocab_size=config['positional_encoding']['input_vocab_size'],
      dropout_rate=config['train_transformer']['dropout_rate']
      )

    # The decoder.
    self.decoder = Decoder(
      num_layers=config['train_transformer']['num_layers_decoder'],
      d_model=config['train_transformer']['d_model'],
      num_attention_heads=config['train_transformer']['num_attention_heads_decoder'],
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


def checkpoints():
  """Creates checkpoints for the trained model"""
  checkpoint_path = './checkpoints/train'
  ckpt = tf.train.Checkpoint(transformer=transformer,
                            optimizer=optimizer)

  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
  # If a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
  return ckpt_manager