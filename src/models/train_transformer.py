import tensorflow as tf
import time
from src.models.transformer import Transformer , CustomSchedule , checkpoints
from src.features.tokenizer_transformer import make_batches

from src.visualization.metrics import loss_function, accuracy_function
from src.data.load_dataset import load_language_dataset 
from src.features.tokenizer_transformer import load_dataset_tokenized
from dvclive import Live
import yaml

# One Stage pipeline. This script will be executed with 'dvc exp run' or 'dvc repro'


with open('params.yaml') as config_file:
    config = yaml.safe_load(config_file)

  ## Set hyperparameters. This will go into the params.yaml file for dvc pipeline. 

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


learning_rate = CustomSchedule(config['train_transformer']['d_model'])

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


temp_learning_rate_schedule = CustomSchedule(config['train_transformer']['d_model'])

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


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


train_examples, val_examples = load_language_dataset(config['load']['model_name'])
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

## Positional embedding is actually declared in transformer

## INITIALIZE DVC LIVE
live = Live()

for epoch in range(config['train_transformer']['EPOCHS']):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate(train_batches):
    train_step(inp, tar)

 ### ------Add metrics to dvc live 
    live.log("train/accuracy", float(train_accuracy.result()))
    live.log("train/loss", float(train_loss.result()))
    
    live.next_step()


    if batch % 50 == 0:
      print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = checkpoints.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

   
  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
  print(f'Training finished')
