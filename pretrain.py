###Setups
import logging
import time

import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

# Import tf_text to load the ops used by the tokenizer saved model
import tensorflow_text  # pylint: disable=unused-import

###TOKENIZER
import collections
import os
import pathlib
import re
import string
import sys
import tempfile

from HyperParams import *
from data_load import Data_Load
from Tokenizer import *
from Transformer import *
from save_model import *
from pretrain_data_preprocess import *

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings
pwd = pathlib.Path.cwd()

###DATA
data_loader = Data_Load(languages=["de", "en", "ru"], subsets=["OpenSubtitles"])

pret_data = data_loader.data_load("de", "ru", size="pretrain")
pret_val = data_loader.data_load("de", "ru", size="val")

pretrain_examples = prep_data(pret_data).filter(filter_deletes)
pretrain_val_examples = prep_data(pret_val).filter(filter_deletes)

data_loader.Make_Examples(pretrain_examples)
data_loader.Make_Examples(pretrain_val_examples)

###Define the tokenizer
tokenizers = tf.Module()
tokenizers.src = CustomTokenizer(reserved_tokens, src_file)
tokenizers.tgt = CustomTokenizer(reserved_tokens, tgt_file)


###Define functions to drop the examples longer than MAX_TOKENS:
def filter_max_tokens(src, tgt):
    num_tokens = tf.maximum(tf.shape(src)[1], tf.shape(tgt)[1])
    return num_tokens < MAX_TOKENS


def tokenize_pairs(src, tgt):
    src = tokenizers.src.tokenize(src)
    # Convert from ragged to dense, padding with zeros.
    src = src.to_tensor()

    tgt = tokenizers.tgt.tokenize(tgt)
    # Convert from ragged to dense, padding with zeros.
    tgt = tgt.to_tensor()
    return src, tgt


def make_batches(ds):
    return (
        ds.cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(filter_max_tokens)
        .prefetch(tf.data.AUTOTUNE)
    )


###Train and validation datasets
train_batches = make_batches(pretrain_examples)
val_batches = make_batches(pretrain_val_examples)


###Optimizer
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)
temp_learning_rate_schedule = CustomSchedule(d_model)


###Loss and metrics
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")


###Transformer
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.src.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.tgt.get_vocab_size().numpy(),
    rate=dropout_rate,
)


####TRAINING####
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp], training=True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> portuguese, tar -> english
    for batch, (inp, tar) in enumerate(train_batches):
        train_step(inp, tar)

        if batch % 100 == 0:
            print(
                f"Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
            )

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f"Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}")

        translator = Translator(tokenizers, transformer)

        # Translation examples
        translated_text, translated_tokens, attention_weights = translator(
            tf.constant(sentence_en)
        )
        print_translation(sentence_en, translated_text, ground_truth_en)
        translated_text, translated_tokens, attention_weights = translator(
            tf.constant(sentence_ru)
        )
        print_translation(sentence_ru, translated_text, ground_truth_ru)

        # Save model
        translator_save_model = ExportTranslator(translator)

        tf.saved_model.save(
            translator_save_model, export_dir=export_dir + str(epoch + 1)
        )

    print(
        f"Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
    )

    print(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")
