import logging
import time

import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text

import collections
import os
import pathlib
import re
import string
import sys
import tempfile

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

from HyperParams import *
from data_load import Data_Load

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings
pwd = pathlib.Path.cwd()


class tokenizer_trainer:
    ###Constructor
    def __init__(self):
        self.vocab_size = vocab_size
        self.reserved_tokens = reserved_tokens
        self.bert_vocab_args = bert_vocab_args
        self.src_file = src_file
        self.tgt_file = tgt_file

    ###Train
    def train(self, train_data):
        # %%time - IPython expression
        vocab = bert_vocab.bert_vocab_from_dataset(
            train_data.batch(1000).prefetch(2), **self.bert_vocab_args
        )

        return vocab

    ###Write a vocabulary file:
    @staticmethod
    def writeFile(filepath, vocab):
        print("file writing")
        with open(filepath, "w", encoding="utf-8") as f:
            for token in vocab:
                print(token, file=f)

    ###Train target and source vocabs
    def src_tgt_vocab(self, src_data, tgt_data):
        print("src tokenizer training")
        if not (os.path.exists(self.src_file)):
            src_vocab = self.train(src_data)

            self.writeFile(self.src_file, src_vocab)

        print("tgt tokenizer training")
        if not (os.path.exists(self.tgt_file)):
            tgt_vocab = self.train(tgt_data)

            self.writeFile(self.tgt_file, tgt_vocab)


START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")


# Customization
def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)


def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[UNK]".
    bad_tokens = [
        re.escape(tok)
        for tok in reserved_tokens
        if tok not in ["[UNK]", "<2en>", "<2pt>"]
    ]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=" ", axis=-1)

    return result


###Class for custom tokenizer
class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = tensorflow_text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text(encoding="utf-8").splitlines()
        self.vocab = tf.Variable(vocab)

        ## Create the signatures for export:

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string)
        )

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
        )
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
        )

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
        )
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
        )

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
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
