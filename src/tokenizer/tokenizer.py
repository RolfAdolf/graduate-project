from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow as tf
import tensorflow_text

import os
import pathlib
from pathlib import Path
import re
from typing import Sequence, Union, List

from src.core.settings import settings


pathLike = Union[os.PathLike, str]


START = tf.argmax(tf.constant(settings.reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(settings.reserved_tokens) == "[END]")


bert_tokenizer_params = {'lower_case': True}

bert_vocab_args = dict(
    vocab_size=settings.vocab_size,
    reserved_tokens=settings.reserved_tokens,
    bert_tokenizer_params=bert_tokenizer_params,
    learn_params={},
)


def train(
        train_data: Sequence
) -> object:
    """
    Train your own tokenizer.
    Args:
        train_data:
            Sequence of the sentences.
    Returns:
        Vocabulary of tokenizer
    """
    vocab = bert_vocab.bert_vocab_from_dataset(
        train_data.batch(1000).prefetch(2), **bert_vocab_args
    )

    return vocab


def write_file(filepath: pathLike, vocab: Sequence) -> None:
    """
    Write a vocabulary file.
    Args:
        filepath:
            Path to a vocabulary file
        vocab:
            Vocabulary to write
    Returns:
        None
    """

    filepath = Path(filepath)

    # Create folder which consist vocabulary file
    parent = filepath.parent
    if not parent.exists():
        print(f"Creating directory {parent}...")
        filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        print(f"Writing file in {filepath}\n")
        for token in vocab:
            print(token, file=f)


def train_and_write(
        data: Sequence,
        file_to_write: pathLike,
    ) -> None:
    """
    Train tokenizer and write it to a file.
    Args:
        data:
            Sentences to train tokenizer.
        file_to_write:
            Path to the file to write
            tokenizer vocabulary.
    Returns:
        None
    """
    file_to_write = Path(file_to_write)
    name = file_to_write.stem

    print(f"Training tokenizer for {name}...")
    if not (os.path.exists(file_to_write)):
        src_vocab = train(data)
        write_file(file_to_write, src_vocab)
    else:
        print(f"You already have tokenizer for {name}.\n")


def src_tgt_train(
        src_data: Sequence,
        tgt_data: Sequence,
) -> None:
    """
    Train source and target vocabularies.
    Args:
        src_data:
            Sequences of the sentences on the
            source language.
        tgt_data:
            Sequences of the sentences on the
            target language.
    Returns:
        None
    """
    train_and_write(src_data, settings.src_file)
    train_and_write(tgt_data, settings.tgt_file)


def add_start_end(ragged: str) -> object:
    """
    Add [START] and [END] tokens.

    Args:
        ragged:
            Sentence.

    Returns:
        None
    """
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)


def cleanup_text(
        reserved_tokens: List,
        token_txt
) -> object:
    """
    Clean the text.
    Args:
        reserved_tokens:
        token_txt:

    Returns:
        Filtered data.
    """
    # Drop the reserved tokens, except for "[UNK]", "<2en>", "<2uk>".
    bad_tokens = [
        re.escape(tok)
        for tok in reserved_tokens
        if tok not in ["[UNK]", "<2en>", "<2uk>"]
    ]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=" ", axis=-1)

    return result


class CustomTokenizer(tf.Module):
    """Class for custom tokenizer."""
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = tensorflow_text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text(encoding="utf-8").splitlines()
        self.vocab = tf.Variable(vocab)

        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string)
        )

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
