import tensorflow as tf

from typing import Callable, Sequence

from src.core.settings import settings


def filter_max_tokens(
        src: object,
        tgt: object
) -> bool:
    """
    Delete too long sentences.
    Args:
        src: Source sentence.
        tgt: Target sentence.
    Returns:
        If false - delete pairs.
    """
    num_tokens = tf.maximum(tf.shape(src)[1], tf.shape(tgt)[1])
    return num_tokens < settings.max_tokens


def tokenize_pairs_wrapper(tokenizers: object) -> Callable:
    """
    Make function based on tokenizers.
    Args:
        tokenizers:
            Loaded tokenizers.
    Returns:
        Callable
    """
    def tokenize_pairs(src, tgt):
        nonlocal tokenizers
        src = tokenizers.src.tokenize(src)
        # Convert from ragged to dense, padding with zeros.
        src = src.to_tensor()

        tgt = tokenizers.tgt.tokenize(tgt)
        # Convert from ragged to dense, padding with zeros.
        tgt = tgt.to_tensor()
        return src, tgt
    return tokenize_pairs


def make_batches(
        ds: Sequence,
        tokenizers: object
) -> object:
    """
    Split dataset on the batches of specified size.
    Args:
        ds:
            Corpus of sentence pairs.
        tokenizers:
            loaded tokenizers.
    Returns:
        Dataset which is sliced on batches.
    """
    tokenize_pairs = tokenize_pairs_wrapper(tokenizers=tokenizers)
    return (
        ds.cache()
        .shuffle(settings.buffer_size)
        .batch(settings.batch_size)
        .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(filter_max_tokens)
        .prefetch(tf.data.AUTOTUNE)
    )
