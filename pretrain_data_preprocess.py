###SETUP
import logging
import time

import numpy as np
import nltk

import tensorflow_datasets as tfds
import tensorflow as tf

import os
import pathlib
import re
import string

# nltk.download('punkt')
logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings


from HyperParams import *


###masking
def choose_masked_word(list_of_words):
    if len(list_of_words) >= PRETRAIN_WORDS_NUMBER:
        # Choice making word in sentence
        n = np.random.randint(0, len(list_of_words))
        masked_word = list_of_words[n]
        return masked_word
    else:
        return "delete"


###Make pair
def make_pair(sentence):
    sentence = ((sentence.numpy()).decode("utf-8")).lower()
    sentence_ = sentence.translate(str.maketrans("", "", string.punctuation + "—"))
    ###Split the sentence
    words = nltk.word_tokenize(sentence_)
    list_of_words = list(set(words))

    ###masking
    ##Choose
    masked_word = choose_masked_word(list_of_words)

    if masked_word != "delete":
        sentence = re.sub(r"\b{}\b".format(masked_word), "[MASK]", sentence)
    else:
        sentence = "delete"

    ###Return pair
    return (sentence.encode("utf-8"), masked_word.encode("utf-8"))


###Making pretrain dataset
def prep_data(dataset, direction=1):
    if direction:
        return dataset.map(
            lambda src, tgt: tf.py_function(
                make_pair, inp=[tgt], Tout=(tf.string, tf.string)
            )
        )
    else:
        return dataset.map(
            lambda src, tgt: tf.py_function(
                make_pair, inp=[src], Tout=(tf.string, tf.string)
            )
        )


def deletes(src, tgt):
    return (tgt.numpy()).decode() != "delete"


def filter_deletes(src, tgt):
    return tf.py_function(deletes, inp=[src, tgt], Tout=(tf.bool))
