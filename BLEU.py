###Setup
import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

import nltk

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

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings
pwd = pathlib.Path.cwd()


class BLEU_tester:
    ###Constructor
    def __init__(self, translator):
        self.experimental = translator

    ###BLEU
    def BLEU_test_by_splitting(self, hypothesis, reference):
        ###Tokenize sentences
        hyp = hypothesis.split()
        ref = reference.split()

        ###BLEU counting
        return nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=[1])

    ###One-pair tester:
    def test_one_pair_by_splitting(self, source, reference):
        ###Make a hypothesis
        hypothesis = ((self.experimental(source)).numpy()).decode("utf-8")
        hypothesis = (
            ((hypothesis.replace(" .", ".")).replace(" ,", ",")).replace(" ?", "?")
        ).replace(" !", "!")
        print(source)
        print(hypothesis)
        reference = (reference.replace("\n", "")).lower()
        print(reference)
        ###Result:
        return self.BLEU_test_by_splitting(hypothesis, reference)

    ###Test on corpus
    def test_corpus_by_splitting(self, test_data):
        ###sum of BLEU points
        sum_BLEU = 0
        i = 1
        ###BLEU score counting
        ###SAMPLE
        for src_examples, tgt_examples in test_data.batch(2 * DATA_TEST_SIZE).take(1):
            for src, tgt in zip(src_examples.numpy(), tgt_examples.numpy()):
                print("+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+")
                print(i)
                bleu = self.test_one_pair_by_splitting(
                    src.decode("utf-8"), tgt.decode("utf-8")
                )
                sum_BLEU += bleu
                print(bleu)
                print("+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+")
                i += 1

        return sum_BLEU / DATA_TEST_SIZE

        ###BLEU

    def BLEU_test_by_tokenizing(hypothesis, reference):
        ###Tokenize sentences
        hyp = hypothesis.split()
        ref = reference.split()

        ###BLEU counting
        return nltk.translate.bleu_score.sentence_bleu(ref, hyp)
