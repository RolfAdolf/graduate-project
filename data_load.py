###SETUP
import logging
from pickle import FALSE
import time

import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow_datasets.datasets.opus.opus_dataset_builder import OpusConfig

import os
import pathlib
import re
import string

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings


from HyperParams import *


####Class for data downlaod and preprocess:
class Data_Load:
    ###Constructor
    def __init__(self, languages=["de", "en", "ru", "uk"], subsets=["OpenSubtitles"]):
        self.languages = languages
        self.subsets = subsets

    ###Train/Valid/Test/Pretrain dictionary
    DATA_SIZE = {
        "train": "train[:{0}]".format(DATA_TRAIN_SIZE),  ### DE->EN, EN->RU directions
        "val": "train[:{0}]".format(DATA_VAL_SIZE),  ### DE->RU direction
        "test": "train[{0}:{1}]".format(
            DATA_VAL_SIZE, DATA_VAL_SIZE + DATA_TEST_SIZE
        ),  ### DE->RU direction
        "aug": "train[:{0}]".format(DATA_AUG_SIZE),  ### De-->De, Ru-->Ru
    }

    ###Add tokens function:
    def Add_tokens(self, ds, src="de", tgt="en", aug=False, reverse=False):
        if aug:
            token = "<2uk> "
        else:
            token = "<2" + tgt + "> "
        ds = ds.map(lambda ex: (token.encode() + ex[src], ex[tgt]))
        return ds

    ###Data_Download
    def data_download(self, src="de", tgt="en"):
        config_src_tgt = OpusConfig(
            version=tfds.core.Version("0.1.0"),
            language_pair=(src, tgt),
            subsets=self.subsets,
        )
        builder_src_tgt = tfds.builder("opus", config=config_src_tgt)
        builder_src_tgt.download_and_prepare()

        return builder_src_tgt

    ###Data_Load
    def data_load(self, src="de", tgt="en", size="train", aug=False, reverse=False):
        builder_src_tgt = (
            self.data_download(src, tgt)
            if not reverse
            else self.data_download(tgt, src)
        )
        SPLIT = self.DATA_SIZE[size]

        examples_src_tgt = builder_src_tgt.as_dataset(split=SPLIT)

        examples_src_tgt = self.Add_tokens(
            examples_src_tgt, src=src, tgt=tgt, reverse=reverse, aug=aug
        )

        return examples_src_tgt

    ###Show some examples of sentence pairs
    def Make_Examples(self, ds, examples_num=3, shuffle=False, buffer_size=1):
        for src_examples, tgt_examples in (
            (ds.shuffle(buffer_size) if shuffle else ds).batch(examples_num).take(1)
        ):
            for src in src_examples.numpy():
                print(src.decode("utf-8"))

            print()

            for tgt in tgt_examples.numpy():
                print(tgt.decode("utf-8"))

        print()
