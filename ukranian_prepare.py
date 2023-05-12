###SETUP
import logging
from sre_parse import expand_template
import time

import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

import os
import pathlib
import re
import string

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings
pwd = pathlib.Path.cwd()

from HyperParams import *
from data_load import Data_Load
from Tokenizer import tokenizer_trainer
import re

"""
###Data downloading and loading
data_loader = Data_Load(languages = ["de", "en", "ru"], subsets = ["OpenSubtitles"])

print("=#=#=#=#=#=#=#=#=#=#=#=#=DE-->UK=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")
config_src_tgt = tfds.translate.opus.OpusConfig(
            version=tfds.core.Version('0.1.0'),
            language_pair=('de', 'uk'),
            subsets=["OpenSubtitles"]
            )

builder_src_tgt = tfds.builder("opus", config=config_src_tgt)
builder_src_tgt.download_and_prepare()

examples_src_tgt = builder_src_tgt.as_dataset()['train']
print(examples_src_tgt)
print(type(examples_src_tgt))

examples_src_tgt = examples_src_tgt.map(lambda ex: ('<2uk> '.encode()+ex['de'], ex['uk']))
data_loader.Make_Examples(examples_src_tgt, 20)

print(len(examples_src_tgt))


fullstring = "pythonist"
substring = "python"
if re.findall(r'[it]', 'pythonist'):
    print('Found')
else:
    print('Not found')

i = 0
count_uk = 0

for src_examples, tgt_examples in examples_src_tgt.batch(len(examples_src_tgt)).take(1): #(ds.shuffle(buffer_size) if shuffle else ds).batch(examples_num).take(1):

    for tgt in tgt_examples.numpy():
        i += 1

        if re.findall(r'[ґєії]', tgt.decode('utf-8')):
            count_uk+=1
        if re.findall(r'[ґєії]', (tgt.decode('utf-8')).lower()):
                print(i, count_uk)
                print(tgt.decode('utf-8'))
                print("Found")
                print("================================================================")
        if (count_uk == 100000):
            print(i, count_uk)
            break

print("================================================================")
print("================================================================")
print(count_uk, i)

"""


def ua_filter(src, tgt):
    return (
        True
        if re.findall(r"[ґєії]", ((tgt.numpy()).decode("utf-8")).lower())
        else False
    )


def filter_ua(src, tgt):
    return tf.py_function(ua_filter, inp=[src, tgt], Tout=(tf.bool))


"""
###Making pretrain dataset
def prep_data(dataset, direction=1):
	if (direction):
		return dataset.map(lambda src, tgt: tf.py_function(make_pair, inp=[tgt], Tout=(tf.string, tf.string)))
	else:
		return dataset.map(lambda src, tgt: tf.py_function(make_pair, inp=[src], Tout=(tf.string, tf.string)))

def deletes(src, tgt):
  return (tgt.numpy()).decode()!='delete'

def filter_deletes(src, tgt):
	return tf.py_function(deletes, inp=[src, tgt], Tout=(tf.bool))
examples_src_tgt = examples_src_tgt.filter(filter_ua)

print("Allright")

data_loader.Make_Examples(examples_src_tgt, 100, True, 300000)
"""
