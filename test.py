from data_load import Data_Load
import numpy as np
import nltk
import string
import tensorflow as tf
import tensorflow_datasets as tfds
import re
from HyperParams import *
from BLEU import *

translator = tf.saved_model.load("big_base/transformer_with_augmentation10")

bt = BLEU_tester(translator)

data_loader = Data_Load(languages=["de", "en", "ru"], subsets=["OpenSubtitles"])

# test_data_de = data_loader.data_load('de', 'ru', size = 'test')
# test_data_ru = data_loader.data_load('de', 'ru', size = 'test2')

test_data = data_loader.data_load("de", "uk", size="test")

data_loader.Make_Examples(test_data, 10, True, DATA_TEST_SIZE)


print(bt.test_corpus_by_splitting(test_data))
