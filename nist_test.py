from data_load import Data_Load
import numpy as np
import nltk
import string
import tensorflow as tf
import tensorflow_datasets as tfds
import re
from HyperParams import *

print("0" * 50)
from NIST import *

print("1" * 50)
translator = tf.saved_model.load("big_Updated/transformer_with_augmentation10")
print("2" * 50)

nt = NIST_tester(translator)

print("3" * 50)
data_loader = Data_Load(languages=["de", "en", "ru"], subsets=["OpenSubtitles"])
print("4" * 50)
# test_data_de = data_loader.data_load('de', 'ru', size = 'test')
# test_data_ru = data_loader.data_load('de', 'ru', size = 'test2')

test_data = data_loader.data_load("de", "uk", size="test")
print("5" * 50)
data_loader.Make_Examples(test_data, 10, True, DATA_TEST_SIZE)
print("6" * 50)


print(nt.test_corpus_by_splitting(test_data))
