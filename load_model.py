###Setup
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

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings
pwd = pathlib.Path.cwd()


# reloaded = tf.saved_model.load('small-transformer10')

# print(reloaded("<2ru> Er wurde krank, weil er im Winter ohne Schal unterwegs war.").numpy().decode("utf-8"))

# print(reloaded)
# print(type(reloaded))

# print(reloaded.signatures.keys())
