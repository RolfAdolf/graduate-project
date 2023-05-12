import logging
import pathlib

from data_load import Data_Load
from Tokenizer import tokenizer_trainer

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings
pwd = pathlib.Path.cwd()

###Data downloading and loading
data_loader = Data_Load(
    languages=["de", "en", "ru", "uk", "pt", "es"], subsets=["OpenSubtitles"]
)

"""
####TRAIN####
###De-->En
print("=#=#=#=#=#=#=#=#=#=#=#=#=DE-->EN=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")
train_examples_de_en = data_loader.data_load('de', 'en')
data_loader.Make_Examples(train_examples_de_en)



###En-->Ru
print("=#=#=#=#=#=#=#=#=#=#=#=#=EN-->RU=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")
train_examples_en_ru = data_loader.data_load('en', 'ru', 'aug')
data_loader.Make_Examples(train_examples_en_ru)


###En-->Uk
print("=#=#=#=#=#=#=#=#=#=#=#=#=EN-->UK=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")
train_examples_en_uk = data_loader.data_load('en', 'uk')
data_loader.Make_Examples(train_examples_en_uk)


###VALIDATION
print("=#=#=#=#=#=#=#=#=#=#=#=#=DE-->UK=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")
val_examples = data_loader.data_load('de', 'uk', size = 'val')
data_loader.Make_Examples(val_examples)
"""


####TRAIN####
##Ru-->En
print("=#=#=#=#=#=#=#=#=#=#=#=#=Ru-->En=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")
train_examples_de_en = data_loader.data_load("de", "en")
data_loader.Make_Examples(train_examples_de_en)


###En-->Pt
print("=#=#=#=#=#=#=#=#=#=#=#=#=En-->Pt=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")
train_examples_en_uk = data_loader.data_load("en", "uk")
data_loader.Make_Examples(train_examples_en_uk)


###Ru-->Es
print("=#=#=#=#=#=#=#=#=#=#=#=#=Ru-->Es=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")
train_examples_de_ru = data_loader.data_load("de", "ru", size="aug", aug=True)
data_loader.Make_Examples(train_examples_de_ru)


###VALIDATION
print("=#=#=#=#=#=#=#=#=#=#=#=#=Ru-->Pt=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")
val_examples = data_loader.data_load("de", "uk", size="val")
data_loader.Make_Examples(val_examples)


train_tok = train_examples_de_en.concatenate(train_examples_en_uk)
train_tok = train_tok.concatenate(train_examples_de_ru)

train_tok = train_tok.concatenate(val_examples)
print(len(train_tok))

print("=#=#=#=#=#=#=#=#=#=#=#=#=EXAMPLES=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")
data_loader.Make_Examples(
    train_tok, shuffle=True, examples_num=20, buffer_size=len(train_tok)
)
print("=#=#=#=#=#=#=#=#=#=#=#=#=EXAMPLES=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")


train_src = train_tok.map(lambda src, tgt: src)
train_tgt = train_tok.map(lambda src, tgt: tgt)


###Tokenizer
tok_trainer = tokenizer_trainer()

print(
    "=#=#=#=#=#=#=#=#=#=#=#=#=TOKENIZER TRAINING=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#"
)

tok_trainer.src_tgt_vocab(train_src, train_tgt)


# ####TRAIN####
# ###De->En
# """
# OpenSubtitles[de-en] consists 18.8M of de-en sentence pairs.
# Make a train De->En dataset with size of DATA_TRAIN_SIZE.
# Insert <2en>-tokens at the beginning of each sentence.
# Train sentences: DATA_TRAIN_SIZE from HyperParams
# """
# train_examples_de_en = data_loader.data_load('de', 'en')
# data_loader.Make_Examples(train_examples_de_en)

# ###En->Ru
# """
# OpenSubtitles[en-ru] consists 21.2M of en-ru sentence pairs.
# Make a train En->Ru dataset with size of DATA_TRAIN_SIZE.
# Insert <2ru>-tokens at the beginning of each sentence.
# Train sentences: DATA_TRAIN_SIZE from HyperParams
# """
# train_examples_en_ru = data_loader.data_load('en', 'ru')
# data_loader.Make_Examples(train_examples_en_ru)

# ####VALIDATION AND PRETRAIN####
# ###De->Ru
# """
# OpenSubtitles[de-ru] consists 9.0M of de-ru sentence pairs.
# Make a train De->Ru dataset with size of DATA_TRAIN_SIZE.
# Insert <2ru>-tokens at the beginning of each sentence.
# Train sentences: DATA_TRAIN_SIZE from HyperParams
# """
# val_examples = data_loader.data_load('de', 'ru', size = 'val')
# data_loader.Make_Examples(val_examples)


# ###PRETRAIN (Ru only)
# ###I extracted ru segment from the same De->Ru dataset, which was used in test and validation
# pret_data = data_loader.data_load('de', 'ru', size = 'pretrain')
# pretrain_examples = prep_data(pret_data).filter(filter_deletes)
# data_loader.Make_Examples(pretrain_examples)

# ##Sample train and validation data.
# #We are merging 2 datasets (De->En and En->Ru)
# #val_examples = data_loader.data_load('de', 'ru', size = 'val')
# #train_examples = train_examples_de_en.concatenate(train_examples_en_ru)


# ####TOKENIZER TRAINING
# ###DATA
# tok_train_examples = train_examples_de_en.concatenate(train_examples_en_ru)
# tok_train_examples = tok_train_examples.concatenate(val_examples)
# print(len(tok_train_examples))
# tok_train_examples = tok_train_examples.concatenate(pretrain_examples)

# print("=#=#=#=#=#=#=#=#=#=#=#=#=EXAMPLES=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")
# data_loader.Make_Examples(tok_train_examples, shuffle=True, examples_num=20, buffer_size = BUFFER_SIZE)
# print("=#=#=#=#=#=#=#=#=#=#=#=#=EXAMPLES=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")

# train_src = tok_train_examples.map(lambda src, tgt: src)
# train_tgt = tok_train_examples.map(lambda src, tgt: tgt)

# ###Tokenizer
# tok_trainer = tokenizer_trainer()

# print("=#=#=#=#=#=#=#=#=#=#=#=#=TOKENIZER TRAINING=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#")

# tok_trainer.src_tgt_vocab(train_src, train_tgt)
