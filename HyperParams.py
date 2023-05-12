###The size of train dataset of each language. It is important to use datasets with the same sizes.
DATA_TRAIN_SIZE = 200000

###The size of validation data.
DATA_VAL_SIZE = 5000

###DATA_TEST_SIZE
DATA_TEST_SIZE = 20000

###Augmentation number
DATA_AUG_SIZE = 50000

###minimum words num in sentence to pretrain
PRETRAIN_WORDS_NUMBER = 3

###Num of maximum tokens in one sentences
MAX_TOKENS = 128

###Vocabulary of tokenizer size
vocab_size = 16000

###Reserved tokens
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]", "<2en>", "<2ru>", "<2uk>"]

###Tokenizer params
bert_tokenizer_params = dict(lower_case=True)

###Vocabulary arguments
bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size=vocab_size,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)

###Source vocablury file name:
src_file = "src_vocab.txt"
###Target vocablury file name:
tgt_file = "tgt_vocab.txt"

###Train data parameters
BUFFER_SIZE = 2 * DATA_TRAIN_SIZE + DATA_AUG_SIZE
BATCH_SIZE = 256


###Transformer parameters
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

EPOCHS = 5

###Model save
export_dir = "transformer_with_augmentation"
import_dir = "transformer_with_augmentation"
# small-transformer: 4, 128, 512, 8, 0.1


###Translation example
sentence_en = "<2uk> Er wurde krank, weil er im Winter ohne Hut ging"
ground_truth_en = "Він захворів, тому що взимку ходив без шапки"
