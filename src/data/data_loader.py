import tensorflow_datasets as tfds
from tensorflow_datasets.datasets.opus.opus_dataset_builder import OpusConfig

from typing import Sequence

from src.core.settings import settings


data_sizes = {
    "train": f"train[:{settings.data_train_size}]",
    "val": f"train[:{settings.data_val_size}]",
    "test": f"train[{settings.data_val_size}:{settings.data_val_size + settings.data_test_size}]",
    "aug": f"train[:{0}]".format(settings.data_aug_size),
}


def add_tokens(ds, src="de", tgt="en", aug=False) -> Sequence:
    """
    Token manipulation function.

    Args:
        ds:
            Dataset with the pairs of the sentences,
            which should be marked with tag
        src:
            Source language
        tgt:
            Target language. Add tags <2tgt> to the sentences.
        aug:
            Special indicator. aug = True means that we get De->Ru
            dataset, which should be marked with the <2uk> tag.

    Returns:
        Dataset of tagged pairs of sentences.
    """
    if aug:
        token = "<2uk> "
    else:
        token = "<2" + tgt + "> "
    ds = ds.map(lambda ex: (token.encode() + ex[src], ex[tgt]))
    return ds


def data_download(
        src: str = "de",
        tgt: str = "en"
) -> tfds.dataset_builders.TfDataBuilder:
    """
    Download the src->tgt corpus.

    Args:
        src:
            Source language.
        tgt:
            Target language.

    Returns:
        src->tgt corpus.
    """
    config_src_tgt = OpusConfig(
        version=tfds.core.Version("0.1.0"),
        language_pair=(src, tgt),
        subsets=["OpenSubtitles"],
    )
    builder_src_tgt = tfds.builder("opus", config=config_src_tgt)
    builder_src_tgt.download_and_prepare()

    return builder_src_tgt


def data_load(
        src: str = "de",
        tgt: str = "en",
        size: str = "train",
        aug: bool = False,
        reverse: bool = False
) -> Sequence:
    """
    Load the src->tgt corpus.

    Args:
        src:
            Source language.
        tgt:
            Target language.
        size:
            Size of the corpus.
        aug:
            Special indicator. aug = True means that we get De->Ru
            dataset, which should be marked with the <2uk> tag.
        reverse:
            reverse = True means that we need <src>-><tgt> corpus.

    Returns:
        Corpus of the sentence pairs.
    """

    if reverse:
        builder_src_tgt = data_download(tgt, src)
    else:
        builder_src_tgt = data_download(src, tgt)

    split_size = data_sizes[size]

    examples_src_tgt = builder_src_tgt.as_dataset(split=split_size)
    examples_src_tgt = add_tokens(
        examples_src_tgt,
        src=src,
        tgt=tgt,
        aug=aug,
    )

    return examples_src_tgt


def make_examples(
        ds: Sequence,
        examples_num: int = 3,
        shuffle: bool = False,
        buffer_size: int = 1
) -> None:
    """
    Show some examples of loaded corpus.
    Args:
        ds:
            Corpus of sentence pairs.
        examples_num:
            Number of sentences pairs to show.
        shuffle:
            If shuffle=true the corpus will be shuffled
            before example extracting.
        buffer_size:
            Buffer shuffle.
    Returns:
        None
    """

    print(f"/n{20*'='} EXAMPLES {20*'='}")
    for src_examples, tgt_examples in (
        (ds.shuffle(buffer_size) if shuffle else ds).batch(examples_num).take(1)
    ):
        for src in src_examples.numpy():
            print(src.decode("utf-8"))

        print()

        for tgt in tgt_examples.numpy():
            print(tgt.decode("utf-8"))

    print(40*'=', '\n')


if __name__ == '__main__':
    ds = data_load('de', 'uk', 'train')
    make_examples(ds, 10, True, 200000)
