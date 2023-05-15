import nltk
import tensorflow_text
import tensorflow as tf

from pathlib import Path

from src.core.settings import settings


def nist_test_by_splitting(hypothesis, reference):
    hyp = hypothesis.split()
    ref = reference.split()

    return nltk.translate.bleu_score.sentence_bleu(
        [ref],
        hyp,
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4,
    )


def test_one_pair_by_splitting(translator, source, reference):

    hypothesis = translator(source).numpy().decode("utf-8")
    hypothesis = hypothesis.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")

    reference = (reference.replace("\n", "")).lower()

    return nist_test_by_splitting(hypothesis, reference)


def test_corpus_by_splitting(translator, test_data):
    sum_nist = 0
    i = 1
    for src_examples, tgt_examples in test_data.batch(2 * settings.data_test_size).take(1):
        for src, tgt in zip(src_examples.numpy(), tgt_examples.numpy()):
            nist = test_one_pair_by_splitting(
                translator,
                src.decode("utf-8"),
                tgt.decode("utf-8")
            )
            sum_nist += nist

    return sum_nist / settings.data_test_size


def nist_test_by_tokenizing(hypothesis, reference):
    hyp = hypothesis.split()
    ref = reference.split()

    return nltk.translate.nist_score.sentence_nist(ref, hyp)


if __name__ == '__main__':
    try:
        from ..data import data_loader
    except ModuleNotFoundError:
        from src.data import data_loader

    import_path = Path(settings.import_dir)

    translator = tf.saved_model.load(import_path / f"translator_{settings.epochs}")
    test_data = data_loader.data_load('de', 'uk', 'test')

    print("Start estimating...")
    result = test_corpus_by_splitting(translator, test_data)
    print(f"\nModel has reached {result:.4f} points of NIST metric.\n")
