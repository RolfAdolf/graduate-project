import tensorflow as tf
import tensorflow_text
import nltk

from pathlib import Path

from src.core.settings import settings


def bleu_test_by_splitting(hypothesis, reference):
    hyp = hypothesis.split()
    ref = reference.split()

    return nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=[1])


def test_one_pair_by_splitting(translator, source, reference):
    hypothesis = translator(source).numpy().decode("utf-8")
    hypothesis = hypothesis.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
    reference = (reference.replace("\n", "")).lower()
    return bleu_test_by_splitting(hypothesis, reference)


def test_corpus_by_splitting(translator, test_data):
    sum_bleu = 0
    for src_examples, tgt_examples in test_data.batch(2 * settings.data_test_size).take(1):
        for src, tgt in zip(src_examples.numpy(), tgt_examples.numpy()):
            bleu = test_one_pair_by_splitting(
                translator,
                src.decode("utf-8"),
                tgt.decode("utf-8")
            )
            sum_bleu += bleu
    return sum_bleu / settings.data_test_size


def bleu_test_by_tokenizing(hypothesis, reference):
    hyp = hypothesis.split()
    ref = reference.split()

    return nltk.translate.bleu_score.sentence_bleu(ref, hyp)


if __name__ == '__main__':
    try:
        from ..transformers import translators
        from ..data import data_loader
    except ModuleNotFoundError:
        from src.data import data_loader
        from src.transformers import translators

    import_path = Path(settings.import_dir)

    translator = tf.saved_model.load(import_path / f"translator_{settings.epochs}")
    test_data = data_loader.data_load('de', 'uk', 'test')

    print("Start estimating...")
    result = test_corpus_by_splitting(translator, test_data)
    print(f"\nModel has reached {result:.4f} points of BLEU metric.\n")
