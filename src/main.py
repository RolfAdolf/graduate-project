import tensorflow as tf

from copy import copy
import time
from pathlib import Path
from typing import Callable

from src.core.settings import settings
from src.data import data_loader
from src.tokenizer import tokenizer
from src.utils import utils
from src.transformers import translators
from src.transformers import transformers
from src.testing import bleu
from src.testing import nist


arguments = [
    ('de', 'en', 'train', False, False),
    ('en', 'uk', 'train', False, False),
    ('de', 'ru', 'aug', True, False),
    ('de', 'uk', 'val', False, False),
]
if not settings.data_aug_size:
    arguments.pop(2)


learning_rate = transformers.CustomSchedule(settings.d_model)

optimizer = tf.keras.optimizers.legacy.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

temp_learning_rate_schedule = transformers.CustomSchedule(settings.d_model)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")


def load_datasets(
        separate: bool = True
) -> object:

    arguments_ = copy(arguments)
    if separate:
        val_examples = data_loader.data_load(*arguments_.pop())

    train_examples = None
    for args in arguments_:
        if not train_examples:
            train_examples = data_loader.data_load(*args)
            continue
        train_examples = train_examples.concatenate(data_loader.data_load(*args))

    if separate:
        print(f"You have {len(train_examples)} sentences in train dataset and {len(val_examples)} in validation.")
        return train_examples, val_examples
    print(f"You have {len(train_examples)} sentences in train dataset.")
    return train_examples


def prepare_datasets(tokenizers):
    train_examples, val_examples = load_datasets()

    data_loader.make_examples(train_examples, 10, True, settings.buffer_size)

    train_batches = utils.make_batches(train_examples, tokenizers)
    val_batches = utils.make_batches(val_examples, tokenizers)

    return train_batches, val_batches


def train_tokenizer() -> object:

    train_tok = load_datasets(separate=False)
    train_src = train_tok.map(lambda src, tgt: src)
    train_tgt = train_tok.map(lambda src, tgt: tgt)

    tokenizer.src_tgt_train(train_src, train_tgt)

    tokenizers = tf.Module()
    tokenizers.src = tokenizer.CustomTokenizer(
        settings.reserved_tokens,
        settings.src_file
    )
    tokenizers.tgt = tokenizer.CustomTokenizer(
        settings.reserved_tokens,
        settings.tgt_file
    )

    return tokenizers


def checkpoint_manage(transformer, optimizer_):
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer_)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt,
        settings.checkpoint_path,
        max_to_keep=5
    )
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")

    return ckpt_manager


def train_step_wrapper(transformer) -> Callable:

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp], training=True)
            loss = transformers.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(transformers.accuracy_function(tar_real, predictions))

    return train_step


def save_transformer(transformer, translator, epoch):

    epoch += 1

    export_dir = Path(settings.export_dir)
    export_dir.mkdir(exist_ok=True, parents=True)

    keras_dir = str(export_dir / f"keras_model_trained_{epoch}")
    weights_filename = str(export_dir / f"weights_{epoch}/weights")
    translator_dir = str(export_dir / f"translator_{epoch}")

    transformer.save(keras_dir)
    transformer.save_weights(weights_filename)

    # Save model
    translator_save_model = translators.ExportTranslator(translator)

    tf.saved_model.save(
        translator_save_model, export_dir=translator_dir
    )


def train(
        transformer,
        train_batches,
        val_batches,
        tokenizers,
        ckpt_manager,
):
    translator = None
    for epoch in range(settings.epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        train_step = train_step_wrapper(transformer=transformer)

        for batch, (inp, tar) in enumerate(train_batches):
            train_step(inp, tar)

            if batch % 100 == 0:
                print(
                    f"Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
                )
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f"Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}")

        if (epoch + 1) % 5 == 0:
            translator = translators.Translator(tokenizers, transformer)

            translated_text, translated_tokens, attention_weights = translator(
                tf.constant(settings.source_sentence)
            )
            translators.print_translation(
                settings.source_sentence,
                translated_text,
                settings.ground_truth
            )

            save_transformer(transformer, translator, epoch)

        print(
            f"Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
        )

        print(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")

    return translator


def test_translator(translator, data):

    result_bleu = bleu.test_corpus_by_splitting(translator, data)
    result_nist = nist.test_corpus_by_splitting(translator, data)

    print(f"\nModel has reached {result_bleu:.4f} points of BLEU metric.")
    print(f"Model has reached {result_nist:.4f} points of NIST metric.\n")


def main():

    tokenizers = train_tokenizer()

    train_batches, val_batches = prepare_datasets(tokenizers)

    transformer = transformers.return_transformer(tokenizers)

    ckpt_manager = checkpoint_manage(transformer, optimizer)

    translator = train(
        transformer=transformer,
        train_batches=train_batches,
        val_batches=val_batches,
        tokenizers=tokenizers,
        ckpt_manager=ckpt_manager
    )

    test_data = data_loader.data_load('de', 'uk', settings.data_test_size)
    test_translator(translator, test_data)


if __name__ == '__main__':
    main()
