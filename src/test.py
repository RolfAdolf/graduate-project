import tensorflow_text
import tensorflow as tf
from jinja2 import Environment, FileSystemLoader, FunctionLoader

import os
from pathlib import Path
from typing import List

from src.core.settings import settings


def preprocess(sentences: List[str]) -> List[str]:
    return [sentence.replace('\n', '') for sentence in sentences]


def return_sentences(path: os.PathLike) -> str:

    path = Path(path)
    with open(path) as file:
        sentences = file.readlines()

    return preprocess(sentences)


def return_filename(path: os.PathLike) -> str:
    path = Path(path)
    return str(path.parent / path.stem) + '_translation.txt'


def return_translator(path: os.PathLike) -> object:
    print("Loading the translator model...")
    translator = tf.saved_model.load(path)
    print("Model has been loaded successfully.\n")
    return translator


def test_transformer(
        translator: object,
        sentences: List[str]
) -> List[str]:

    translations = []
    filename = return_filename(settings.test_sentences)

    with open(filename, 'w') as file:
        for sent in sentences:
            translation = translator(sent).numpy().decode() + '\n'
            file.write(translation)
            translations.append(translation)

    return translations


def print_translations(sentences, translations):
    file_loader = FileSystemLoader(settings.templates_folder)
    env = Environment(loader=file_loader)
    tm = env.get_template(settings.template_translation)

    output = tm.render(src_tgt_zip=zip(sentences, translations))
    print(output)


def main():

    import_path = Path(settings.import_dir)
    translator = return_translator(import_path / f"translator_{settings.epochs}")

    sentences = return_sentences(settings.test_sentences)

    translations = test_transformer(translator, sentences)

    print_translations(sentences, translations)


if __name__ == '__main__':
    main()
