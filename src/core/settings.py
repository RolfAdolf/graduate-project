from pydantic import BaseSettings
from pydantic.error_wrappers import ValidationError

from typing import List


class Settings(BaseSettings):
    data_train_size: int
    data_aug_size: int
    data_val_size: int
    data_test_size: int
    vocab_size: int
    reserved_tokens: List[str]
    src_file: str
    tgt_file: str
    max_tokens: int
    batch_size: int
    buffer_size: int = 0
    num_layers: int
    d_model: int
    dff: int
    num_heads: int
    dropout_rate: float
    checkpoint_path: str
    epochs: int
    export_dir: str
    import_dir: str
    source_sentence: str
    ground_truth: str
    test_sentences: str
    templates_folder: str
    template_translation: str


try:
    settings = Settings(_env_file="./.env", _env_file_encoding="utf-8")
except ValidationError:
    settings = Settings(_env_file="../.env", _env_file_encoding="utf-8")

settings.buffer_size = 2 * settings.data_train_size + settings.data_aug_size


if __name__ == '__main__':
    print(settings)
