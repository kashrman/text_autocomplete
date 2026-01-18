"""
data_utils.py
Утилиты для загрузки/очистки/разбиения датасета.

структура:
- data/raw_dataset.csv
- data/dataset_processed.csv
- data/train.csv, data/val.csv, data/test.csv
"""
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


# пути
RAW_PATH = "data/raw_dataset.csv"
PROCESSED_PATH = "data/dataset_processed.csv"
TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
TEST_PATH = "data/test.csv"


def clear_datatset():
    # Очистка набора. Согласно заднию нужно так сделать,
    # но скорее всего используемый мной токенизатор позволяет корректно учиться и без этого
    with open(RAW_PATH, "r", encoding="utf-8", errors="ignore") as f:
        texts = [line.rstrip("\n") for line in f]

    df = pd.DataFrame({"text": texts})

    def _clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()  # к нижнему регистру
        text = re.sub(r"[^a-z0-9 ]+", " ", text)  # оставить только буквы и цифры
        text = re.sub(r"\s+", " ", text).strip()  # убрать дублирующиеся пробелы
        return text

    df_clean = df["text"].map(_clean_text)
    df_clean = df_clean[df_clean.str.len() > 0].reset_index(drop=True) # выкинем пустые после очистки

    df_clean.to_csv(PROCESSED_PATH, index=False, header=False)

    return df_clean


def split_dataset(df_clean, test_size=0.8):
    # разбиение набора на части
    train_df, tmp_df = train_test_split(
        df_clean,
        test_size=1-test_size,
        random_state=42,
        shuffle=True
    )

    val_df, test_df = train_test_split(
        tmp_df,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    train_df.to_csv(TRAIN_PATH, index=False, header=False)
    val_df.to_csv(VAL_PATH, index=False, header=False)
    test_df.to_csv(TEST_PATH, index=False, header=False)

    return train_df, val_df, test_df


def prepare_all_data(force: bool = False, debug_test: bool = False):
    # Основная функция для подготовки наборов данных
    if (
        not force
        and not debug_test
        and os.path.isfile(TRAIN_PATH)
        and os.path.isfile(VAL_PATH)
        and os.path.isfile(TEST_PATH)
    ):
        print("All data files already exist, simple read")

        train_df = pd.read_csv(TRAIN_PATH, header=None)
        val_df = pd.read_csv(VAL_PATH, header=None)
        test_df = pd.read_csv(TEST_PATH, header=None)
    
    else:
        print("Create cleared data file")
        df = clear_datatset()

        # Если отладка, то для теста оставить в наборе только первые 10000 строк
        if debug_test:
            df = df.head(10000)
            print(f"First 5 texts:\n{df.head()}\ntotal length = {len(df)}\n")

        print("Split data files")
        train_df, val_df, test_df = split_dataset(df)
    
    print(f"Lenght: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_all_data(force=True)

