"""
next_token_dataset.py
Torch Dataset для задачи next-token prediction (автодополнение текста).
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class NextTokenDataset(Dataset):
    def __init__(self, ids_list, eos_id):
        self.ids_list = ids_list
        self.eos_id = eos_id

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        ids = self.ids_list[idx]
        ids = torch.tensor(ids, dtype=torch.long)

        if ids.numel() < 2:
            ids = torch.tensor([self.eos_id, self.eos_id], dtype=torch.long)

        return {
            "input_ids": ids[:-1],  # X
            "labels": ids[1:]       # Y (сдвиг вправо)
        }


def collate_fn(batch, pad_id: int):
    xs = [item["input_ids"] for item in batch]
    ys = [item["labels"] for item in batch]
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)

    x_pad = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=-100)  # для ignore_index

    attention_mask = (x_pad != pad_id).long()

    return {"input_ids": x_pad, "attention_mask": attention_mask, "labels": y_pad, "lengths": lengths}


def GenDataLoaders(train_df, val_df, test_df, tokenizer):
    # Создает Torch Dataset-ы для задачи next-token prediction (автодополнение текста)
    # tokenizer берется как входной, чтобы он был инициализирован одинаково для будущего сравнения с трансформером
    
    # претокенизация
    train_ids = tokenizer(list(train_df), truncation=True, max_length=128, padding=False)["input_ids"]
    val_ids   = tokenizer(list(val_df),   truncation=True, max_length=128, padding=False)["input_ids"]
    test_ids  = tokenizer(list(test_df),  truncation=True, max_length=128, padding=False)["input_ids"]

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    train_ds = NextTokenDataset(train_ids, eos_id=eos_id)
    val_ds   = NextTokenDataset(val_ids,   eos_id=eos_id)
    test_ds  = NextTokenDataset(test_ids,  eos_id=eos_id)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, pad_id=pad_id))
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, pad_id=pad_id))
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, pad_id=pad_id))
    
    return train_loader, val_loader, test_loader 

