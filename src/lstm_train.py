"""
lstm_train.py
Обучение LSTM модели.
"""

import os
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from eval_lstm import evaluate_rouge_3of4


def train_lstm(
    model,
    train_loader,
    val_loader,
    tokenizer,
    device,
    n_epochs: int = 5,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    eval_every_epochs: int = 1,
    max_new_tokens_eval: int = 32,
    save_dir: str = "models",
    save_name: str = "lstm_best.pt",
):
    model.to(device)

    # labels с паддингом -100, значит ignore_index=-100 [web:31]
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # [web:31]
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [train]"):
            inputs = batch["input_ids"].to(device)     # [B,T]
            lengths = batch["lengths"].to(device)      # [B]
            labels = batch["labels"].to(device)        # [B,T]

            optimizer.zero_grad()

            logits = model(inputs, lengths)            # [B,T,V]
            # CrossEntropyLoss ждёт [N,C,*], поэтому делаем [B,V,T] [web:31]
            loss = loss_fn(logits.transpose(1, 2), labels)  # [web:31]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        # Валидация по loss (быстро)
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [val]"):
                inputs = batch["input_ids"].to(device)
                lengths = batch["lengths"].to(device)
                labels = batch["labels"].to(device)

                logits = model(inputs, lengths)
                loss = loss_fn(logits.transpose(1, 2), labels)  # [web:31]
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))

        # ROUGE (медленнее, поэтому можно делать раз в несколько эпох)
        rouge_str = ""
        if (epoch + 1) % eval_every_epochs == 0:
            rouge = evaluate_rouge_3of4(
                model=model,
                dataloader=val_loader,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens_eval,
                do_sample=False,
                top_k=None,
                temperature=1.0,
                # limit_batches=50,  # для теста можно включить, чтобы не ждать слишком долго
            )
            rouge_str = f", ROUGE1-F={rouge['rouge1_f']:.4f}, ROUGE2-F={rouge['rouge2_f']:.4f} (n={rouge['n_samples']})"

        print(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}{rouge_str}"
        )

        # -------- save best checkpoint --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            best_state = copy.deepcopy(model.state_dict())

            ckpt_path = os.path.join(save_dir, save_name)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "tokenizer_name": getattr(tokenizer, "name_or_path", None),
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to: {ckpt_path} (val_loss={best_val_loss:.4f})")

    # в конце загрузим лучший стейт обратно
    if best_state is not None:
        model.load_state_dict(best_state)

    return model

