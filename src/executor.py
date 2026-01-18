# Запуск всех операций (executor.py)

import torch
from transformers import pipeline, AutoTokenizer

from data_utils import prepare_all_data
from next_token_dataset import GenDataLoaders
from lstm_model import LSTMNextToken
from lstm_train import train_lstm
from eval_lstm import evaluate_rouge_3of4
from eval_transformer_pipeline import evaluate_distilgpt2_pipeline_rouge_3of4

torch.manual_seed(42)

# 1) Tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

# GPT2 обычно без PAD
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_size = len(tokenizer)
pad_id = tokenizer.pad_token_id or 0
eos_id = tokenizer.eos_token_id

# 2) Devices
debug_test = True
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline_device = 0 if torch.cuda.is_available() else -1  # pipeline ждёт int/-1

# 3) Data
train_df, val_df, test_df = prepare_all_data(force=True, debug_test=debug_test)
train_loader, val_loader, test_loader = GenDataLoaders(train_df, val_df, test_df, tokenizer)

# 4) LSTM train
model = LSTMNextToken(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_size=128,
    num_layers=1,
    dropout=0.1,
    pad_id=pad_id,
)

model = train_lstm(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    tokenizer=tokenizer,
    device=device,
    n_epochs=5,
    lr=1e-3,
    grad_clip=1.0,
    eval_every_epochs=1,
    max_new_tokens_eval=20,
)

# если потом нужно загрузить модель для инференса
# ckpt = torch.load("models/lstm_best.pt", map_location=device)
# model.load_state_dict(ckpt["model_state_dict"])
# model.eval()

rouge_lstm_val = evaluate_rouge_3of4(
    model, val_loader, tokenizer, device,
    max_new_tokens=32,
    do_sample=False
)
print("LSTM VAL:", rouge_lstm_val)

# 5) distilgpt2 pipeline (ONE instance, reuse everywhere)
gpt2_gen = pipeline(
    "text-generation",
    model="distilgpt2",
    tokenizer=tokenizer,   # тот же объект токенизатора для надежности
    device=pipeline_device,
)

# 6) distilgpt2 eval
rouge_gpt2_val = evaluate_distilgpt2_pipeline_rouge_3of4(
    dataloader=val_loader,
    gen=gpt2_gen,
    max_new_tokens_cap=64,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    limit_batches=None,
)
print("GPT2 VAL:", rouge_gpt2_val)


# 7) Side-by-side examples
list_examples = [
    "i am going",
    "tomorrow i will",
    "this movie is",
    "Company Google is",
    "If you compare Google and Yandex, you could say that",
    "Our mentor is very smart and",
    "The distilgpt2 model is very strange, but it allows",
    "I would like",
]

for prefix in list_examples:
    # LSTM: генерим ids и декодим
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device)

    lstm_ids = model.generate(
        prefix_ids,
        max_new_tokens=20,
        do_sample=True,
        top_k=50,
        temperature=1.0,
        eos_id=eos_id,
    )
    lstm_full = tokenizer.decode(lstm_ids.tolist(), skip_special_tokens=True)
    lstm_suffix = lstm_full[len(prefix):].strip()

    # GPT2 pipeline: сразу просим только continuation
    out = gpt2_gen(
        prefix,
        max_new_tokens=20,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=1,
        return_full_text=False,
    )
    gpt2_suffix = out[0]["generated_text"].strip()

    print("PREFIX:", prefix)
    print("LSTM +:", lstm_suffix)
    print("GPT2 +:", gpt2_suffix)
    print("-" * 60)

