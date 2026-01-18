
import torch

from data_utils import prepare_all_data
from next_token_dataset import GenDataLoaders, tokenizer
from lstm_model import LSTMNextToken
from lstm_train import train_lstm
from eval_lstm import evaluate_rouge_3of4
from eval_transformer_pipeline import evaluate_distilgpt2_pipeline_rouge_3of4


device = "cuda" if torch.cuda.is_available() else "cpu"


def main_executor(debug_test):

    train_df, val_df, test_df = prepare_all_data(force=False, debug_test=debug_test)

    train_loader, val_loader, test_loader = GenDataLoaders(train_df, val_df, test_df)

    vocab_size = len(tokenizer) # для bert: tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id or 0

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
        n_epochs=10,
        lr=1e-3,
        grad_clip=1.0,
        eval_every_epochs=1,
        max_new_tokens_eval=20,
    )

    # Финальная оценка для lstm на val_loader (можно добавить и на test_loader)
    rouge_lstm_val = evaluate_rouge_3of4(model, val_loader, tokenizer, device, max_new_tokens=32, do_sample=False)
    print("LSTM VAL:", rouge_lstm_val)

    # Финальная оценка для gpt2 на val_loader (можно добавить и на test_loader)
    rouge_gpt2_val = evaluate_distilgpt2_pipeline_rouge_3of4(
        dataloader=val_loader,
        device=0 if device=="cuda" else -1,
        model_name="distilgpt2",
        max_new_tokens_cap=64,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        limit_batches=None,
    )
    print("GPT2 VAL:", rouge_gpt2_val)

    # Несколько примеров генерации
    list_examples = [
        "i am going",
        "tomorrow i will",
        "this movie is",
        'Company Google is',
        'If you compare Google and Yandex, you could say that',
        'Our mentor is very smart and',
        'The distilgpt2 model is very strange, but it allows',
        'I would like',
    ]

    for example in list_examples:
        example_ids = tokenizer(example, add_special_tokens=False)["input_ids"]
        example_ids = torch.tensor(example_ids, dtype=torch.long, device=device)

        gen_ids = model.generate(
           example_ids,
           max_new_tokens=20,
           do_sample=True,
           top_k=50,
           temperature=1.0,
           return_full_text=False  # вернуть только продолжение
        )
        print("PREFIX:", example)
        print("GEN:", tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True))
        print("-"*60)


if __name__ == "__main__":
    main_executor(debug_test=True)

