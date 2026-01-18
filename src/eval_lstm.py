"""
eval_lstm.py
Оценка LSTM модели: код замера и вывода метрики ROUGE. (ROUGE + генерация 3/4→1/4).
"""
import torch
from rouge_score import rouge_scorer


def _ids_to_text(tokenizer, ids):
    # ids: 1D tensor/list
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


@torch.no_grad()
def evaluate_rouge_3of4(
    model,
    dataloader,
    tokenizer,
    device,
    max_new_tokens: int = 32,
    do_sample: bool = False,
    top_k: int | None = None,
    temperature: float = 1.0,
    limit_batches: int | None = None,
):
    """
    Сценарий: берем полный текст (input+labels), режем на prefix=3/4 и target=1/4.
    Генерируем продолжение для prefix и считаем ROUGE между generated_suffix и target.
    """
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)  # [web:6]

    total_r1, total_r2, n = 0.0, 0.0, 0

    for bi, batch in enumerate(dataloader):
        if limit_batches is not None and bi >= limit_batches:
            break

        x = batch["input_ids"].to(device)         # [B,T]
        y = batch["labels"].to(device)            # [B,T] (с -100 на паддингах)
        lengths = batch["lengths"].to(device)     # [B]

        B = x.size(0)
        for i in range(B):
            L = int(lengths[i].item())
            if L < 4:
                continue

            # восстановим "полную" последовательность ids длины L+1:
            # full = [x0..x_{L-1}] + [y_{L-1}]
            last_target = y[i, L - 1].item()
            if last_target == -100:
                continue

            full_ids = torch.cat(
                [x[i, :L], torch.tensor([last_target], device=device, dtype=torch.long)]
            )  # [L+1]

            cut = max(1, int(0.75 * full_ids.numel()))
            prefix_ids = full_ids[:cut]     # 3/4
            target_ids = full_ids[cut:]     # 1/4

            # генерим не больше, чем длина таргета (чтобы сравнение было честнее)
            gen_len = min(max_new_tokens, int(target_ids.numel()))
            gen_ids = model.generate(
                prefix_ids=prefix_ids,
                lengths=torch.tensor([prefix_ids.numel()], device=device),
                max_new_tokens=gen_len,
                eos_id=tokenizer.eos_token_id,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
            )

            gen_suffix = gen_ids[prefix_ids.numel():]  # только сгенерированное продолжение

            ref_text = _ids_to_text(tokenizer, target_ids)
            hyp_text = _ids_to_text(tokenizer, gen_suffix)

            if len(ref_text) == 0 or len(hyp_text) == 0:
                continue

            scores = scorer.score(ref_text, hyp_text)  # [web:6]
            total_r1 += scores["rouge1"].fmeasure
            total_r2 += scores["rouge2"].fmeasure
            n += 1

    if n == 0:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "n_samples": 0}

    return {"rouge1_f": total_r1 / n, "rouge2_f": total_r2 / n, "n_samples": n}
