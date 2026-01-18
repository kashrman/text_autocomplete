"""
src/eval_transformer_pipeline.py
оценка трансформера "distilgpt2"
"""
import torch
from rouge_score import rouge_scorer
from transformers import pipeline


def _decode(tokenizer, ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


@torch.no_grad()
def evaluate_distilgpt2_pipeline_rouge_3of4(
    dataloader,
    gen, # pipeline object
    max_new_tokens_cap: int = 64,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 0.9,
    limit_batches: int | None = 50,
):
    tok = gen.tokenizer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)

    total_r1, total_r2, n = 0.0, 0.0, 0

    for bi, batch in enumerate(dataloader):
        if limit_batches is not None and bi >= limit_batches:
            break

        x = batch["input_ids"]      # [B,T]
        y = batch["labels"]         # [B,T]
        lengths = batch["lengths"]  # [B]

        B = x.size(0)
        for i in range(B):
            L = int(lengths[i].item())
            if L < 4:
                continue

            last_target = y[i, L - 1].item()
            if last_target == -100:
                continue

            full_ids = torch.cat([x[i, :L], torch.tensor([last_target])])  # [L+1]
            cut = max(1, int(0.75 * full_ids.numel()))
            prefix_ids = full_ids[:cut]
            target_ids = full_ids[cut:]

            prefix_text = _decode(tok, prefix_ids)
            ref_text = _decode(tok, target_ids)
            if not prefix_text or not ref_text:
                continue

            gen_len = min(int(target_ids.numel()), max_new_tokens_cap)

            out = gen(
                prefix_text,
                max_new_tokens=gen_len,          # рекомендованный способ ограничивать добавляемые токены
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=1,
                return_full_text=False,          # вернуть только “добавку”, без префикса
            )
            hyp_text = out[0]["generated_text"].strip()

            if not hyp_text:
                continue

            scores = scorer.score(ref_text, hyp_text)
            total_r1 += scores["rouge1"].fmeasure
            total_r2 += scores["rouge2"].fmeasure
            n += 1

    if n == 0:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "n_samples": 0}

    return {"rouge1_f": total_r1 / n, "rouge2_f": total_r2 / n, "n_samples": n}

