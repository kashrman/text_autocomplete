"""
lstm_model.py
LSTM-модель для языкового моделирования (next token).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMNextToken(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        """
        input_ids: [B, T] (X)
        lengths:   [B]
        return logits: [B, T, V] — предсказание следующего токена для каждого t
        """
        emb = self.embedding(input_ids)  # [B,T,E]

        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B,T,H]

        logits = self.fc(out)  # [B,T,V]
        return logits

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: torch.Tensor,      # [T] или [1,T]
        lengths: torch.Tensor | None = None,
        max_new_tokens: int = 20,
        eos_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = 50,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Возвращает ids = prefix + продолжение.
        Простой авторегрессионный цикл: каждый раз берём logits последней позиции.
        """
        self.eval()
        device = next(self.parameters()).device

        if prefix_ids.dim() == 1:
            ids = prefix_ids.unsqueeze(0).to(device)  # [1,T]
        else:
            ids = prefix_ids.to(device)

        if lengths is None:
            lengths = torch.tensor([ids.size(1)], device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(ids, lengths=lengths)         # [1,T,V]
            next_logits = logits[:, -1, :]                      # [1,V]

            if temperature != 1.0:
                next_logits = next_logits / max(temperature, 1e-8)

            if top_k is not None:
                v, ix = torch.topk(next_logits, k=top_k, dim=-1)
                filt = torch.full_like(next_logits, float("-inf"))
                filt.scatter_(1, ix, v)
                next_logits = filt

            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # [1,1]
            else:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)  # [1,1]

            ids = torch.cat([ids, next_id], dim=1)  # [1, T+1]
            lengths = lengths + 1

            if eos_id is not None and next_id.item() == eos_id:
                break

        return ids.squeeze(0)  # [T_total]
