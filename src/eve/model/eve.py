from dataclasses import dataclass

import torch
from torch import nn

from eve.config import ModelConfig
from eve.model.attention import MultiHeadAttention
from eve.model.blocks import DecoderBlock, EncoderBlock, LayerNorm
from eve.model.embeddings import Embeddings

INIT_STD = 0.02  # GPT-2's initialization: small normal weights


@dataclass
class Questions:
    """Logits of the fixed questions (softmax gives the probabilities)."""

    intent: torch.Tensor  # (B, n_intents): CALL / ASK / UNSUPPORTED / NONE, in INTENTS order
    slots: torch.Tensor  # (B, n_slots, max_tools + 1): "what is call #i?" over tools, then STOP


@dataclass
class Output:
    """What a forward pass returns: question logits and/or next-token logits."""

    questions: Questions | None  # None without tool positions (pretraining)
    lm: torch.Tensor | None  # (B, n_dec, vocab_size) next-token logits; None without dec_ids


class QuestionBlock(nn.Module):
    """The Jev part: fixed questions answered in one pass.

    Input is ``1 + n_slots`` learned query vectors: q_intent, then q_1..q_n ("what is call #i?").
    They go through decoder blocks with the causal mask off, so each query sees the other
    queries and, through cross-attention, the whole encoder output. They are learned vectors,
    not text, because the questions never change: their meaning comes from the labels.

    - Intent head: a linear layer on q_intent, like the book's ch. 6 classifier head.
    - Slot head, a pointer: call #i's answer is a score against each tool in the prompt,
      ``(W_q·h_i) · (W_k·t_k) / √pointer_dim``, where t_k is the encoder output at the k-th
      ``<tool>`` marker, plus a learned STOP key. The answers are the prompt's own tools, so the
      head can't invent a tool name and works for tools it never saw in training.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1 + cfg.n_slots, cfg.d_model) * INIT_STD)
        self.layers = nn.ModuleList(
            DecoderBlock(cfg, causal=False) for _ in range(cfg.n_question_layers)
        )
        self.norm = LayerNorm(cfg.d_model)
        self.intent_head = nn.Linear(cfg.d_model, cfg.n_intents)
        self.W_q = nn.Linear(cfg.d_model, cfg.pointer_dim, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.pointer_dim, bias=False)
        self.stop_key = nn.Parameter(torch.randn(cfg.pointer_dim) * INIT_STD)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tool_pos: torch.Tensor,
        tool_mask: torch.Tensor,
    ) -> Questions:
        """``memory`` (B, m, d) and ``memory_mask`` (B, m) from the encoder; ``tool_pos`` and
        ``tool_mask`` (B, max_tools) from the collator, padded tools marked False."""
        b = memory.shape[0]
        h = self.queries.expand(b, -1, -1)  # the same questions for every example
        for layer in self.layers:
            h = layer(h, memory, memory_mask=memory_mask)
        h = self.norm(h)

        intent = self.intent_head(h[:, 0])

        tools = memory[torch.arange(b, device=memory.device)[:, None], tool_pos]  # (B, T, d)
        keys = torch.cat([self.W_k(tools), self.stop_key.expand(b, 1, -1)], dim=1)  # (B, T+1, p)
        q = self.W_q(h[:, 1:])  # (B, n_slots, p)
        slots = q @ keys.transpose(1, 2) / keys.shape[-1] ** 0.5  # (B, n_slots, T+1)
        allowed = torch.cat([tool_mask, tool_mask.new_ones(b, 1)], dim=1)  # STOP is always allowed
        slots = slots.masked_fill(~allowed[:, None, :], float("-inf"))  # padded tools: P = 0
        return Questions(intent=intent, slots=slots)


class Eve(nn.Module):
    """The whole model (book ch. 4.6, `GPTModel`, turned into an encoder-decoder):

    embeddings -> encoder blocks -> final norm = ``memory``, one vector per encoder token. Then:
    - the question block reads ``memory`` (the Jev part);
    - the decoder reads its own tokens plus ``memory`` and an LM head turns its output into
      next-token logits (the Needle part, and span corruption in pretraining).

    The LM head is tied to the token embeddings: the same (vocab, d) matrix maps token -> vector
    at the input and vector -> logits at the output. GPT-2 does this (book ch. 4.6); for us it
    saves 2M of ~12M parameters.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = Embeddings(cfg)
        self.encoder = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.n_encoder_layers))
        self.encoder_norm = LayerNorm(cfg.d_model)
        self.decoder = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.n_decoder_layers))
        self.decoder_norm = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.questions = QuestionBlock(cfg)
        self.apply(self._init_weights)
        self.lm_head.weight = self.embeddings.tok.weight  # weight tying

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2's initialization: small normal weights (std 0.02), zero biases. PyTorch's
        default N(0, 1) embeddings would give the tied LM head logits with std ≈ √d ≈ 16."""
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, std=INIT_STD)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def encode(self, enc_ids: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        """(B, n) token ids and mask -> (B, n, d) ``memory``."""
        x = self.embeddings.encoder(enc_ids)
        for block in self.encoder:
            x = block(x, enc_mask)
        memory: torch.Tensor = self.encoder_norm(x)
        return memory

    def ask_questions(
        self,
        memory: torch.Tensor,
        enc_mask: torch.Tensor,
        tool_pos: torch.Tensor,
        tool_mask: torch.Tensor,
    ) -> Questions:
        """Answer the fixed questions from encoder ``memory`` and the ``<tool>`` positions."""
        answers: Questions = self.questions(memory, enc_mask, tool_pos, tool_mask)
        return answers

    def decode(
        self,
        dec_ids: torch.Tensor,
        memory: torch.Tensor,
        enc_mask: torch.Tensor,
        dec_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B, n_dec) decoder ids -> (B, n_dec, vocab_size) logits; position i predicts token
        i + 1. Training feeds the whole target sequence at once (teacher forcing); generation
        calls this again with the sequence one token longer each step."""
        x = self.embeddings.decoder(dec_ids)
        for block in self.decoder:
            x = block(x, memory, mask=dec_mask, memory_mask=enc_mask)
        logits: torch.Tensor = self.lm_head(self.decoder_norm(x))
        return logits

    def forward(
        self,
        enc_ids: torch.Tensor,
        enc_mask: torch.Tensor,
        dec_ids: torch.Tensor | None = None,
        dec_mask: torch.Tensor | None = None,
        tool_pos: torch.Tensor | None = None,
        tool_mask: torch.Tensor | None = None,
    ) -> Output:
        """Takes the tensors the collators make; labels stay in the batch for the loss."""
        memory = self.encode(enc_ids, enc_mask)
        questions = lm = None
        if tool_pos is not None:
            assert tool_mask is not None, "tool_pos comes with tool_mask"
            questions = self.ask_questions(memory, enc_mask, tool_pos, tool_mask)
        if dec_ids is not None:
            lm = self.decode(dec_ids, memory, enc_mask, dec_mask)
        return Output(questions=questions, lm=lm)

    def set_sdpa(self, enabled: bool) -> None:
        """Switch every attention layer to PyTorch's fused kernel (fast) or the manual path."""
        for m in self.modules():
            if isinstance(m, MultiHeadAttention):
                m.use_sdpa = enabled

    def parameter_counts(self) -> dict[str, int]:
        """Parameters per part, like the book's count in ch. 4.6. The LM head has none of its
        own: it reuses the token embeddings."""

        def count(*modules: nn.Module) -> int:
            return sum(p.numel() for m in modules for p in m.parameters())

        return {
            "embeddings": count(self.embeddings),
            "encoder": count(self.encoder, self.encoder_norm),
            "decoder": count(self.decoder, self.decoder_norm),
            "questions": count(self.questions),
            "total": count(self),
        }


def ask(model: Eve, batch: dict[str, torch.Tensor]) -> Questions:
    """Encoder + question block only: the decoder isn't needed for the questions."""
    memory = model.encode(batch["enc_ids"], batch["enc_mask"])
    return model.ask_questions(memory, batch["enc_mask"], batch["tool_pos"], batch["tool_mask"])
