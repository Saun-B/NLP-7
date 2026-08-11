"""Torch datasets: word-level vocabulary (E1) and PhoBERT alignment (E2–E4).

The two model families need different views of the same JSONL rows:

**BiLSTM (E1)** consumes lexical words directly. A ``Vocabulary`` built
**from train only** maps word → id, with ``<PAD>=0`` and ``<UNK>=1`` reserved.
Padding positions get ``labels = IGNORE_INDEX`` so the loss ignores them.

**PhoBERT (E2–E4)** consumes BPE subwords. One lexical word can become several
subwords, but the punctuation belongs *after the whole word*, so the gold label
is placed on the **last subword** of the word and every other position
(non-final subwords, ``<s>``, ``</s>``, padding) is ``IGNORE_INDEX``. That
gives the invariant this project depends on:

    every lexical word is supervised exactly once, and no word is dropped.

Sequences longer than ``max_length`` subwords are **split into several
windows at word boundaries** — never truncated. A word therefore lands in
exactly one window, so pooling predictions over all windows reconstructs the
full word-level prediction sequence.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

from src.data.constants import (
    IGNORE_INDEX,
    LABEL2ID,
    PAD_ID,
    PAD_TOKEN,
    PHOBERT_MAX_LENGTH,
    UNK_ID,
    UNK_TOKEN,
)
from src.data.schema import Example
from src.utils.io import iter_jsonl, read_json, write_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

__all__ = [
    "load_examples",
    "Vocabulary",
    "BiLSTMDataset",
    "collate_bilstm",
    "SubwordWindow",
    "align_words_to_subwords",
    "PhoBERTEncoder",
    "PhoBERTDataset",
    "collate_phobert",
]





def load_examples(
    path: PathLike, *, limit: Optional[int] = None, intern_tokens: bool = True
) -> List[Example]:
    """Read a processed JSONL split into :class:`~src.data.schema.Example` objects.

    ``limit`` truncates the split — used only by smoke tests, never for a real
    training run (the notebooks pass ``limit=None``).

    ``intern_tokens`` deduplicates token strings. ``json.loads`` allocates a
    fresh ``str`` for every occurrence, so a 4.9M-token split would hold 4.9M
    string objects; the corpus only has ~31k distinct words, and interning
    collapses those hundreds of megabytes to a few.
    """
    import sys as _sys

    intern = _sys.intern if intern_tokens else (lambda s: s)
    out: List[Example] = []
    for i, row in enumerate(iter_jsonl(path)):
        if limit is not None and i >= limit:
            break
        ex = Example.from_dict(row)
        if intern_tokens:
            ex.tokens = [intern(t) for t in ex.tokens]
            ex.labels = [intern(l) for l in ex.labels]
        out.append(ex)
    if not out:
        raise ValueError(f"No examples loaded from {path}")
    logger.info("Loaded %d examples from %s", len(out), Path(path).name)
    return out





class Vocabulary:
    """Word → id mapping for the BiLSTM.

    Built **from the training split only**: letting validation words into the
    vocabulary would be a (small but real) form of leakage, and would also make
    validation look easier than deployment.
    """

    def __init__(self, itos: Sequence[str], min_freq: int = 1, source_split: str = "train"):
        self.itos: List[str] = list(itos)
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}
        self.min_freq = min_freq
        self.source_split = source_split
        if self.itos[PAD_ID] != PAD_TOKEN or self.itos[UNK_ID] != UNK_TOKEN:
            raise ValueError(
                f"Reserved slots wrong: expected {PAD_TOKEN!r} at {PAD_ID} and "
                f"{UNK_TOKEN!r} at {UNK_ID}, got {self.itos[:2]!r}"
            )


    @classmethod
    def build(
        cls,
        examples: Iterable[Example],
        *,
        min_freq: int = 1,
        max_size: Optional[int] = None,
        source_split: str = "train",
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for ex in examples:
            counter.update(ex.tokens)


        candidates = [tok for tok, c in counter.items() if c >= min_freq]
        candidates.sort(key=lambda t: (-counter[t], t))
        if max_size is not None:
            candidates = candidates[: max(0, max_size - 2)]

        itos = [PAD_TOKEN, UNK_TOKEN] + candidates
        vocab = cls(itos, min_freq=min_freq, source_split=source_split)
        logger.info(
            "Built vocabulary from %s: %d types (min_freq=%d, %d distinct words seen)",
            source_split,
            len(vocab),
            min_freq,
            len(counter),
        )
        return vocab


    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def encode(self, tokens: Sequence[str]) -> List[int]:
        stoi = self.stoi
        return [stoi.get(tok, UNK_ID) for tok in tokens]

    def unk_rate(self, examples: Iterable[Example]) -> float:
        total = 0
        unk = 0
        for ex in examples:
            for tok in ex.tokens:
                total += 1
                if tok not in self.stoi:
                    unk += 1
        return unk / total if total else 0.0


    def save(self, path: PathLike) -> Path:
        return write_json(
            path,
            {
                "min_freq": self.min_freq,
                "source_split": self.source_split,
                "size": len(self.itos),
                "pad_token": PAD_TOKEN,
                "unk_token": UNK_TOKEN,
                "itos": self.itos,
            },
        )

    @classmethod
    def load(cls, path: PathLike) -> "Vocabulary":
        blob = read_json(path)
        return cls(
            blob["itos"],
            min_freq=blob.get("min_freq", 1),
            source_split=blob.get("source_split", "train"),
        )


class BiLSTMDataset(Dataset):
    """Word-id sequences + label-id sequences for the BiLSTM."""

    def __init__(self, examples: Sequence[Example], vocab: Vocabulary):
        self.examples = list(examples)
        self.vocab = vocab
        self.label2id = dict(LABEL2ID)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(self.vocab.encode(ex.tokens), dtype=torch.long),
            "labels": torch.tensor([self.label2id[l] for l in ex.labels], dtype=torch.long),
            "length": len(ex.tokens),
            "example_id": ex.id,
            "index": idx,
        }


def collate_bilstm(
    batch: Sequence[Dict[str, Any]],
    *,
    pad_id: int = PAD_ID,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, Any]:
    """Right-pad a batch; padded label positions are ``ignore_index``."""
    max_len = max(int(item["length"]) for item in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), ignore_index, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    lengths = torch.zeros(bsz, dtype=torch.long)

    for i, item in enumerate(batch):
        n = int(item["length"])
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        attention_mask[i, :n] = 1
        lengths[i] = n

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "lengths": lengths,
        "example_ids": [item["example_id"] for item in batch],
    }





@dataclass
class SubwordWindow:
    """One PhoBERT input window covering a contiguous range of lexical words."""

    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]

    word_index: List[int]
    first_word: int
    num_words: int
    example_id: str = ""
    window_index: int = 0

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def num_supervised(self) -> int:
        return sum(1 for l in self.labels if l != IGNORE_INDEX)


def align_words_to_subwords(
    word_subword_ids: Sequence[Sequence[int]],
    label_ids: Sequence[int],
    *,
    bos_id: int,
    eos_id: int,
    max_length: int = PHOBERT_MAX_LENGTH,
    ignore_index: int = IGNORE_INDEX,
    example_id: str = "",
) -> List[SubwordWindow]:
    """Lay lexical words out into ``<s> … </s>`` windows of ``<= max_length``.

    * The label of word *j* is written at the position of its **last** subword.
    * Every other position — non-final subwords, ``<s>``, ``</s>`` — is
      ``ignore_index``, so ``CrossEntropyLoss`` supervises each word once.
    * When the words do not fit in one window, the sequence is **split at a
      word boundary** into consecutive windows. No word is ever dropped and no
      word is split across two windows.
    * Degenerate case: a single word whose subword count exceeds the window
      capacity is truncated to the capacity (its last kept subword carries the
      label). This cannot happen with PhoBERT on this corpus — a Vietnamese
      syllable is 1–3 BPE pieces — but the branch exists so the invariant
      "no word is dropped" holds unconditionally.
    """
    if len(word_subword_ids) != len(label_ids):
        raise ValueError(
            f"len(word_subword_ids)={len(word_subword_ids)} != len(label_ids)={len(label_ids)}"
        )
    if max_length < 3:
        raise ValueError(f"max_length must be >= 3, got {max_length}")

    capacity = max_length - 2
    windows: List[SubwordWindow] = []

    cur_ids: List[int] = []
    cur_labels: List[int] = []
    cur_word_index: List[int] = []
    cur_first_word = 0
    cur_num_words = 0

    def flush() -> None:
        nonlocal cur_ids, cur_labels, cur_word_index, cur_first_word, cur_num_words
        if not cur_ids:
            return
        windows.append(
            SubwordWindow(
                input_ids=[bos_id] + cur_ids + [eos_id],
                attention_mask=[1] * (len(cur_ids) + 2),
                labels=[ignore_index] + cur_labels + [ignore_index],
                word_index=[-1] + cur_word_index + [-1],
                first_word=cur_first_word,
                num_words=cur_num_words,
                example_id=example_id,
                window_index=len(windows),
            )
        )
        cur_ids, cur_labels, cur_word_index = [], [], []
        cur_num_words = 0

    for word_idx, (pieces, label_id) in enumerate(zip(word_subword_ids, label_ids)):
        pieces = list(pieces)
        if not pieces:
            raise ValueError(
                f"word {word_idx} of example {example_id!r} produced zero subwords; "
                "the encoder must substitute <unk> instead."
            )
        if len(pieces) > capacity:
            logger.warning(
                "Example %s word %d expands to %d subwords (> capacity %d); "
                "truncating that single word.",
                example_id,
                word_idx,
                len(pieces),
                capacity,
            )
            pieces = pieces[:capacity]

        if cur_ids and len(cur_ids) + len(pieces) > capacity:
            flush()
            cur_first_word = word_idx

        if not cur_ids:
            cur_first_word = word_idx

        n = len(pieces)
        cur_ids.extend(pieces)
        cur_labels.extend([ignore_index] * (n - 1) + [label_id])
        cur_word_index.extend([-1] * (n - 1) + [word_idx])
        cur_num_words += 1

    flush()


    supervised = [w for win in windows for w in win.word_index if w >= 0]
    if supervised != list(range(len(word_subword_ids))):
        raise AssertionError(
            f"alignment lost or reordered words for example {example_id!r}: "
            f"{len(supervised)} supervised vs {len(word_subword_ids)} words"
        )
    return windows


class PhoBERTEncoder:
    """Tokenizer wrapper that memoises word → subword-id lookups.

    The corpus has ~31k distinct words spread over millions of tokens, so
    caching turns tokenisation from the dominant cost into a dict lookup.
    """

    def __init__(self, tokenizer, *, max_length: int = PHOBERT_MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
        self.unk_id = tokenizer.unk_token_id
        if None in (self.bos_id, self.eos_id, self.pad_id, self.unk_id):
            raise ValueError("Tokenizer is missing one of bos/eos/pad/unk ids.")
        self._cache: Dict[str, Tuple[int, ...]] = {}

    def word_to_ids(self, word: str) -> Tuple[int, ...]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        pieces = self.tokenizer.tokenize(word)
        if not pieces:
            ids: Tuple[int, ...] = (self.unk_id,)
        else:
            ids = tuple(self.tokenizer.convert_tokens_to_ids(pieces))
        self._cache[word] = ids
        return ids

    def encode_example(self, example: Example) -> List[SubwordWindow]:
        word_ids = [list(self.word_to_ids(tok)) for tok in example.tokens]
        label_ids = [LABEL2ID[l] for l in example.labels]
        return align_words_to_subwords(
            word_ids,
            label_ids,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            max_length=self.max_length,
            example_id=example.id,
        )

    def encode_words(
        self, words: Sequence[str], *, example_id: str = "inference"
    ) -> List[SubwordWindow]:
        """Encode unlabelled words for inference.

        Same windowing as training, so text of any length is split at word
        boundaries instead of being truncated. ``labels`` come back as
        ``IGNORE_INDEX`` placeholders — at inference we only need
        ``word_index`` to map each supervised position back to its word.
        """
        if not words:
            return []
        word_ids = [list(self.word_to_ids(w)) for w in words]
        placeholder = [LABEL2ID["O"]] * len(words)
        windows = align_words_to_subwords(
            word_ids,
            placeholder,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            max_length=self.max_length,
            example_id=example_id,
        )
        for w in windows:
            w.labels = [IGNORE_INDEX] * len(w.labels)
        return windows

    @property
    def cache_size(self) -> int:
        return len(self._cache)


@dataclass
class PhoBERTDatasetStats:
    num_examples: int = 0
    num_windows: int = 0
    num_words: int = 0
    num_subwords: int = 0
    examples_split_into_multiple_windows: int = 0
    max_window_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_examples": self.num_examples,
            "num_windows": self.num_windows,
            "num_words": self.num_words,
            "num_subwords": self.num_subwords,
            "windows_per_example": round(
                self.num_windows / self.num_examples if self.num_examples else 0.0, 4
            ),
            "subwords_per_word": round(
                self.num_subwords / self.num_words if self.num_words else 0.0, 4
            ),
            "examples_split_into_multiple_windows": (
                self.examples_split_into_multiple_windows
            ),
            "max_window_length": self.max_window_length,
        }


class PhoBERTDataset(Dataset):
    """Pre-encoded PhoBERT windows.

    One dataset item = one window. Encoding happens once in ``__init__`` (fast
    thanks to :class:`PhoBERTEncoder`'s cache) so the training loop never pays
    tokenisation cost.

    Windows are stored as compact NumPy arrays rather than Python lists: the
    train split is ~5.4M subword positions, which as boxed Python ints would
    cost several hundred megabytes. ``int32``/``int16`` arrays bring that down
    to a few tens of MB.
    """

    def __init__(
        self,
        examples: Sequence[Example],
        encoder: PhoBERTEncoder,
        *,
        log_every: int = 20000,
    ):
        import numpy as np

        self.encoder = encoder
        self.stats = PhoBERTDatasetStats(num_examples=len(examples))

        self._input_ids: List[Any] = []
        self._labels: List[Any] = []
        self._word_index: List[Any] = []
        self._example_ids: List[str] = []
        self._window_indices: List[int] = []

        for i, ex in enumerate(examples):
            windows = encoder.encode_example(ex)
            if len(windows) > 1:
                self.stats.examples_split_into_multiple_windows += 1
            for w in windows:
                self._input_ids.append(np.asarray(w.input_ids, dtype=np.int32))
                self._labels.append(np.asarray(w.labels, dtype=np.int16))
                self._word_index.append(np.asarray(w.word_index, dtype=np.int16))
                self._example_ids.append(w.example_id)
                self._window_indices.append(w.window_index)
                self.stats.num_subwords += len(w.input_ids)
                self.stats.max_window_length = max(
                    self.stats.max_window_length, len(w.input_ids)
                )
            self.stats.num_words += len(ex.tokens)
            if log_every and (i + 1) % log_every == 0:
                logger.info("Encoded %d/%d examples", i + 1, len(examples))

        self.stats.num_windows = len(self._input_ids)
        logger.info("PhoBERT dataset ready: %s", self.stats.to_dict())

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ids = self._input_ids[idx]
        return {
            "input_ids": torch.from_numpy(ids.astype("int64")),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels": torch.from_numpy(self._labels[idx].astype("int64")),
            "word_index": torch.from_numpy(self._word_index[idx].astype("int64")),
            "length": len(ids),
            "example_id": self._example_ids[idx],
            "window_index": self._window_indices[idx],
        }


def collate_phobert(
    batch: Sequence[Dict[str, Any]],
    *,
    pad_id: int = 1,
    ignore_index: int = IGNORE_INDEX,
    pad_to_multiple_of: Optional[int] = 8,
) -> Dict[str, Any]:
    """Right-pad a batch of windows.

    ``pad_to_multiple_of=8`` keeps sequence lengths tensor-core friendly under
    fp16 on the GPU; it costs a handful of ignored positions.
    """
    max_len = max(int(item["length"]) for item in batch)
    if pad_to_multiple_of:
        max_len = int(math.ceil(max_len / pad_to_multiple_of) * pad_to_multiple_of)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), ignore_index, dtype=torch.long)
    word_index = torch.full((bsz, max_len), -1, dtype=torch.long)

    for i, item in enumerate(batch):
        n = int(item["length"])
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]
        labels[i, :n] = item["labels"]
        word_index[i, :n] = item["word_index"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "word_index": word_index,
        "example_ids": [item["example_id"] for item in batch],
        "window_indices": [item["window_index"] for item in batch],
    }
