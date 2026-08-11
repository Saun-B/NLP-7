from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.constants import ID2LABEL, IGNORE_INDEX, LABELS, NUM_LABELS
from src.evaluation.metrics import metrics_from_confusion_matrix
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["evaluate", "predict_word_labels", "sample_predictions"]

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    loss_fn: Optional[torch.nn.Module] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    progress: bool = True,
    desc: str = "validation",
) -> Dict[str, Any]:
    model.eval()
    cm = np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.int64)
    total_loss = 0.0
    total_supervised = 0
    num_batches = 0

    iterator = dataloader
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(dataloader, desc=desc, leave=False)
        except Exception:
            iterator = dataloader

    autocast_enabled = bool(use_amp and device.type == "cuda")

    for batch in iterator:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
            logits = _forward_logits(model, input_ids, attention_mask)

        logits_f32 = logits.float()
        if loss_fn is not None:
            loss = loss_fn(logits_f32.reshape(-1, logits_f32.size(-1)), labels.reshape(-1))
            if torch.isfinite(loss):
                total_loss += float(loss.detach())
                num_batches += 1

        preds = logits_f32.argmax(dim=-1)
        mask = labels != IGNORE_INDEX
        gold_flat = labels[mask].to(torch.int64)
        pred_flat = preds[mask].to(torch.int64)
        total_supervised += int(gold_flat.numel())

        if gold_flat.numel():
            idx = gold_flat * NUM_LABELS + pred_flat
            counts = torch.bincount(idx, minlength=NUM_LABELS * NUM_LABELS)
            cm += counts.view(NUM_LABELS, NUM_LABELS).cpu().numpy()

    mean_loss = total_loss / num_batches if num_batches else None
    metrics = metrics_from_confusion_matrix(cm, loss=mean_loss)
    metrics["num_supervised_positions"] = total_supervised
    return metrics

def _forward_logits(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Call the model and return logits, handling both model families."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits if hasattr(out, "logits") else out

@torch.no_grad()
def predict_word_labels(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    max_batches: Optional[int] = None,
) -> Dict[str, List[str]]:
    model.eval()
    per_example: Dict[str, Dict[int, str]] = {}
    autocast_enabled = bool(use_amp and device.type == "cuda")

    for b_i, batch in enumerate(dataloader):
        if max_batches is not None and b_i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"]

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
            logits = _forward_logits(model, input_ids, attention_mask)
        preds = logits.float().argmax(dim=-1).cpu()

        word_index = batch.get("word_index")
        example_ids = batch["example_ids"]

        for row in range(preds.size(0)):
            ex_id = example_ids[row]
            slot = per_example.setdefault(ex_id, {})
            row_mask = labels[row] != IGNORE_INDEX
            positions = row_mask.nonzero(as_tuple=True)[0].tolist()
            for order, pos in enumerate(positions):
                if word_index is not None:
                    w = int(word_index[row, pos])
                else:
                    w = order
                slot[w] = ID2LABEL[int(preds[row, pos])]

    return {
        ex_id: [slot[k] for k in sorted(slot)] for ex_id, slot in per_example.items()
    }

def sample_predictions(
    model: torch.nn.Module,
    examples: Sequence[Any],
    predictions: Dict[str, List[str]],
    *,
    n: int = 10,
) -> List[Dict[str, Any]]:
    from src.data.statistics import render_text_with_punctuation

    rows: List[Dict[str, Any]] = []
    for ex in examples:
        pred = predictions.get(ex.id)
        if pred is None or len(pred) != len(ex.tokens):
            continue
        correct = sum(1 for g, p in zip(ex.labels, pred) if g == p)
        rows.append(
            {
                "id": ex.id,
                "input_text": " ".join(ex.tokens),
                "gold_text": render_text_with_punctuation(ex.tokens, ex.labels),
                "predicted_text": render_text_with_punctuation(ex.tokens, pred),
                "num_words": len(ex.tokens),
                "token_accuracy": correct / len(ex.tokens),
            }
        )
        if len(rows) >= n:
            break
    return rows
