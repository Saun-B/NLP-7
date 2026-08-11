from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.inference.predictor import PunctuationRestorationPredictor, RestorationResult
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
PathLike = Union[str, Path]

DEFAULT_MAX_WORDS = 5000

__all__ = [
    "PunctuationService",
    "ServiceResponse",
    "get_service",
    "reset_service",
]

@dataclass
class ServiceResponse:
    ok: bool
    input_text: str = ""
    restored_text: str = ""
    num_words: int = 0
    num_sentences: int = 0
    latency_ms: float = 0.0
    experiment_id: str = ""
    model_name: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PunctuationService:
    def __init__(
        self,
        *,
        model_selection_path: Optional[PathLike] = None,
        device: Optional[Any] = None,
        max_words: int = DEFAULT_MAX_WORDS,
        warmup: bool = True,
    ):
        self.model_selection_path = model_selection_path
        self.device = device
        self.max_words = max_words
        self._predictor: Optional[PunctuationRestorationPredictor] = None
        self._lock = threading.Lock()
        self._warmup_requested = warmup
        self._warmup_ms: Optional[float] = None

    @property
    def predictor(self) -> PunctuationRestorationPredictor:
        """Load on first use, once, thread-safely."""
        if self._predictor is None:
            with self._lock:
                if self._predictor is None:
                    self._predictor = PunctuationRestorationPredictor.from_selected_model(
                        self.model_selection_path, device=self.device
                    )
                    if self._warmup_requested:
                        self._warmup_ms = self._predictor.warmup()
        return self._predictor

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def model_info(self) -> Dict[str, Any]:
        info = dict(self.predictor.describe())
        info["warmup_ms"] = self._warmup_ms
        info["max_words"] = self.max_words
        return info

    def restore(self, text: str, **kwargs: Any) -> ServiceResponse:
        if text is None or not str(text).strip():
            return ServiceResponse(
                ok=False,
                input_text=text or "",
                error="Vui lòng nhập văn bản cần khôi phục dấu câu.",
                error_type="empty_input",
            )

        try:
            predictor = self.predictor
        except FileNotFoundError as exc:
            return self._failure(text, "missing_artifact", str(exc))
        except Exception as exc:
            return self._failure(text, type(exc).__name__, str(exc))

        try:
            result: RestorationResult = predictor.restore(
                text, max_words=self.max_words, **kwargs
            )
        except ValueError as exc:
            return self._failure(text, "input_too_long_or_invalid", str(exc))
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                return self._failure(
                    text,
                    "out_of_memory",
                    "GPU hết bộ nhớ khi xử lý văn bản này. Hãy thử đoạn ngắn hơn.",
                )
            return self._failure(text, "runtime_error", message)
        except Exception as exc:
            logger.exception("Unexpected inference failure")
            return self._failure(text, type(exc).__name__, str(exc))

        return ServiceResponse(
            ok=True,
            input_text=result.input_text,
            restored_text=result.restored_text,
            num_words=result.num_words,
            num_sentences=result.num_sentences,
            latency_ms=result.latency_ms,
            experiment_id=result.experiment_id,
            model_name=result.model_name,
            details=result.to_dict(),
        )

    def restore_many(self, texts: Sequence[str], **kwargs: Any) -> List[ServiceResponse]:
        return [self.restore(t, **kwargs) for t in texts]

    def _failure(self, text: str, error_type: str, message: str) -> ServiceResponse:
        logger.warning("Inference failed (%s): %s", error_type, message)
        return ServiceResponse(
            ok=False, input_text=text, error=message, error_type=error_type
        )

_service: Optional[PunctuationService] = None
_service_lock = threading.Lock()


def get_service(**kwargs: Any) -> PunctuationService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = PunctuationService(**kwargs)
    return _service

def reset_service() -> None:
    """Drop the shared service (used by tests)."""
    global _service
    with _service_lock:
        _service = None
