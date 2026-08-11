"""Inference: text in, punctuated text out, using the locked winner checkpoint.

    from src.inference import PunctuationRestorationPredictor

    predictor = PunctuationRestorationPredictor.from_selected_model()
    print(predictor.restore("hôm nay trời đẹp bạn có muốn đi dạo không").restored_text)

This prints ``Hôm nay trời đẹp. Bạn có muốn đi dạo không?``

Modules
-------
``tokenizer``      free text → lexical words (URLs, emails, numbers protected)
``predictor``      checkpoint loading + label prediction (windowed, no truncation)
``reconstruction`` words + labels → punctuated, sentence-capitalised text
``service``        cached singleton + structured error handling for CLI/UI
"""

from src.inference.predictor import PunctuationRestorationPredictor, RestorationResult
from src.inference.reconstruction import (
    ReconstructionResult,
    reconstruct_text,
    reconstruct_with_details,
    split_into_sentences,
)
from src.inference.service import (
    PunctuationService,
    ServiceResponse,
    get_service,
    reset_service,
)
from src.inference.tokenizer import (
    InferenceTokenizer,
    TokenizedInput,
    tokenize_for_inference,
)

__all__ = [
    "PunctuationRestorationPredictor",
    "RestorationResult",
    "InferenceTokenizer",
    "TokenizedInput",
    "tokenize_for_inference",
    "reconstruct_text",
    "reconstruct_with_details",
    "split_into_sentences",
    "ReconstructionResult",
    "PunctuationService",
    "ServiceResponse",
    "get_service",
    "reset_service",
]
