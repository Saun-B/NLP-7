"""Vietnamese Punctuation Restoration — source package.

Layout
------
``src.data``        Raw parsing, normalization, chunking, deduplication,
                    schema/validation, statistics, torch datasets.
``src.models``      BiLSTM tagger, PhoBERT token classifier, model factory.
``src.training``    Seeding, losses, optimizer/scheduler, trainers,
                    checkpoint manager.
``src.evaluation``  Metrics (including Punctuation Macro-F1) and evaluation.
``src.utils``       Shared IO, hashing, logging, environment capture.
"""

__version__ = "1.0.0-phase1"
