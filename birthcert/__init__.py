"""Lightweight, anti-hallucination extractor for Egyptian birth certificates.

A focused pipeline (not the general case-study one): a single fixed template,
a small vision model (Qwen2.5-VL-3B), strict null-discipline prompting, and a
deterministic validation layer that rejects fabricated values.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
