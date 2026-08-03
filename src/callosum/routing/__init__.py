"""Learning-router architecture.

Replaces the legacy LLM-classifier + heuristic-fallback + circuit-breaker
machinery with a pluggable pipeline:

    request → features → capability filter → quality predict → cost-weighted select → dispatch

Each stage is a Protocol with swappable implementations chosen at startup
from config. Cold-start path runs with UniformPriorPredictor → cost
ordering picks cheapest compatible cell → local-first by default with
zero arbitrary thresholds.
"""
