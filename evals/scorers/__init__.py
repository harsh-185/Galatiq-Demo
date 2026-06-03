"""Scorers used by the eval scripts.

`code_scorers` are deterministic comparisons (no LLM cost, ~ms per call).
`llm_judges` are rubric-graded LLM calls (cost API tokens; rate-limited).

Convention: every scorer returns either a float in [0, 1] or a dict shaped
``{"name": str, "score": float, "metadata": dict}``. Both work as Braintrust
scorer outputs and as Python truthiness checks for local debugging.
"""
