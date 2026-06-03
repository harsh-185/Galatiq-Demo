"""Variant configurations for eval experiments.

A variant is a set of environment-variable overrides that the eval harness
applies before running the pipeline. Each variant gets its own Braintrust
experiment, so the dashboard shows them side-by-side.

To add a variant: append a dict with {name, env, description}. The eval
scripts validate the name and apply the env via apply_variant().
"""
from __future__ import annotations

import os


VARIANTS: list[dict] = [
    {
        "name": "grok3_full",
        "env": {
            "GROK_MODEL": "grok-3",
            "GALATIQ_LLM_AGENTS": "1",
            "LLM_PROVIDER": "grok",
        },
        "description": "Production default: grok-3 + all LLM specialist agents on",
    },
    {
        "name": "grok3_mini",
        "env": {
            "GROK_MODEL": "grok-3-mini",
            "GALATIQ_LLM_AGENTS": "1",
            "LLM_PROVIDER": "grok",
        },
        "description": "Smaller, cheaper Grok variant",
    },
    {
        "name": "deterministic_baseline",
        "env": {
            "GALATIQ_LLM_AGENTS": "0",
        },
        "description": "All specialist LLMs disabled; only ingestion LLM fallback runs",
    },
    {
        "name": "openai_4omini",
        "env": {
            "LLM_PROVIDER": "openai",
            "OPENAI_MODEL": "gpt-4o-mini",
            "GALATIQ_LLM_AGENTS": "1",
        },
        "description": "OpenAI baseline for cross-family comparison",
    },
]


def apply_variant(name: str) -> dict:
    """Set env vars for the named variant; return the variant dict. Raises if
    the name isn't registered."""
    for v in VARIANTS:
        if v["name"] == name:
            for k, val in v["env"].items():
                os.environ[k] = val
            return v
    available = ", ".join(v["name"] for v in VARIANTS)
    raise ValueError(f"unknown variant {name!r}; available: {available}")
