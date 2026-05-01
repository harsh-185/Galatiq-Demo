"""LangChain `with_structured_output` wrapper that retries on Pydantic ValidationError."""
from __future__ import annotations

from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from galatiq.llm.client import get_chat_model

T = TypeVar("T", bound=BaseModel)


def extract_structured(
    schema: type[T],
    *,
    system: str,
    user: str,
    max_retries: int = 2,
) -> tuple[T, int]:
    """Call the LLM, coerce to `schema`, and self-correct on ValidationError.

    Returns the validated model plus the number of retries consumed.
    """
    llm = get_chat_model()
    structured = llm.with_structured_output(schema)
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = structured.invoke(messages)
            if not isinstance(result, schema):
                # LangChain occasionally returns a dict; coerce explicitly.
                result = schema.model_validate(result)
            return result, attempt
        except ValidationError as e:
            last_err = e
            messages.append(
                HumanMessage(
                    content=(
                        "Your previous response failed schema validation. "
                        "Fix the issues and return ONLY a JSON object matching the schema.\n\n"
                        f"Errors:\n{e}"
                    )
                )
            )
    raise RuntimeError(f"structured extraction failed after {max_retries} retries: {last_err}")
