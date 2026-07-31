"""study_buddy: provider-agnostic LLM toolkit for Phase 1 (LLM Fundamentals).

Swap between local Ollama and cloud providers (OpenAI, Gemini, DeepSeek)
behind one interface, with real USD cost tracking.
"""

from study_buddy.config import settings
from study_buddy.providers import get_provider
from study_buddy.providers.base import ChatResponse, LLMProvider, Message, Usage
from study_buddy.cost import CostTracker
from study_buddy.pricing import estimate_cost, load_pricing
from study_buddy.rag import RagStore
from study_buddy.tokens import count_message_tokens, count_tokens

__all__ = [
    "settings",
    "get_provider",
    "ChatResponse",
    "LLMProvider",
    "Message",
    "Usage",
    "CostTracker",
    "estimate_cost",
    "load_pricing",
    "RagStore",
    "count_tokens",
    "count_message_tokens",
]
