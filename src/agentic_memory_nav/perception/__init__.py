"""Semantic perception backends, tracking, and sub-agent dispatch.

The unified sub-agent dispatcher (:class:`SubAgentDispatcher`) is the single entry-point
used by the Isaac Sim main loop to load (and guarantee) the LingBot-Map mapping agent at
simulation start. The LingBot model runtime (:class:`LingBotMapAgent`) is intentionally
NOT imported here: it requires the isolated LingBot venv (torch / lingbot_map) and is only
launched as a standalone process by :class:`LingBotMapAgentClient`.
"""

from agentic_memory_nav.perception.subagent import (
    LingBotMapAgentClient,
    SubAgentDispatcher,
)

__all__ = [
    "LingBotMapAgentClient",
    "SubAgentDispatcher",
]