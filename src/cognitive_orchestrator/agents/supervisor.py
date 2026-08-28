"""Supervisor / main conversational agent.

The single front door the user talks to. It classifies the request: either it is
actionable (hand off to the cognitive loop) or it is ambiguous (ask a clarifying
question). In a full system this agent also streams natural-language progress
back to the user -- the "real-time user feedback" requirement.
"""
from __future__ import annotations

from ..llm import reasoning_llm

_SYSTEM = """ROLE: supervisor
You are the robot's conversational front-end. The robot has a camera and will perceive the scene itself, so do NOT ask about object counts, colors, sizes, or positions — those come from perception. Choose "clarify" ONLY if the request names no actionable task at all (e.g. a greeting or a vague "help me"). For any concrete instruction to manipulate, sort, move, or set aside objects, choose "execute". Return ONLY JSON: {"action":"execute|clarify","message":"..."}.
"""


def route(user_message: str) -> dict:
    return reasoning_llm.chat_json(_SYSTEM, user_message)
