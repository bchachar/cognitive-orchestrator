"""Central configuration.

All environment-specific values (which vLLM endpoint, which model, where the
edge skill-service lives) are injected here so no other module hard-codes a URL.
This is the seam that lets the exact same agent code run:
  - fully local + mocked (CI / your laptop),
  - against vLLM on the A40 node (cloud/server reasoning),
  - talking to a skill-service running "on the robot" (edge).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CO_", env_file=".env", extra="ignore")

    # MOCK mode returns deterministic canned model outputs -> no GPU needed.
    mock_models: bool = True

    # Reasoning LLM (planner / supervisor / verifier), OpenAI-compatible (vLLM).
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen2.5-14B-Instruct"
    llm_api_key: str = "EMPTY"  # vLLM ignores it but the client requires a value

    # Vision-language model (perception). Separate endpoint => "distributed sub-agents".
    vlm_base_url: str = "http://localhost:8001/v1"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_api_key: str = "EMPTY"

    # Edge skill-service (the FastAPI app that would run on the robot).
    skill_service_url: str = "http://localhost:8080"

    # Cognitive-loop safety limits.
    max_replans: int = 2
    max_step_retries: int = 2  # per failed step, before falling back to a full replan
    request_timeout_s: float = 120.0


settings = Settings()
