# Repository Guidelines

## Project Structure & Module Organization
- Core runtime lives in `agent/` (conversation loop, tool orchestration) with shared utilities in `llm/` for prompt management.
- Tool implementations are in `tools/` (IRC, shell, python, files). Add new tool modules under this directory and wire them via `config/tools.yaml`.
- HTTP entry points (`server.py`, `main.py`, `chat.py`) expect configs from `config/` plus session state in `sessions/`. Tests sit in `tests/` and scenario harnesses in `sessions_test/`.
- Assets such as models and Docker helpers reside in `models/` and scripts like `start_vllm_docker.sh`; keep large blobs outside git when possible.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate` — local virtualenv (required before the commands below).
- `pip install -r requirements.txt` — installs runtime, tooling, and local `terrarium-irc` dependency.
- `python server.py` starts the OpenAI-compatible HTTP API on `:8080`; `python main.py` launches the full agent runtime; `python chat.py` opens the interactive CLI.
- `pytest` runs the full suite; use `pytest tests/test_harness.py -k sse` for focused harness coverage.

## Coding Style & Naming Conventions
- Python throughout: 4-space indentation, `snake_case` for functions/vars, `PascalCase` for classes, and module-level constants in `UPPER_SNAKE_CASE`.
- Apply `black .` before opening a PR; rely on type hints and keep public interfaces annotated for `mypy`.
- Prefer explicit dataclasses/pydantic models for config payloads; keep tool IDs aligned with entries in `config/tools.yaml`.

## Testing Guidelines
- Tests live beside subject modules in `tests/` using `pytest` + `pytest-asyncio` (`async def` tests must include the `@pytest.mark.asyncio` decorator).
- Name files `test_<module>.py` and fixtures `*_fixture`. When adding tools, provide end-to-end exercises under `tests/tools/`.
- Target at least smoke coverage for new commands and report regressions by reproducing them via `pytest -k "<issue-id>"`.

## Commit & Pull Request Guidelines
- Follow the existing Conventional Commit style (`feat: add SSE streaming`, `fix: session loading bug`, `docs: update API guide`). Scope tags (e.g., `agent:`) are welcome when touching multiple areas.
- PRs should summarize behavior changes, list test commands run, and reference related issues or harness scenarios. Attach logs or screenshots when impacting the HTTP server or CLI UX.
- Ensure configuration or API changes mention required env vars or model artifacts so operators can reproduce the environment.

## Security & Configuration Tips
- Never commit secrets; reference environment variables loaded by `config/agent.yaml`.
- Validate file and tool permissions—new tools must respect sandbox constraints and avoid arbitrary filesystem writes outside the workspace.
- Verify vLLM Docker settings via `start_vllm_docker.sh` before deploying agent changes that rely on GPU inference.
- When serving Qwen3 locally for tool calls, use `start_vllm_docker_qwen3.sh` (defaults to `--tool-call-parser hermes`). No separate reasoning parser is needed; thinking is instruction-driven.
