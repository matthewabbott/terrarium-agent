#!/usr/bin/env python3
"""
Terrarium Agent HTTP API Server

FastAPI server implementing the agent HTTP API contract defined in AGENT_API.md.
Provides OpenAI-compatible chat completion endpoints for external clients.

Architecture:
- Stateless server (no conversation history storage)
- FIFO request queue (sequential processing)
- Persistent vLLM connection (warm)
- Clients manage their own conversation contexts
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import json
import httpx

from llm.vllm_client import VLLMClient


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models (OpenAI-compatible)
# ============================================================================

class ChatMessage(BaseModel):
    """Single message in conversation history."""
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID for tool responses")
    name: Optional[str] = Field(None, description="Tool name for tool responses")


class ChatCompletionRequest(BaseModel):
    """Request for chat completion endpoint."""
    model: Optional[str] = Field(None, description="Model name (auto-detected if omitted)")
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2048, ge=1, description="Maximum tokens to generate")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming (Server-Sent Events)")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Function/tool definitions")
    tool_choice: Optional[Any] = Field(None, description="Tool choice (e.g., 'auto' or a specific function)")


class GenerateRequest(BaseModel):
    """Request for simple generate endpoint."""
    prompt: str = Field(..., description="User's message")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(512, ge=1, description="Maximum tokens to generate")


class ChatCompletionResponse(BaseModel):
    """Response for chat completion endpoint (OpenAI-compatible)."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class GenerateResponse(BaseModel):
    """Response for simple generate endpoint."""
    response: str
    model: str


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str
    vllm_status: str
    model: Optional[str]
    queue_length: int


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vllm"


class ModelsResponse(BaseModel):
    """Response for list models endpoint."""
    object: str = "list"
    data: List[ModelInfo]


class TokenizeRequest(BaseModel):
    """Request for tokenize endpoint."""
    prompt: str = Field(..., description="Text to tokenize")


class TokenizeResponse(BaseModel):
    """Response for tokenize endpoint."""
    tokens: List[int] = Field(..., description="Token IDs")
    count: int = Field(..., description="Number of tokens")



# ============================================================================
# Request Queue (FIFO)
# ============================================================================

class RequestQueue:
    """
    FIFO request queue for sequential processing.

    Ensures one request at a time to leverage vLLM's automatic prefix caching.
    """

    def __init__(self):
        self.queue = asyncio.Queue()
        self.processing = False
        self._queue_length = 0

    async def enqueue(self, request_func):
        """
        Add request to queue and wait for processing.

        Args:
            request_func: Async function to execute

        Returns:
            Result from request_func
        """
        # Create future for this request
        future = asyncio.Future()

        # Add to queue
        await self.queue.put((request_func, future))
        self._queue_length += 1

        # Wait for result
        result = await future
        return result

    async def process_queue(self):
        """Process queue in FIFO order (runs as background task)."""
        while True:
            try:
                # Get next request
                request_func, future = await self.queue.get()
                self._queue_length -= 1

                # Process request
                try:
                    self.processing = True
                    result = await request_func()
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.processing = False
                    self.queue.task_done()

            except Exception as e:
                logger.error(f"Queue processing error: {e}")

    def get_length(self) -> int:
        """Get current queue length."""
        return self._queue_length


# ============================================================================
# Application State
# ============================================================================

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


class Metrics:
    """Lightweight counters for observability."""

    def __init__(self) -> None:
        self.http_requests = 0
        self.http_errors = 0
        self.http_latency_ms = 0.0
        self.stream_requests = 0
        self.stream_errors = 0
        self.stream_open = 0
        self.stream_latency_ms = 0.0

    def record_http(self, latency_ms: float, errored: bool) -> None:
        self.http_requests += 1
        self.http_latency_ms = latency_ms
        if errored:
            self.http_errors += 1

    def record_stream_start(self) -> None:
        self.stream_requests += 1
        self.stream_open += 1

    def record_stream_end(self, latency_ms: Optional[float] = None, errored: bool = False) -> None:
        self.stream_open = max(0, self.stream_open - 1)
        if latency_ms is not None:
            self.stream_latency_ms = latency_ms
        if errored:
            self.stream_errors += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "http_requests": self.http_requests,
            "http_errors": self.http_errors,
            "http_last_latency_ms": round(self.http_latency_ms, 2),
            "stream_requests": self.stream_requests,
            "stream_errors": self.stream_errors,
            "stream_open": self.stream_open,
            "stream_last_latency_ms": round(self.stream_latency_ms, 2),
        }


class AppState:
    """Global application state."""

    def __init__(self):
        self.vllm_client: Optional[VLLMClient] = None
        self.request_queue: RequestQueue = RequestQueue()
        self.vllm_url: str = "http://localhost:8000"
        self.model: Optional[str] = None
        self.startup_time: int = int(time.time())
        self.stream_semaphore: asyncio.Semaphore = asyncio.Semaphore(
            _env_int("AGENT_MAX_CONCURRENT_STREAMS", 4)
        )
        self.request_timeout_seconds: float = _env_float("AGENT_REQUEST_TIMEOUT_SECONDS", 120.0)
        self.metrics = Metrics()

    async def initialize(self):
        """Initialize vLLM client and verify connection."""
        logger.info("Initializing agent server...")

        # Create vLLM client
        self.vllm_client = VLLMClient(base_url=self.vllm_url)

        # Check health
        logger.info(f"Connecting to vLLM at {self.vllm_url}...")
        is_healthy = await self.vllm_client.health_check()

        if not is_healthy:
            raise Exception(f"vLLM server not responding at {self.vllm_url}")

        logger.info("✓ vLLM server is healthy")

        # Get models
        models = await self.vllm_client.get_models()

        if models:
            self.model = models[0]
            self.vllm_client.model = self.model
            logger.info(f"✓ Using model: {self.model}")
            if len(models) > 1:
                logger.info(f"  Available models: {', '.join(models)}")
        else:
            logger.warning("⚠️  No models detected")
            self.model = "default"

        # Start queue processor
        asyncio.create_task(self.request_queue.process_queue())
        logger.info("✓ Request queue started")

        logger.info("Agent server ready!")

    async def shutdown(self):
        """Cleanup resources."""
        logger.info("Shutting down agent server...")
        if self.vllm_client:
            await self.vllm_client.shutdown()
        logger.info("Shutdown complete")


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    await app_state.initialize()
    yield
    # Shutdown
    await app_state.shutdown()


app = FastAPI(
    title="Terrarium Agent API",
    description="HTTP API for Terrarium Agent - OpenAI-compatible chat completion service",
    version="1.0.0",
    lifespan=lifespan
)

# Global state
app_state = AppState()


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Format HTTP exceptions as OpenAI-compatible errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "http_error",
                "code": exc.status_code
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Format general exceptions as OpenAI-compatible errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": f"Internal server error: {str(exc)}",
                "type": "server_error",
                "code": 500
            }
        }
    )


# ============================================================================
# Endpoints
# ============================================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.

    Supports both streaming (SSE) and non-streaming responses.
    Clients manage their own conversation contexts.
    """
    # Validate request
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Invalid request: 'messages' field cannot be empty"
        )

    # Convert to format vLLM expects
    messages = []
    for msg in request.messages:
        entry: Dict[str, Any] = {"role": msg.role, "content": msg.content}
        # Include tool metadata if present
        if msg.tool_call_id:
            entry["tool_call_id"] = msg.tool_call_id
        if msg.name:
            entry["name"] = msg.name
        messages.append(entry)

    # Handle streaming response
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(messages, request),
            media_type="text/event-stream"
        )

    # Handle non-streaming response
    async def process_request():
        start = time.perf_counter()
        errored = False
        try:
            response = await asyncio.wait_for(
                app_state.vllm_client.chat(
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stop=request.stop,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                ),
                timeout=app_state.request_timeout_seconds,
            )
            return response
        except Exception as e:
            errored = True
            logger.error(f"Chat completion error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate response: {str(e)}"
            )
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            app_state.metrics.record_http(latency_ms, errored)

    # Enqueue and wait for processing
    raw_response = await app_state.request_queue.enqueue(process_request)

    if "choices" not in raw_response or not raw_response["choices"]:
        raise HTTPException(
            status_code=500,
            detail="Invalid response from model"
        )
    choice = raw_response["choices"][0]
    message = choice.get("message", {})

    # Format as OpenAI-compatible response
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        object="chat.completion",
        created=int(time.time()),
        model=app_state.model or "default",
        choices=[
            {
                "index": 0,
                "message": message,
                "finish_reason": choice.get("finish_reason", "stop")
            }
        ],
        usage={
            "prompt_tokens": 0,  # vLLM doesn't provide this easily
            "completion_tokens": 0,
            "total_tokens": 0
        }
    )


async def stream_chat_completion(messages: List[Dict], request: ChatCompletionRequest):
    """
    Stream chat completion chunks as Server-Sent Events (SSE).

    For simplicity, streaming requests bypass the queue and go directly to vLLM.
    This is acceptable because:
    1. Streaming is typically used for interactive UIs where responsiveness matters
    2. vLLM handles concurrent requests efficiently with continuous batching
    3. Queue is primarily for IRC bot use case which uses non-streaming

    This is an async generator that yields SSE-formatted chunks.
    """
    start = time.perf_counter()
    errored = False
    try:
        async with app_state.stream_semaphore:
            app_state.metrics.record_stream_start()
            # Stream from vLLM (no queue wait)
            async for chunk in app_state.vllm_client.chat_stream(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop,
                tools=request.tools,
                tool_choice=request.tool_choice,
            ):
                # Format as SSE: "data: {json}\n\n"
                yield f"data: {json.dumps(chunk)}\n\n"
                # Allow event loop to process
                await asyncio.sleep(0)

            # Send completion signal
            yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        errored = True
        # Send error as SSE event
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        app_state.metrics.record_stream_end(latency_ms, errored)


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Simple generation endpoint for single-turn interactions.

    Convenience endpoint that doesn't require message array formatting.
    """
    # Build messages array
    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.prompt})

    # Define request function
    async def process_request():
        try:
            response = await app_state.vllm_client.chat(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            return response
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate response: {str(e)}"
            )

    # Enqueue and wait for processing
    raw_response = await app_state.request_queue.enqueue(process_request)
    if "choices" not in raw_response or not raw_response["choices"]:
        raise HTTPException(
            status_code=500,
            detail="Invalid response from model"
        )
    response_text = raw_response["choices"][0].get("message", {}).get("content", "")

    return GenerateResponse(
        response=response_text,
        model=app_state.model or "default"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns server status and vLLM readiness.
    """
    # Check vLLM health
    vllm_status = "unknown"
    if app_state.vllm_client:
        is_healthy = await app_state.vllm_client.health_check()
        vllm_status = "ready" if is_healthy else "unavailable"

    # Determine overall status
    status = "healthy" if vllm_status == "ready" else "degraded"

    return HealthResponse(
        status=status,
        vllm_status=vllm_status,
        model=app_state.model,
        queue_length=app_state.request_queue.get_length()
    )


@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    """
    Tokenize text using vLLM's tokenizer.

    Returns token IDs and count for the input text.
    """
    if not app_state.vllm_client:
        raise HTTPException(status_code=503, detail="Server not ready")

    try:
        tokens = await app_state.vllm_client.tokenize(request.prompt)
        return TokenizeResponse(tokens=tokens, count=len(tokens))
    except Exception as e:
        logger.error(f"Tokenize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """
    Lightweight JSON metrics for observability.

    Includes proxy metrics and vLLM backend metrics (KV cache usage, request counts).
    """
    if not app_state.vllm_client:
        raise HTTPException(status_code=503, detail="Server not ready")
    payload = app_state.metrics.to_dict()
    payload["queue_length"] = app_state.request_queue.get_length()
    payload["model"] = app_state.model

    # Fetch vLLM metrics (Prometheus format)
    # Note: vLLM metric names may use colons or underscores depending on version
    # e.g., "vllm:gpu_cache_usage_perc" or "vllm_gpu_cache_usage_perc"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{app_state.vllm_url}/metrics", timeout=2.0)
            if resp.status_code == 200:
                # Parse Prometheus format for key metrics
                for raw_line in resp.text.splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Normalize: check for both colon and underscore variants
                    # Format: metric_name{labels} value  OR  metric_name value
                    # Note: vLLM uses "kv_cache_usage_perc" (not gpu/cpu split)
                    if "kv_cache_usage_perc" in line:
                        payload["vllm_kv_cache_pct"] = float(line.split()[-1])
                    elif "num_requests_running" in line:
                        payload["vllm_requests_running"] = int(float(line.split()[-1]))
                    elif "num_requests_waiting" in line:
                        payload["vllm_requests_waiting"] = int(float(line.split()[-1]))
    except Exception as e:
        logger.debug(f"Could not fetch vLLM metrics: {e}")
        # Metrics unavailable - fields simply won't be present

    return payload


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    List available models.

    Returns models available on the vLLM server.
    """
    if not app_state.vllm_client:
        raise HTTPException(
            status_code=503,
            detail="Server not ready"
        )

    models = await app_state.vllm_client.get_models()

    return ModelsResponse(
        object="list",
        data=[
            ModelInfo(
                id=model,
                object="model",
                created=app_state.startup_time,
                owned_by="vllm"
            )
            for model in models
        ]
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Terrarium Agent API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "generate": "/v1/generate",
            "health": "/health",
            "models": "/v1/models"
        },
        "docs": "/docs"
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Terrarium Agent HTTP API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with defaults
  python server.py

  # Custom port
  python server.py --port 9000

  # Custom vLLM URL
  python server.py --vllm-url http://192.168.1.100:8000

  # Enable auto-reload for development
  python server.py --reload

Server will run on http://localhost:8080 by default.
API documentation available at http://localhost:8080/docs
        """
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)"
    )
    parser.add_argument(
        "--vllm-url",
        default=os.getenv("VLLM_URL", "http://localhost:8000"),
        help="vLLM server URL (default: http://localhost:8000 or VLLM_URL env)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)"
    )

    args = parser.parse_args()

    # Update app state with config
    app_state.vllm_url = args.vllm_url

    # Update logging level
    logging.getLogger().setLevel(args.log_level.upper())

    # Run server
    logger.info(f"Starting Terrarium Agent API server on {args.host}:{args.port}")
    logger.info(f"vLLM backend: {args.vllm_url}")
    logger.info(f"API docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
