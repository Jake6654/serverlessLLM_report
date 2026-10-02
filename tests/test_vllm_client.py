"""Unit tests for the streaming vLLM completion client."""

import io
import json
from urllib.error import HTTPError, URLError
from unittest.mock import Mock

import pytest

import serverless_llm.vllm_client as vllm_client_module
from serverless_llm.config import (
    ModelConfig,
    RequestConfig,
    ServerConfig,
)
from serverless_llm.vllm_client import (
    CompletionResult,
    VLLMClient,
    VLLMClientError,
)


class FakeStreamingResponse:
    """Provide a context-managed iterable that behaves like an HTTP stream."""

    def __init__(
        self,
        lines: list[bytes],
        *,
        status: int = 200,
    ) -> None:
        self._lines = lines
        self.status = status

    def __enter__(self) -> "FakeStreamingResponse":
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        return None

    def __iter__(self):
        return iter(self._lines)


@pytest.fixture
def model_config() -> ModelConfig:
    """Return deterministic model settings for client tests."""

    return ModelConfig(
        name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        revision="test-revision",
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        max_num_seqs=16,
        kv_cache_dtype="auto",
    )


@pytest.fixture
def server_config() -> ServerConfig:
    """Return deterministic server settings for client tests."""

    return ServerConfig(
        host="127.0.0.1",
        port=8000,
        startup_timeout_seconds=180,
        request_timeout_seconds=30,
        readiness_poll_interval_seconds=1.0,
    )


@pytest.fixture
def request_config() -> RequestConfig:
    """Return a streamed completion request configuration."""

    return RequestConfig(
        endpoint="/v1/completions",
        prompt="Explain serverless computing.",
        max_tokens=32,
        temperature=0.0,
        stream=True,
    )


@pytest.fixture
def client(
    model_config: ModelConfig,
    server_config: ServerConfig,
    request_config: RequestConfig,
) -> VLLMClient:
    """Create a vLLM client without contacting a real server."""

    return VLLMClient(
        model_config=model_config,
        server_config=server_config,
        request_config=request_config,
    )


def sse_chunk(
    text: str,
    finish_reason: str | None = None,
) -> bytes:
    """Encode one OpenAI-compatible completion chunk as an SSE line."""

    payload = {
        "choices": [
            {
                "text": text,
                "finish_reason": finish_reason,
            }
        ]
    }
    return f"data: {json.dumps(payload)}\n".encode("utf-8")


def test_completion_result_calculates_latency_components() -> None:
    result = CompletionResult(
        request_started_at_seconds=10.0,
        first_token_at_seconds=10.75,
        completed_at_seconds=12.5,
        generated_text="Hello",
        finish_reason="stop",
    )

    assert result.ttft_seconds == pytest.approx(0.75)
    assert result.generation_duration_seconds == pytest.approx(1.75)
    assert result.total_latency_seconds == pytest.approx(2.5)


def test_completion_url_uses_configured_endpoint(
    client: VLLMClient,
) -> None:
    assert client.completion_url == (
        "http://127.0.0.1:8000/v1/completions"
    )


def test_build_payload_uses_model_and_request_settings(
    client: VLLMClient,
) -> None:
    assert client._build_payload("A custom prompt") == {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "A custom prompt",
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": True,
    }


def test_build_request_creates_json_post_request(
    client: VLLMClient,
) -> None:
    request = client._build_request("A custom prompt")

    assert request.full_url == (
        "http://127.0.0.1:8000/v1/completions"
    )
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Accept") == "text/event-stream"
    assert json.loads(request.data.decode("utf-8")) == {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "A custom prompt",
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": True,
    }


@pytest.mark.parametrize(
    "raw_line",
    [
        b"\n",
        b": keep-alive\n",
        b"event: completion\n",
    ],
)
def test_decode_sse_line_ignores_non_data_lines(
    raw_line: bytes,
) -> None:
    assert VLLMClient._decode_sse_line(raw_line) is None


def test_decode_sse_line_extracts_data() -> None:
    assert VLLMClient._decode_sse_line(
        b'data: {"choices": []}\n'
    ) == '{"choices": []}'
    assert VLLMClient._decode_sse_line(
        b"data: [DONE]\n"
    ) == "[DONE]"


def test_decode_sse_line_rejects_non_utf8_data() -> None:
    with pytest.raises(
        VLLMClientError,
        match="non-UTF-8",
    ):
        VLLMClient._decode_sse_line(b"\xff\xfe")


def test_parse_completion_chunk_extracts_text_and_finish_reason() -> None:
    data = json.dumps(
        {
            "choices": [
                {
                    "text": "Hello",
                    "finish_reason": "stop",
                }
            ]
        }
    )

    assert VLLMClient._parse_completion_chunk(data) == (
        "Hello",
        "stop",
    )


@pytest.mark.parametrize(
    "data",
    [
        "not-json",
        json.dumps({}),
        json.dumps({"choices": []}),
        json.dumps({"choices": [None]}),
    ],
)
def test_parse_completion_chunk_rejects_invalid_structure(
    data: str,
) -> None:
    with pytest.raises(
        VLLMClientError,
        match="invalid completion chunk",
    ):
        VLLMClient._parse_completion_chunk(data)


def test_parse_completion_chunk_rejects_non_string_text() -> None:
    data = json.dumps(
        {
            "choices": [
                {
                    "text": 123,
                    "finish_reason": None,
                }
            ]
        }
    )

    with pytest.raises(VLLMClientError, match="text must be a string"):
        VLLMClient._parse_completion_chunk(data)


def test_parse_completion_chunk_rejects_invalid_finish_reason() -> None:
    data = json.dumps(
        {
            "choices": [
                {
                    "text": "Hello",
                    "finish_reason": 123,
                }
            ]
        }
    )

    with pytest.raises(
        VLLMClientError,
        match="finish_reason must be a string or null",
    ):
        VLLMClient._parse_completion_chunk(data)


def test_complete_uses_default_prompt_and_measures_stream(
    client: VLLMClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeStreamingResponse(
        [
            b"\n",
            sse_chunk(""),
            sse_chunk("Hello"),
            sse_chunk(" world", "stop"),
            b"data: [DONE]\n",
        ]
    )
    fake_urlopen = Mock(return_value=response)
    monotonic = Mock(side_effect=[10.0, 10.75, 12.5])
    monkeypatch.setattr(
        vllm_client_module,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        vllm_client_module.time,
        "monotonic",
        monotonic,
    )

    result = client.complete()

    assert result == CompletionResult(
        request_started_at_seconds=10.0,
        first_token_at_seconds=10.75,
        completed_at_seconds=12.5,
        generated_text="Hello world",
        finish_reason="stop",
    )
    request = fake_urlopen.call_args.args[0]
    assert json.loads(request.data.decode("utf-8"))["prompt"] == (
        "Explain serverless computing."
    )
    assert fake_urlopen.call_args.kwargs == {"timeout": 30}


def test_complete_uses_explicit_prompt(
    client: VLLMClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_urlopen = Mock(
        return_value=FakeStreamingResponse(
            [
                sse_chunk("Custom answer", "stop"),
                b"data: [DONE]\n",
            ]
        )
    )
    monkeypatch.setattr(
        vllm_client_module,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        vllm_client_module.time,
        "monotonic",
        Mock(side_effect=[1.0, 1.2, 1.5]),
    )

    client.complete("Custom prompt")

    request = fake_urlopen.call_args.args[0]
    assert json.loads(request.data.decode("utf-8"))["prompt"] == (
        "Custom prompt"
    )


@pytest.mark.parametrize("prompt", ["", "   ", 123])
def test_complete_rejects_invalid_prompt(
    client: VLLMClient,
    prompt: object,
) -> None:
    with pytest.raises(ValueError, match="prompt"):
        client.complete(prompt)  # type: ignore[arg-type]


def test_complete_reports_http_error(
    client: VLLMClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = HTTPError(
        url=client.completion_url,
        code=400,
        msg="Bad Request",
        hdrs=None,
        fp=io.BytesIO(b'{"error":"invalid model"}'),
    )
    monkeypatch.setattr(
        vllm_client_module,
        "urlopen",
        Mock(side_effect=error),
    )
    monkeypatch.setattr(
        vllm_client_module.time,
        "monotonic",
        Mock(return_value=1.0),
    )

    with pytest.raises(
        VLLMClientError,
        match=r"HTTP 400.*invalid model",
    ):
        client.complete()


@pytest.mark.parametrize(
    "network_error",
    [
        URLError("connection refused"),
        TimeoutError("request timed out"),
    ],
)
def test_complete_reports_network_and_timeout_errors(
    client: VLLMClient,
    monkeypatch: pytest.MonkeyPatch,
    network_error: Exception,
) -> None:
    monkeypatch.setattr(
        vllm_client_module,
        "urlopen",
        Mock(side_effect=network_error),
    )
    monkeypatch.setattr(
        vllm_client_module.time,
        "monotonic",
        Mock(return_value=1.0),
    )

    with pytest.raises(
        VLLMClientError,
        match="Could not complete the vLLM request",
    ):
        client.complete()


def test_complete_rejects_unexpected_http_status(
    client: VLLMClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vllm_client_module,
        "urlopen",
        Mock(return_value=FakeStreamingResponse([], status=204)),
    )
    monkeypatch.setattr(
        vllm_client_module.time,
        "monotonic",
        Mock(return_value=1.0),
    )

    with pytest.raises(
        VLLMClientError,
        match="unexpected HTTP status 204",
    ):
        client.complete()


def test_complete_rejects_stream_without_done_event(
    client: VLLMClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vllm_client_module,
        "urlopen",
        Mock(
            return_value=FakeStreamingResponse(
                [sse_chunk("partial text")]
            )
        ),
    )
    monkeypatch.setattr(
        vllm_client_module.time,
        "monotonic",
        Mock(side_effect=[1.0, 1.2, 1.5]),
    )

    with pytest.raises(
        VLLMClientError,
        match=r"without a \[DONE\] event",
    ):
        client.complete()


def test_complete_rejects_stream_without_generated_text(
    client: VLLMClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vllm_client_module,
        "urlopen",
        Mock(
            return_value=FakeStreamingResponse(
                [
                    sse_chunk("", "stop"),
                    b"data: [DONE]\n",
                ]
            )
        ),
    )
    monkeypatch.setattr(
        vllm_client_module.time,
        "monotonic",
        Mock(side_effect=[1.0, 1.5]),
    )

    with pytest.raises(
        VLLMClientError,
        match="without generated text",
    ):
        client.complete()
