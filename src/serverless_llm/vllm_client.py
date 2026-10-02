from dataclasses import dataclass


import json
import time
from json import JSONDecodeError
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from serverless_llm.config import ModelConfig, RequestConfig, ServerConfig

@dataclass(frozen=True)
class CompletionResult:
    """Store timing and output from one vLLM completion request."""

    request_started_at_seconds: float
    first_token_at_seconds: float
    completed_at_seconds: float
    generated_text: str
    finish_reason: str | None

    @property
    def ttft_seconds(self) -> float:
        """Return time from request start to the first generated token."""

        return (
            self.first_token_at_seconds
            - self.request_started_at_seconds
        )

    @property
    def generation_duration_seconds(self) -> float:
        """Return time from the first token to stream completion"""

        return (
        self.completed_at_seconds - self.first_token_at_seconds
        )

    @property
    def total_latency_seconds(self) -> float:
        """Return time from reqeust start to stream completion"""

        return(
        self.completed_at_seconds - self.request_started_at_seconds
        )

# 예외 클래스 
class VLLMClientError(RuntimeError):
    """Rasied When a vLLM completion request fails."""

class VLLMClient:
    """Send measured completion reqeusts to a running vLLM server."""

    def __init__(
            self,
            model_config: ModelConfig,
            server_config: ServerConfig,
            request_config: RequestConfig
            ) -> None:
            self._model_config = model_config # model name
            self._server_config = server_config # host, port, request timeout
            self._request_config = request_config # endpoint, prompt, temp

    @property # property 는 외부에서 별도의 인자를 받지 않는 getter 을 사용한다
    def completion_url(self) -> str:
        """Return the configured vLLM completion endpoint."""

        return(
            # http://127.0.0.1:8000/v1/completions
            f"http://{self._server_config.host}:"
            f"{self._server_config.port}"
            f"{self._request_config.endpoint}"
        )

    # build 는 prompt 을 전달받아야하기 떄문에 일반 메서드를 사용
    def _build_payload(self, prompt: str) -> dict[str, object]:
        """Build the JSON body sent to the vLLM completion endpoint."""

        return {
        "model": self._model_config.name,
        "prompt": prompt,
        "max_tokens": self._request_config.max_tokens,
        "temperature": self._request_config.temperature,
        "stream": self._request_config.stream,
        }

    def _build_request(self, prompt:str) -> Request:
        """Build an HTTP POST reqeust for one streamed completion."""

        palyload = self._build_payload(prompt)
        # JSON 문자열로 변환 -> 문자열을 bytes로 변환
        request_body = json.dumps(palyload).encode("utf-8")

        return Request(
            url=self.completion_url,
            data=request_body,
            headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            },
            method="POST",
        )

    @staticmethod
    def _decode_sse_line(raw_line: bytes) ->  str | None:
        """Extract the data fielf from one SSE response line."""
        # 입력: b'data: {"choices":[{"text":"Hello"}]}\n'
        # output: '{"choices":[{"text":"Hello"}]}' 
        # data 부분만 return 시킨다 prefix는 없애고

        try:
            #UTF-8 문자열로 변환한다
            line = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise VLLMClientError(
                "vLLM returned a non-UTF-8 streaming response"
            ) from error

        # Empty lines separate SSE events and contain no completion data.
        if not line:
            return None

        # Ignore SSE fields other than data, such as event or comment lines.
        if not line.startswith("data:"):
            return None

        return line.removeprefix("data:").strip()

    @staticmethod
    def _parse_completion_chunk(
        data: str,
    ) -> tuple[str, str | None]:
        """
        data = (
        '{"choices":[{"text":"Hello",'
        '"finish_reason":null}]}'
        ) ->
    
            {
        "choices": [
            {
                "text": "Hello",
                "finish_reason": None,
            }
        ]
    }
        """

        try:
            #JSON 문자열을 Pythond dict 으로 변환
            chunk = json.loads(data)
            choices = chunk["choices"]
            choice = choices[0]
            text_piece = choice.get("text", "")
            finish_reason = choice.get("finish_reason")
        except (
            AttributeError,
            JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            raise VLLMClientError(
                "vLLM returned an invalid completion chunk"
            ) from error

        if not isinstance(text_piece, str):
            raise VLLMClientError(
                "completion chunk text must be a string"
            )

        if (
            finish_reason is not None
            and not isinstance(finish_reason, str)
        ):
            raise VLLMClientError(
                "completion finish_reason must be a string or null"
            )

        return text_piece, finish_reason

    def complete(
    self,
    prompt: str | None = None,
    ) -> CompletionResult:
        """Send one streamed completion request and measure its latency."""

        selected_prompt = (
            self._request_config.prompt
            if prompt is None
            else prompt
        )

        if not isinstance(selected_prompt, str):
            raise ValueError("prompt must be a string")

        if not selected_prompt.strip():
            raise ValueError("prompt must not be empty")

        request = self._build_request(selected_prompt)

        request_started_at_seconds = time.monotonic()
        first_token_at_seconds: float | None = None
        generated_parts: list[str] = []
        finish_reason: str | None = None
        received_done_event = False

        try:
            with urlopen(
                request,
                timeout=self._server_config.request_timeout_seconds,
            ) as response:
                if response.status != 200:
                    raise VLLMClientError(
                        "vLLM returned unexpected HTTP status "
                        f"{response.status}"
                    )

                for raw_line in response:
                    data = self._decode_sse_line(raw_line)

                    # Empty and non-data SSE lines do not contain tokens.
                    if data is None:
                        continue

                    # The OpenAI-compatible stream ends with data: [DONE].
                    if data == "[DONE]":
                        received_done_event = True
                        break

                    text_piece, chunk_finish_reason = (
                        self._parse_completion_chunk(data)
                    )

                    # Record TTFT only for the first non-empty text piece.
                    if (
                        text_piece
                        and first_token_at_seconds is None
                    ):
                        first_token_at_seconds = time.monotonic()

                    generated_parts.append(text_piece)

                    if chunk_finish_reason is not None:
                        finish_reason = chunk_finish_reason

        except HTTPError as error:
            error_body = error.read().decode(
                "utf-8",
                errors="replace",
            )
            raise VLLMClientError(
                f"vLLM request failed with HTTP {error.code}: "
                f"{error_body}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise VLLMClientError(
                f"Could not complete the vLLM request: {error}"
            ) from error

        completed_at_seconds = time.monotonic()

        if not received_done_event:
            raise VLLMClientError(
                "vLLM stream ended without a [DONE] event"
            )

        if first_token_at_seconds is None:
            raise VLLMClientError(
                "vLLM stream completed without generated text"
            )

        return CompletionResult(
            request_started_at_seconds=request_started_at_seconds,
            first_token_at_seconds=first_token_at_seconds,
            completed_at_seconds=completed_at_seconds,
            generated_text="".join(generated_parts),
            finish_reason=finish_reason,
        )
