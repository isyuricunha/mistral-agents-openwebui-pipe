# title: Mistral Agents
# author: Yuri Cunha
# version: 1.4.0
# license: AGPL-3.0

import requests
import json
import base64
import mimetypes
import time
from typing import List, Dict, Generator, Iterator, Optional, Tuple, Union
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        MISTRAL_API_KEY: str = Field(default="")
        AGENT_ID: str = Field(
            default="ag_"
        )  # Your Agent ID aqui or put in var, please, use VAR!
        FILES_URL_EXPIRY_HOURS: int = Field(default=24)
        PASSTHROUGH_OPENWEBUI_TOOLS: bool = Field(default=False)
        USE_BETA_CONVERSATIONS: bool = Field(default=True)
        HTTP_MAX_RETRIES: int = Field(default=3)
        HTTP_BACKOFF_BASE_SECONDS: float = Field(default=1.0)
        HTTP_BACKOFF_MAX_SECONDS: float = Field(default=20.0)
        DEBUG_RATE_LIMIT: bool = Field(default=False)
        MIN_SECONDS_BETWEEN_CONVERSATION_CALLS: float = Field(default=0.0)
        COOLDOWN_SECONDS_AFTER_IMAGE: float = Field(default=0.0)
        COOLDOWN_SECONDS_ON_429: float = Field(default=10.0)

    def __init__(self):
        self.type = "manifold"
        self.id = "mistral-agent"
        self.name = "mistral-agent/"
        self.valves = self.Valves()
        self.agents_endpoint = "https://api.mistral.ai/v1/agents/completions"
        self.conversations_start_endpoint = "https://api.mistral.ai/v1/conversations"
        self.conversations_append_endpoint = "https://api.mistral.ai/v1/conversations/{conversation_id}"
        self.files_url_endpoint = "https://api.mistral.ai/v1/files/{file_id}/url"
        self._conversation_ids_by_chat_id: Dict[str, str] = {}
        self._last_conversation_call_ts_by_chat_id: Dict[str, float] = {}
        self._cooldown_until_ts_by_chat_id: Dict[str, float] = {}
        self._global_cooldown_until_ts: float = 0.0

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": "agent", "name": f"Mistral Agent ({self.valves.AGENT_ID[:10]})"}]

    def pipe(self, body: dict, __user__: Optional[dict] = None) -> Union[str, Generator[bytes, None, None], dict]:
        local_text = self._get_last_user_text(body)
        remaining = self._get_global_cooldown_remaining_seconds()
        if remaining > 0:
            if self._is_rate_limit_question(local_text):
                return self._local_rate_limit_explanation(remaining)
            return self._local_cooldown_message(remaining)

        headers = {
            "Authorization": f"Bearer {self.valves.MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "agent_id": self.valves.AGENT_ID,
            "messages": body.get("messages", []),
            "stream": bool(body.get("stream", False)),
        }

        extra_passthrough_keys = [
            "frequency_penalty",
            "max_tokens",
            "metadata",
            "n",
            "prediction",
            "presence_penalty",
            "prompt_mode",
            "random_seed",
            "response_format",
            "stop",
            "temperature",
            "top_p",
        ]

        if self.valves.PASSTHROUGH_OPENWEBUI_TOOLS:
            extra_passthrough_keys.extend(
                [
                    "parallel_tool_calls",
                    "tool_choice",
                    "tools",
                ]
            )
        for key in extra_passthrough_keys:
            if key in body:
                payload[key] = body[key]

        if not self.valves.MISTRAL_API_KEY:
            return "Error: MISTRAL_API_KEY is not configured in valves."
        if not self.valves.AGENT_ID or self.valves.AGENT_ID == "ag_":
            return "Error: AGENT_ID is not configured in valves."

        if payload["stream"]:
            if self.valves.USE_BETA_CONVERSATIONS:
                return self._stream_conversation_completion(headers=headers, body=body)
            return self._stream_agent_completion(headers=headers, payload=payload)

        if self.valves.USE_BETA_CONVERSATIONS:
            return self._non_stream_conversation_completion(headers=headers, body=body)
        return self._non_stream_agent_completion(headers=headers, payload=payload)

    def _get_last_user_text(self, body: dict) -> str:
        messages = body.get("messages")
        if not isinstance(messages, list):
            return ""

        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            return str(content)
        return ""

    def _is_rate_limit_question(self, text: str) -> bool:
        t = (text or "").lower()
        triggers = [
            "erro 429",
            "error 429",
            "429",
            "rate limit",
            "too many requests",
            "limite",
            "limitação",
            "limite de taxa",
        ]
        return any(x in t for x in triggers)

    def _get_global_cooldown_remaining_seconds(self) -> int:
        now = time.monotonic()
        remaining = self._global_cooldown_until_ts - now
        return int(remaining) + 1 if remaining > 0 else 0

    def _set_global_cooldown_seconds(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        if seconds <= 0:
            return
        until = time.monotonic() + seconds
        self._global_cooldown_until_ts = max(
            self._global_cooldown_until_ts, until)
        if self.valves.DEBUG_RATE_LIMIT:
            print(f"[mistral-rl] global cooldown seconds={seconds:.2f}")

    def _local_cooldown_message(self, remaining_seconds: int) -> str:
        return (
            f"Rate limit da Mistral atingido (HTTP 429). "
            f"Aguarde ~{remaining_seconds}s e tente novamente."
        )

    def _local_rate_limit_explanation(self, remaining_seconds: int) -> str:
        return (
            "HTTP 429 (Too Many Requests) significa que você excedeu o limite de requisições/tokens "
            "do seu workspace na Mistral (rate limit). "
            f"Neste momento, aguarde ~{remaining_seconds}s e tente de novo."
        )

    def _extract_last_user_input(self, body: dict) -> Union[str, List[dict]]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return ""

        last_user = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user = msg
                break

        if not isinstance(last_user, dict):
            return ""

        content = last_user.get("content", "")
        return [{"role": "user", "content": content}]

    def _request_with_retries(
        self,
        method: str,
        url: str,
        headers: dict,
        json_payload: Optional[dict],
        params: Optional[dict],
        timeout: int,
        stream: bool = False,
    ) -> requests.Response:
        last_exc: Optional[Exception] = None
        max_retries = max(0, int(self.valves.HTTP_MAX_RETRIES))

        for attempt in range(max_retries + 1):
            try:
                if self.valves.DEBUG_RATE_LIMIT:
                    print(f"[mistral-http] {method} {url} attempt={attempt}")

                r = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_payload,
                    timeout=timeout,
                    stream=stream,
                )

                if self.valves.DEBUG_RATE_LIMIT:
                    print(f"[mistral-http] status={r.status_code} url={url}")

                if r.status_code != 429:
                    r.raise_for_status()
                    return r

                if self.valves.DEBUG_RATE_LIMIT:
                    retry_after_header = r.headers.get("Retry-After")
                    ratelimit_headers = {
                        k: v
                        for k, v in r.headers.items()
                        if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"
                    }
                    print(
                        f"[mistral-http] 429 url={url} retry-after={retry_after_header} headers={ratelimit_headers}"
                    )

                retry_after = r.headers.get("Retry-After")
                sleep_seconds: float
                if retry_after:
                    try:
                        sleep_seconds = float(retry_after)
                    except ValueError:
                        sleep_seconds = self.valves.HTTP_BACKOFF_BASE_SECONDS
                else:
                    sleep_seconds = float(
                        self.valves.HTTP_BACKOFF_BASE_SECONDS) * (2**attempt)

                cooldown_seconds = sleep_seconds
                if cooldown_seconds <= 0:
                    cooldown_seconds = float(
                        self.valves.COOLDOWN_SECONDS_ON_429 or 0.0)
                self._set_global_cooldown_seconds(cooldown_seconds)

                sleep_seconds = min(
                    float(self.valves.HTTP_BACKOFF_MAX_SECONDS), max(0.0, sleep_seconds))
                if attempt >= max_retries:
                    r.raise_for_status()
                time.sleep(sleep_seconds)

            except Exception as e:
                last_exc = e
                if attempt >= max_retries:
                    raise
                sleep_seconds = float(
                    self.valves.HTTP_BACKOFF_BASE_SECONDS) * (2**attempt)
                sleep_seconds = min(
                    float(self.valves.HTTP_BACKOFF_MAX_SECONDS), max(0.0, sleep_seconds))
                time.sleep(sleep_seconds)

        raise RuntimeError(last_exc) if last_exc else RuntimeError(
            "Request failed")

    def _format_http_error(self, e: Exception) -> str:
        if isinstance(e, requests.HTTPError) and e.response is not None:
            status = e.response.status_code
            if status == 429:
                retry_after = e.response.headers.get("Retry-After")
                hint = f" Retry after {retry_after}s." if retry_after else ""
                return f"Rate limit hit (HTTP 429). Please wait a bit and try again.{hint}"
            return f"HTTP {status}: {e.response.text[:500]}"
        return str(e)

    def _apply_conversation_rate_limit(self, body: dict) -> None:
        chat_id = body.get("chat_id")
        if not isinstance(chat_id, str) or not chat_id:
            return

        now = time.monotonic()

        cooldown_until = self._cooldown_until_ts_by_chat_id.get(chat_id, 0.0)
        if cooldown_until > now:
            wait_seconds = cooldown_until - now
            if self.valves.DEBUG_RATE_LIMIT:
                print(
                    f"[mistral-rl] cooldown chat_id={chat_id} sleep={wait_seconds:.2f}s")
            time.sleep(wait_seconds)
            now = time.monotonic()

        min_interval = float(
            self.valves.MIN_SECONDS_BETWEEN_CONVERSATION_CALLS or 0.0)
        if min_interval <= 0:
            return

        last_ts = self._last_conversation_call_ts_by_chat_id.get(chat_id)
        if last_ts is None:
            return

        elapsed = now - last_ts
        if elapsed >= min_interval:
            return

        wait_seconds = min_interval - elapsed
        if self.valves.DEBUG_RATE_LIMIT:
            print(
                f"[mistral-rl] min-interval chat_id={chat_id} sleep={wait_seconds:.2f}s")
        time.sleep(wait_seconds)

    def _start_conversation(self, headers: dict, body: dict) -> dict:
        chat_id = body.get("chat_id")

        self._apply_conversation_rate_limit(body)

        inputs = self._extract_last_user_input(body)
        if not inputs:
            return {}

        payload: dict = {
            "agent_id": self.valves.AGENT_ID,
            "inputs": inputs,
            "stream": False,
        }

        completion_args: dict = {}
        for k in ["temperature", "top_p", "max_tokens", "stop", "random_seed", "presence_penalty", "frequency_penalty"]:
            if k in body:
                completion_args[k] = body[k]
        if completion_args:
            payload["completion_args"] = completion_args

        r = self._request_with_retries(
            method="POST",
            url=self.conversations_start_endpoint,
            headers=headers,
            json_payload=payload,
            params=None,
            timeout=180,
        )
        data = r.json()

        if isinstance(chat_id, str) and chat_id:
            self._last_conversation_call_ts_by_chat_id[chat_id] = time.monotonic(
            )
        conv_id = data.get("conversation_id")
        if isinstance(conv_id, str) and isinstance(chat_id, str) and chat_id:
            self._conversation_ids_by_chat_id[chat_id] = conv_id
        return data

    def _append_conversation(self, headers: dict, conversation_id: str, body: dict) -> dict:
        chat_id = body.get("chat_id")

        self._apply_conversation_rate_limit(body)
        inputs = self._extract_last_user_input(body)
        if not inputs:
            return {}

        payload: dict = {
            "inputs": inputs,
            "stream": False,
            "store": True,
            "handoff_execution": "server",
        }

        completion_args: dict = {}
        for k in ["temperature", "top_p", "max_tokens", "stop", "random_seed", "presence_penalty", "frequency_penalty"]:
            if k in body:
                completion_args[k] = body[k]
        if completion_args:
            payload["completion_args"] = completion_args

        url = self.conversations_append_endpoint.format(
            conversation_id=conversation_id)
        r = self._request_with_retries(
            method="POST",
            url=url,
            headers=headers,
            json_payload=payload,
            params=None,
            timeout=180,
        )
        data = r.json()

        if isinstance(chat_id, str) and chat_id:
            self._last_conversation_call_ts_by_chat_id[chat_id] = time.monotonic(
            )

        new_conv_id = data.get("conversation_id")
        if isinstance(chat_id, str) and chat_id and isinstance(new_conv_id, str) and new_conv_id:
            self._conversation_ids_by_chat_id[chat_id] = new_conv_id

        return data

    def _extract_from_conversation_response(self, data: dict) -> Tuple[str, List[dict], List[dict]]:
        outputs = data.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return "", [], []

        last_message_output = None
        for entry in reversed(outputs):
            if not isinstance(entry, dict):
                continue
            if entry.get("type") == "message.output" and entry.get("role") == "assistant":
                last_message_output = entry
                break

        if not isinstance(last_message_output, dict):
            for entry in reversed(outputs):
                if isinstance(entry, dict) and entry.get("type") == "message.output":
                    last_message_output = entry
                    break

        if not isinstance(last_message_output, dict):
            return "", [], []

        text, references, file_chunks = self._extract_text_references_files(
            last_message_output)
        return text, references, file_chunks

    def _non_stream_conversation_completion(self, headers: dict, body: dict) -> str:
        try:
            chat_id = body.get("chat_id")
            conversation_id = ""
            if isinstance(chat_id, str) and chat_id in self._conversation_ids_by_chat_id:
                conversation_id = self._conversation_ids_by_chat_id[chat_id]

            if conversation_id:
                data = self._append_conversation(
                    headers=headers, conversation_id=conversation_id, body=body)
            else:
                data = self._start_conversation(headers=headers, body=body)
                if not isinstance(data, dict) or not data.get("conversation_id"):
                    return "Error: Could not start a Mistral conversation."

            text, references, file_chunks = self._extract_from_conversation_response(
                data)

            if isinstance(chat_id, str) and chat_id and file_chunks and float(self.valves.COOLDOWN_SECONDS_AFTER_IMAGE or 0.0) > 0:
                self._cooldown_until_ts_by_chat_id[chat_id] = time.monotonic(
                ) + float(self.valves.COOLDOWN_SECONDS_AFTER_IMAGE)
                if self.valves.DEBUG_RATE_LIMIT:
                    print(
                        f"[mistral-rl] image cooldown chat_id={chat_id} seconds={float(self.valves.COOLDOWN_SECONDS_AFTER_IMAGE):.2f}"
                    )

            extra = self._build_extra_markdown(
                headers=headers, references=references, file_chunks=file_chunks)
            if extra:
                text = f"{text}\n\n{extra}" if text else extra
            return text
        except Exception as e:
            return f"Error: {self._format_http_error(e)}"

    def _stream_conversation_completion(self, headers: dict, body: dict) -> Generator[bytes, None, None]:
        try:
            # To reliably include images and sources, we execute non-stream and emulate OpenAI SSE.
            text = self._non_stream_conversation_completion(
                headers=headers, body=body)
            yield (
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"content": text},
                                "index": 0,
                            }
                        ],
                        "object": "chat.completion.chunk",
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            ).encode("utf-8")
            yield b"data: [DONE]\n\n"
        except Exception as e:
            yield (
                "data: "
                + json.dumps(
                    {
                        "error": {"message": self._format_http_error(e), "type": "pipe_error"},
                        "object": "error",
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            ).encode("utf-8")
            yield b"data: [DONE]\n\n"

    def _non_stream_agent_completion(self, headers: dict, payload: dict) -> str:
        try:
            response = self._request_with_retries(
                method="POST",
                url=self.agents_endpoint,
                headers=headers,
                json_payload=payload,
                params=None,
                timeout=120,
            )
            data = response.json()
            message = self._extract_message_from_completion(data)
            text, references, file_chunks = self._extract_text_references_files(
                message)
            extra = self._build_extra_markdown(
                headers=headers, references=references, file_chunks=file_chunks)
            if extra:
                text = f"{text}\n\n{extra}" if text else extra
            return text
        except Exception as e:
            return f"Error: {self._format_http_error(e)}"

    def _stream_agent_completion(self, headers: dict, payload: dict) -> Generator[bytes, None, None]:
        references: List[dict] = []
        file_chunks: List[dict] = []

        try:
            response = self._request_with_retries(
                method="POST",
                url=self.agents_endpoint,
                headers=headers,
                json_payload=payload,
                params=None,
                timeout=120,
                stream=True,
            )

            for event in self._iter_sse_events(response):
                if event is None:
                    break

                updated_event, new_refs, new_files = self._normalize_stream_event(
                    event)
                references.extend(new_refs)
                file_chunks.extend(new_files)

                if updated_event is not None:
                    yield f"data: {json.dumps(updated_event, ensure_ascii=False)}\n\n".encode("utf-8")

            extra = self._build_extra_markdown(
                headers=headers, references=references, file_chunks=file_chunks)
            if extra:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "choices": [
                                {
                                    "delta": {"content": f"\n\n{extra}"},
                                    "index": 0,
                                }
                            ],
                            "object": "chat.completion.chunk",
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                ).encode("utf-8")

            yield b"data: [DONE]\n\n"
        except Exception as e:
            yield (
                "data: "
                + json.dumps(
                    {
                        "error": {"message": self._format_http_error(e), "type": "pipe_error"},
                        "object": "error",
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            ).encode("utf-8")
            yield b"data: [DONE]\n\n"

    def _iter_sse_events(self, response: requests.Response) -> Iterator[Optional[dict]]:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue

            data = line[len("data:"):].strip()
            if not data:
                continue
            if data == "[DONE]":
                yield None
                return

            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue

    def _extract_message_from_completion(self, data: dict) -> dict:
        choices = data.get("choices") or []
        if not choices:
            return {}

        first = choices[0] or {}
        message = first.get("message") or {}
        if isinstance(message, dict) and "tool_calls" in message:
            message = {**message}
            message.pop("tool_calls", None)
        return message

    def _extract_text_references_files(
        self, message: dict
    ) -> Tuple[str, List[dict], List[dict]]:
        content = message.get("content")
        if isinstance(content, str):
            return content, [], []
        if not isinstance(content, list):
            return "", [], []

        text_parts: List[str] = []
        references: List[dict] = []
        file_chunks: List[dict] = []

        for chunk in content:
            if not isinstance(chunk, dict):
                continue

            chunk_type = chunk.get("type")
            if chunk_type == "text":
                piece = chunk.get("text")
                if isinstance(piece, str) and piece:
                    text_parts.append(piece)
            elif chunk_type == "tool_reference":
                references.append(chunk)
            elif chunk_type == "tool_file":
                file_chunks.append(chunk)

        return "".join(text_parts), references, file_chunks

    def _normalize_stream_event(self, event: dict) -> Tuple[Optional[dict], List[dict], List[dict]]:
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return event, [], []

        new_event = {**event}
        new_choices: List[dict] = []
        collected_refs: List[dict] = []
        collected_files: List[dict] = []

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                new_choices.append(choice)
                continue

            new_delta_base = {**delta}
            if "tool_calls" in new_delta_base:
                new_delta_base.pop("tool_calls", None)

            content = delta.get("content")
            if isinstance(content, list):
                text_parts: List[str] = []
                for chunk in content:
                    if not isinstance(chunk, dict):
                        continue
                    chunk_type = chunk.get("type")
                    if chunk_type == "text":
                        piece = chunk.get("text")
                        if isinstance(piece, str) and piece:
                            text_parts.append(piece)
                    elif chunk_type == "tool_reference":
                        collected_refs.append(chunk)
                    elif chunk_type == "tool_file":
                        collected_files.append(chunk)

                new_delta = new_delta_base
                new_text = "".join(text_parts)
                new_delta["content"] = new_text if new_text else ""
                new_choice = {**choice, "delta": new_delta}
                new_choices.append(new_choice)
            else:
                new_choice = {**choice, "delta": new_delta_base}
                new_choices.append(new_choice)

        new_event["choices"] = new_choices
        return new_event, collected_refs, collected_files

    def _build_extra_markdown(self, headers: dict, references: List[dict], file_chunks: List[dict]) -> str:
        sources_md = self._format_sources_markdown(references)
        images_md = self._format_images_markdown(
            headers=headers, file_chunks=file_chunks)

        parts = [part for part in [images_md, sources_md] if part]
        return "\n\n".join(parts)

    def _format_sources_markdown(self, references: List[dict]) -> str:
        cleaned: List[Tuple[str, str, str]] = []
        seen_urls = set()

        for ref in references:
            if not isinstance(ref, dict):
                continue
            url = ref.get("url")
            if not isinstance(url, str) or not url:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = ref.get("title")
            source = ref.get("source")
            cleaned.append(
                (
                    title if isinstance(title, str) and title else url,
                    url,
                    source if isinstance(source, str) and source else "",
                )
            )

        if not cleaned:
            return ""

        lines = ["## Sources"]
        for idx, (title, url, source) in enumerate(cleaned, start=1):
            suffix = f" - {source}" if source else ""
            lines.append(f"{idx}. [{title}]({url}){suffix}")
        return "\n".join(lines)

    def _format_images_markdown(self, headers: dict, file_chunks: List[dict]) -> str:
        urls: List[str] = []
        for chunk in file_chunks:
            if not isinstance(chunk, dict):
                continue
            file_id = chunk.get("file_id")
            if not isinstance(file_id, str) or not file_id:
                continue

            signed_url = self._get_file_signed_url(
                headers=headers, file_id=file_id)
            if signed_url:
                urls.append(signed_url)
                continue

            fallback = self._download_file_as_data_url(
                headers=headers, chunk=chunk)
            if fallback:
                urls.append(fallback)

        if not urls:
            return ""

        return "\n".join([f"![image]({url})" for url in urls])

    def _get_file_signed_url(self, headers: dict, file_id: str) -> str:
        try:
            url = self.files_url_endpoint.format(file_id=file_id)
            params = {"expiry": int(self.valves.FILES_URL_EXPIRY_HOURS)}
            r = self._request_with_retries(
                method="GET",
                url=url,
                headers=headers,
                json_payload=None,
                params=params,
                timeout=60,
            )
            data = r.json()
            signed = data.get("url")
            return signed if isinstance(signed, str) else ""
        except Exception:
            return ""

    def _download_file_as_data_url(self, headers: dict, chunk: dict) -> str:
        try:
            file_id = chunk.get("file_id")
            if not isinstance(file_id, str) or not file_id:
                return ""

            content_url = f"https://api.mistral.ai/v1/files/{file_id}/content"
            r = self._request_with_retries(
                method="GET",
                url=content_url,
                headers=headers,
                json_payload=None,
                params=None,
                timeout=120,
            )
            file_bytes = r.content

            file_name = chunk.get("file_name")
            file_type = chunk.get("file_type")
            mime = ""
            if isinstance(file_type, str) and file_type:
                mime = mimetypes.types_map.get(f".{file_type.lower()}", "")
            if not mime and isinstance(file_name, str) and file_name:
                guessed, _ = mimetypes.guess_type(file_name)
                mime = guessed or ""
            if not mime:
                mime = "application/octet-stream"

            encoded = base64.b64encode(file_bytes).decode("ascii")
            return f"data:{mime};base64,{encoded}"
        except Exception:
            return ""
