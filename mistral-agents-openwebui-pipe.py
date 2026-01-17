# title: Mistral Agents
# author: Yuri Cunha
# version: 1.0.0
# license: MIT

import requests
import json
import base64
import mimetypes
from typing import List, Dict, Generator, Iterator, Optional, Tuple, Any, Union
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        MISTRAL_API_KEY: str = Field(default="")
        AGENT_ID: str = Field(
            default="ag_"
        )  # Your Agent ID aqui or put in var, please, use VAR!
        FILES_URL_EXPIRY_HOURS: int = Field(default=24)
        PASSTHROUGH_OPENWEBUI_TOOLS: bool = Field(default=False)

    def __init__(self):
        self.type = "manifold"
        self.id = "mistral-agent"
        self.name = "mistral-agent/"
        self.valves = self.Valves()
        self.agents_endpoint = "https://api.mistral.ai/v1/agents/completions"
        self.files_url_endpoint = "https://api.mistral.ai/v1/files/{file_id}/url"

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": "agent", "name": f"Mistral Agent ({self.valves.AGENT_ID[:10]})"}]

    def pipe(self, body: dict, __user__: Optional[dict] = None) -> Union[str, Generator[bytes, None, None], dict]:
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
            return self._stream_agent_completion(headers=headers, payload=payload)

        return self._non_stream_agent_completion(headers=headers, payload=payload)

    def _non_stream_agent_completion(self, headers: dict, payload: dict) -> str:
        try:
            response = requests.post(
                self.agents_endpoint,
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

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
            return f"Error: {e}"

    def _stream_agent_completion(self, headers: dict, payload: dict) -> Generator[bytes, None, None]:
        references: List[dict] = []
        file_chunks: List[dict] = []

        try:
            response = requests.post(
                self.agents_endpoint,
                headers=headers,
                json=payload,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

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
                        "error": {"message": str(e), "type": "pipe_error"},
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
            r = requests.get(url, headers=headers, params=params, timeout=60)
            r.raise_for_status()
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
            r = requests.get(content_url, headers=headers, timeout=120)
            r.raise_for_status()
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
