# title: Mistral Agents
# author: Yuri Cunha
# version: 1.0.0
# license: MIT

import requests
import json
from typing import List, Dict, Generator
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        MISTRAL_API_KEY: str = Field(default="")
        AGENT_ID: str = Field(
            default="ag_"
        )  # Your Agent ID aqui or put in var, please, use VAR!

    def __init__(self):
        self.type = "manifold"
        self.id = "mistral-agent"
        self.name = "mistral-agent/"
        self.valves = self.Valves()
        self.agents_endpoint = "https://api.mistral.ai/v1/agents/completions"

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": "agent", "name": f"Mistral Agent ({self.valves.AGENT_ID[:10]})"}]

    def pipe(self, body: dict) -> Generator:
        headers = {
            "Authorization": f"Bearer {self.valves.MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "agent_id": self.valves.AGENT_ID,
            "messages": body["messages"],
            "stream": body.get("stream", False),
        }

        response = requests.post(
            self.agents_endpoint, headers=headers, json=payload, stream=True
        )

        for line in response.iter_lines():
            if line:
                yield line.decode("utf-8")
