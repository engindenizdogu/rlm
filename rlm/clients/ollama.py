from collections import defaultdict
from typing import Any

from ollama import AsyncClient, Client

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class OllamaClient(BaseLM):
    """
    LM Client for running models locally with Ollama.
    Ollama must be running locally (default: http://localhost:11434).
    """

    def __init__(
        self,
        model_name: str | None = None,
        host: str = "http://localhost:11434",
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.client = Client(host=host)
        self.async_client = AsyncClient(host=host)
        self.host = host

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Ollama client.")

        response = self.client.chat(model=model, messages=messages, stream=False)
        self._track_cost(response, model)
        return response["message"]["content"]

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Ollama client.")

        response = await self.async_client.chat(model=model, messages=messages, stream=False)
        self._track_cost(response, model)
        return response["message"]["content"]

    def _track_cost(self, response: dict[str, Any], model: str):
        self.model_call_counts[model] += 1

        # Ollama provides prompt_eval_count and eval_count
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)

        self.model_input_tokens[model] += input_tokens
        self.model_output_tokens[model] += output_tokens
        self.model_total_tokens[model] += input_tokens + output_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = input_tokens
        self.last_completion_tokens = output_tokens

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
