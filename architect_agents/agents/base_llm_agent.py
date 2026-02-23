import requests
import json


class BaseLLMAgent:
    def __init__(
        self,
        model="openai/gpt-oss-120b",
        api_key=None,
        temperature=0.3,
    ):
        self.model = model
        self.temperature = temperature

        if api_key is None:
            raise ValueError("API key required")

        self.headers = {
            "Authorization": f"Bearer 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",
            "Content-Type": "application/json",
        }

        self.url = "https://openrouter.ai/api/v1/chat/completions"

    # ----------------------------
    # single reasoning call
    # ----------------------------
    def call_llm(self, messages):

        response = requests.post(
            url=self.url,
            headers=self.headers,
            data=json.dumps({
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "reasoning": {"enabled": True}
            })
        )

        response = response.json()
        message = response["choices"][0]["message"]
        print(message["content"])
        return message["content"]

    # ----------------------------
    # helper for simple prompt
    # ----------------------------
    def generate(self, system_prompt, user_prompt):

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.call_llm(messages)
