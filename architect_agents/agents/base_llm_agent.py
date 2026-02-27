import requests
import json


class BaseLLMAgent:
    def __init__(
        self,
        model="openai/gpt-oss-120b",
        api_key="Bearer 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",
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



