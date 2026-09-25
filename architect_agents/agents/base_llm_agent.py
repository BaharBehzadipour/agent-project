# import requests
# import json


# class BaseLLMAgent:
#     def __init__(
#         self,
#         model="openai/gpt-oss-120b",
#         api_key="Bearer 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",
#         temperature=0.3,
#     ):
#         self.model = model
#         self.temperature = temperature

#         if api_key is None:
#             raise ValueError("API key required")

#         self.headers = {
#             "Authorization": f"Bearer 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",
#             "Content-Type": "application/json",
#         }

#         self.url = "https://openrouter.ai/api/v1/chat/completions"

#     # ----------------------------
#     # single reasoning call
#     # ----------------------------
#     def call_llm(self, messages):

#         response = requests.post(
#             url=self.url,
#             headers=self.headers,
#             data=json.dumps({
#                 "model": self.model,
#                 "messages": messages,
#                 "temperature": self.temperature,
#                 "reasoning": {"enabled": True}
#             })
#         )

#         response = response.json()
#         message = response["choices"][0]["message"]
#         print(message["content"])
#         return message["content"]

#     # ----------------------------
#     # helper for simple prompt
#     # ----------------------------
#     def generate(self, system_prompt, user_prompt):

#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ]

#         return self.call_llm(messages)



import os
import requests
import json


class BaseLLMAgent:
    def __init__(
        self,
        model="gemini-3.5-flash-lite",
        api_key=None,
        temperature=0.3,
    ):
        self.model = model
        self.temperature = temperature

        # Never hardcode API keys in source. Set with:
        #   export GEMINI_API_KEY=...
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set the GEMINI_API_KEY environment variable "
                "or pass api_key explicitly."
            )

        self.headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        self.url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )

    # ----------------------------
    # single reasoning call
    # ----------------------------
    def call_llm(self, messages):
        """
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        Gemini has no separate "system" role in the basic REST call, so we fold
        any system message into a systemInstruction block and convert the rest
        into Gemini's `contents` format (role "model" instead of "assistant").
        """
        system_parts = []
        contents = []

        for msg in messages:
            role = msg["role"]
            text = msg["content"]

            if role == "system":
                system_parts.append({"text": text})
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": text}]})

        payload = {
            "contents": contents,
            "generationConfig": {"temperature": self.temperature},
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        response = requests.post(
            url=self.url,
            headers=self.headers,
            data=json.dumps(payload),
        )
        response.raise_for_status()
        response = response.json()

        candidates = response.get("candidates", [])
        if not candidates:
            # e.g. blocked by safety filters -> promptFeedback explains why
            raise RuntimeError(f"No candidates returned: {response.get('promptFeedback')}")

        content = "".join(
            part.get("text", "") for part in candidates[0]["content"]["parts"]
        )
        print(content)
        return content

    # ----------------------------
    # helper for simple prompt
    # ----------------------------
    def generate(self, system_prompt, user_prompt):

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.call_llm(messages)
