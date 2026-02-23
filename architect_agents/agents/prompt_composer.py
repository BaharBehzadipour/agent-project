# class PromptComposer:
#     def __init__(self, agents):
#         self.agents = agents

#     def act(self, metadata: dict) -> str:
#         parts = []
#         for agent in self.agents:
#             text = agent.act(metadata)
#             if text:
#                 parts.append(text)

#         return " ".join(parts)


# class PromptComposer:
#     def __init__(self, agents):
#         self.agents = agents

#     def act(self, metadata: dict) -> str:
#         parts = []
#         for agent in self.agents:
#             text = agent.act(metadata)
#             if text:
#                 parts.append(text)
#         prompt = " ".join(parts)
#         print("@@@@@@@@@@@@@@")
#         print(prompt)
#         print("@@@@@@@@@@@@@@")
#         print(metadata)
#         print("@@@@@@@@@@@@@@")
#         print(parts)
#         # Ensure that the combined prompt is not too long for the model
#         prompt = self.shorten_prompt(prompt)
#         print("@@@@@@@@@@@@@@")
#         print(prompt)
#         return prompt

#     def shorten_prompt(self, prompt, max_length=77):
#         # tokens = prompt.split()  # Split into words
#         # if len(tokens) > max_length:
#         #     tokens = tokens[:max_length]  # Cut it to the allowed length
#         # return ' '.join(tokens)
#         return prompt[:max_length]




from .base_llm_agent import BaseLLMAgent


class PromptComposer(BaseLLMAgent):

    def __init__(self, agents, max_chars=77):
        super().__init__()
        self.agents = agents
        self.max_chars = max_chars

    def act(self, metadata: dict) -> str:
        parts = []

        for agent in self.agents:
            text = agent.act(metadata)
            if text:
                parts.append(text)

        prompt = " ".join(parts)

        print("@@@@@@@@@@@@@@ ORIGINAL")
        print(prompt)
        print("LEN:", len(prompt))

        if len(prompt) <= self.max_chars:
            return prompt

        compressed = self.compress_with_llm(prompt)

        print("@@@@@@@@@@@@@@ COMPRESSED")
        print(compressed)
        print("LEN:", len(compressed))

        return compressed

    def compress_with_llm(self, prompt: str) -> str:

        system = "You compress text for image generation prompts."

        user = f"""
Rewrite this architectural prompt to be concise but descriptive.

STRICT RULES:
- Maximum {self.max_chars} characters TOTAL
- Keep key visual architectural features
- No explanations
- One single line
- Keep meaning

TEXT:
{prompt}

Return only the compressed prompt.
"""

        result = self.generate(system, user).strip()

        # تضمین نهایی (اگر LLM رعایت نکرد)
        if len(result) > self.max_chars:
            result = result[:self.max_chars].rsplit(" ", 1)[0]

        return result
