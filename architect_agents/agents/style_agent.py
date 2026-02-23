# class StyleAgent:
#     def act(self, metadata: dict) -> str:
#         style = metadata.get("style", "")
#         if not style:
#             return ""
#         return (
#             f"Architectural style: {style}. "
#             f"Preserve characteristic forms, ornaments, façade language, and proportions of {style} architecture."

#         )

from .base_llm_agent import BaseLLMAgent


class StyleAgent(BaseLLMAgent):

    def __init__(self):
        super().__init__()

    def act(self, metadata: dict) -> str:

        style = metadata.get("style")
        if not style:
            return ""

        system = "You are an expert architectural prompt engineer."

        user = f"""
Create a concise but descriptive architectural prompt fragment.

Architectural style: {style}

Requirements:
- Preserve characteristic forms
- Preserve ornamentation language
- Preserve façade composition
- Preserve structural proportions
- Use terminology appropriate for architectural design prompts

Return only the prompt text.
Keep it compact but visually descriptive.
"""

        return self.generate(system, user)
