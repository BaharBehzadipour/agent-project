# class PeriodAgent:
#     def act(self, metadata: dict) -> str:
#         period = metadata.get("period", "")
#         if not period:
#             return ""
#         return (
#             f"Historical period: {period}. "
#             f"Reflect construction techniques, structural logic, and aesthetics of the {period} era."

#         )



from .base_llm_agent import BaseLLMAgent
import os
class PeriodAgent(BaseLLMAgent):

    def __init__(self):
        super().__init__()

    def act(self, metadata: dict):

        period = metadata.get("period")
        if not period:
            return ""

        system = "You are an expert architectural prompt engineer."

        user = f"""
Create a concise but descriptive prompt fragment
describing architectural historical period.

Historical period: {period}

Reflect:
- construction techniques
- structural logic
- materials
- aesthetic principles
typical of the {period} era.

Return only the prompt text.
"""

        return self.generate(system, user)
