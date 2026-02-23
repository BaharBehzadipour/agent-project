# class CountryAgent:
#     def act(self, metadata: dict) -> str:
#         country = metadata.get("country_normalized") or metadata.get("region")
#         if not country:
#             return ""
#         return (
#             f"Geographical context: {country}. "
#             f"Include regional identity, climate adaptation, and cultural architectural motifs."

#         )

from .base_llm_agent import BaseLLMAgent


class CountryAgent(BaseLLMAgent):

    def __init__(self):
        super().__init__()

    def act(self, metadata: dict):

        country = metadata.get("country_normalized") or metadata.get("region")
        if not country:
            return ""

        system = "You are an expert architectural prompt engineer."

        user = f"""
Create a concise but descriptive prompt fragment
describing geographical architectural context.

Location: {country}

Reflect:
- regional identity
- climate adaptation
- local materials
- cultural architectural motifs
- vernacular influences

Return only the prompt text.
"""

        return self.generate(system, user)
