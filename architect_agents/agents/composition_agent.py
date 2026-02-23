# class CompositionAgent:
#     def act(self, metadata: dict) -> str:
#         return (
#             "Architectural composition: balanced massing, coherent structure, strong perspective, "
#             "dramatic natural lighting, and detailed façade articulation."

#         )


from .base_llm_agent import BaseLLMAgent


class CompositionAgent(BaseLLMAgent):

    def __init__(self):
        super().__init__()

    def act(self, metadata: dict):

        system = "You are an expert architectural prompt engineer."

        user = """
Create a concise but descriptive prompt fragment
describing architectural composition for image generation.

Include concepts such as:
- balanced massing
- coherent structural hierarchy
- strong perspective
- dramatic natural lighting
- façade articulation
- spatial depth
- visual harmony

Return only the prompt text.
"""

        return self.generate(system, user)
