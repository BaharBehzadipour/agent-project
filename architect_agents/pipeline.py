from agents.style_agent import StyleAgent
from agents.period_agent import PeriodAgent
from agents.country_agent import CountryAgent
from agents.material_agent import MaterialAgent
from agents.function_agent import FunctionAgent
from agents.composition_agent import CompositionAgent
from agents.prompt_composer import PromptComposer
from agents.critic_agent import CriticAgent
from agents.refinement_agent import RefinementAgent
from models.controlnet_executor import ControlNetExecutor

class ArchitecturePipeline:
    def __init__(self):
        self.agents = [
            StyleAgent(),
            PeriodAgent(),
            CountryAgent(),
            # MaterialAgent(),
            FunctionAgent(),
            CompositionAgent()
        ]

        self.composer = PromptComposer(self.agents)
        self.executor = ControlNetExecutor()
        self.critic = CriticAgent()
        self.refiner = RefinementAgent()

    def run(self, metadata,  conditioning_path, iterations=3):
        prompt = self.composer.act(metadata)

        for i in range(iterations):
            image = self.executor.generate( conditioning_path, prompt)
            score = self.critic.score(image, prompt)
            print(f"Iteration {i+1} score: {score:.3f}")

            prompt = self.refiner.refine(prompt, score)

            if score > 0.9:
                break


        return image, prompt, score
