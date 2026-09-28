from agents.style_agent import StyleAgent
from agents.period_agent import PeriodAgent
from agents.country_agent import CountryAgent
from agents.material_agent import MaterialAgent
from agents.function_agent import FunctionAgent
from agents.composition_agent import CompositionAgent
from agents.prompt_composer import PromptComposer
from agents.critic_agent import CriticAgent
from agents.refinement_agent import RefinementAgent
from models.controlnet_executor import ControlNetExecutor, GenerationTimeout
import time
from PIL import Image

class ArchitecturePipeline:
    def __init__(self):
        self.agents = [
            StyleAgent(),
            PeriodAgent(),
            CountryAgent(),
            # MaterialAgent(),
            # FunctionAgent(),
            CompositionAgent()
        ]

        self.composer = PromptComposer(self.agents)
        self.executor = ControlNetExecutor()
        self.critic = CriticAgent()
        self.refiner = RefinementAgent()

    def run(self, metadata, conditioning_path, iterations=3, timeout_seconds=None):
        """
        timeout_seconds: max total time for this sample (all iterations).
        On timeout, GenerationTimeout is raised (generation is truly aborted).
        """
        deadline = (time.monotonic() + timeout_seconds) if timeout_seconds else None

        def check_deadline(where):
            if deadline is not None and time.monotonic() > deadline:
                raise GenerationTimeout(f"timeout ({where})")

        prompt = self.composer.act(metadata)
        check_deadline("after prompt composition")

        best_image = None
        best_prompt = None
        best_score = float("-inf")

        for i in range(iterations):
            try:
                image = self.executor.generate(conditioning_path, prompt, deadline=deadline)
                sketch = Image.open(conditioning_path).convert("RGB")
                score = self.critic.score(
                    image=image,
                    prompt=prompt,
                    sketch=sketch,
                    metadata=metadata
                )
                print(f"Iteration {i+1} score: {score:.3f}")

                if score > best_score:
                    best_score = score
                    best_image = image
                    best_prompt = prompt
                    print(f"  -> new best (score={best_score:.3f})")

                if score > 0.9:
                    break

                check_deadline(f"after iteration {i+1}")
                prompt = self.refiner.refine(prompt, score)

            except Exception as e:
                # If at least one iteration finished, keep the best result so far
                if best_image is not None:
                    print(
                        f"⚠️ Iteration {i+1} failed ({type(e).__name__}: {e}). "
                        f"Returning best so far (score={best_score:.3f})."
                    )
                    break
                # nothing to salvage -> let the caller handle it
                raise

        return best_image, best_prompt, best_score
