import time
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

class GenerationTimeout(Exception):
    """Raised when image generation exceeds its allowed time budget."""


class ControlNetExecutor:
    def __init__(
        self,
        controlnet_path="/content/drive/MyDrive/epoch 50000",
        base_model="runwayml/stable-diffusion-v1-5"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path,
                                                         torch_dtype=self.dtype).to(self.device)

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model,
            controlnet=self.controlnet,
            torch_dtype=self.dtype
        ).to(self.device)

    def generate(self, conditioning_path: str, prompt: str, deadline=None):
        """
        deadline: absolute time.monotonic() value. If it passes while the
        diffusion loop is running, GenerationTimeout is raised and the
        generation is really aborted (checked after every denoising step).
        """
        cond = Image.open(conditioning_path).convert("RGB")

        def _check_deadline(pipe, step, timestep, callback_kwargs):
            if deadline is not None and time.monotonic() > deadline:
                raise GenerationTimeout(f"aborted at denoising step {step}")
            return callback_kwargs

        image = self.pipe(
            prompt=prompt,
            image=cond,
            num_inference_steps=30,
            callback_on_step_end=_check_deadline,
            # guidance_scale=7.5
        ).images[0]

        return image
