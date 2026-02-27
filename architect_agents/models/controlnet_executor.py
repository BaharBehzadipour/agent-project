import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

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

    def generate(self,conditioning_path: str, prompt: str):
        cond = Image.open(conditioning_path).convert("RGB")

        image = self.pipe(
            prompt=prompt,
            image=cond,
            num_inference_steps=30
            # guidance_scale=7.5
        ).images[0]


        return image



