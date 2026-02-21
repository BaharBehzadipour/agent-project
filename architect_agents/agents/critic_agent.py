import torch
from transformers import CLIPProcessor, CLIPModel

class CriticAgent:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def score(self, image, prompt: str) -> float:
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits_per_image
        score = torch.sigmoid(logits).item()
        return score