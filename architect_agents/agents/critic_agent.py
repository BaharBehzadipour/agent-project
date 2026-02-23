# import torch
# from transformers import CLIPProcessor, CLIPModel

# class CriticAgent:
#     def __init__(self, model_name="openai/clip-vit-large-patch14"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = CLIPModel.from_pretrained(model_name).to(self.device)
#         self.processor = CLIPProcessor.from_pretrained(model_name)

#     def score(self, image, prompt: str) -> float:
#         inputs = self.processor(text=[prompt], images=image, return_tensors="pt").to(self.device)
#         outputs = self.model(**inputs)
#         logits = outputs.logits_per_image
#         score = torch.sigmoid(logits).item()

#         return score



import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel


class CriticAgent:

    def __init__(
        self,
        model_name="openai/clip-vit-large-patch14",
        w_prompt=0.4,
        w_sketch=0.3,
        w_metadata=0.2,
        w_aesthetic=0.1
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # weights for final score
        self.w_prompt = w_prompt
        self.w_sketch = w_sketch
        self.w_metadata = w_metadata
        self.w_aesthetic = w_aesthetic

    # --------------------------------------------------
    # utilities
    # --------------------------------------------------

    def _image_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        return F.normalize(emb, dim=-1)

    def _text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        emb = self.model.get_text_features(**inputs)
        return F.normalize(emb, dim=-1)

    def _cosine(self, a, b):
        return torch.matmul(a, b.T).item()

    # --------------------------------------------------
    # scoring components
    # --------------------------------------------------

    def score_prompt_alignment(self, image, prompt):
        img_emb = self._image_embedding(image)
        txt_emb = self._text_embedding(prompt)
        return self._cosine(img_emb, txt_emb)

    def score_sketch_alignment(self, image, sketch):
        if sketch is None:
            return 0.0
        img_emb = self._image_embedding(image)
        sketch_emb = self._image_embedding(sketch)
        return self._cosine(img_emb, sketch_emb)

    def score_metadata_alignment(self, image, metadata: dict):
        if not metadata:
            return 0.0

        meta_text = ", ".join(str(v) for v in metadata.values() if v)
        img_emb = self._image_embedding(image)
        txt_emb = self._text_embedding(meta_text)

        return self._cosine(img_emb, txt_emb)

    # def score_aesthetic(self, image):
    #     """
    #     نسخه ساده:
    #     sharpness / contrast proxy
    #     می‌تونی بعداً مدل aesthetic predictor بذاری
    #     """
    #     img = torch.tensor(image).float()
    #     return float(img.std() / 255.0)

    def score_aesthetic(self, image):
    
        import torchvision.transforms as T
    
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
    
        img = transform(image).unsqueeze(0).to(self.device)
    
        # اگر مدل aesthetic داری اینجا پاس بده
        score = self.aesthetic_model(img)
    
        return score.item()

    # --------------------------------------------------
    # final score
    # --------------------------------------------------

    def score(
        self,
        image,
        prompt: str,
        sketch=None,
        metadata=None
    ):
        s_prompt = self.score_prompt_alignment(image, prompt)
        s_sketch = self.score_sketch_alignment(image, sketch)
        s_meta = self.score_metadata_alignment(image, metadata)
        s_aes = self.score_aesthetic(image)

        final = (
            self.w_prompt * s_prompt +
            self.w_sketch * s_sketch +
            self.w_metadata * s_meta +
            self.w_aesthetic * s_aes
        )

        # return {
        #     "final": final,
        #     "prompt": s_prompt,
        #     "sketch": s_sketch,
        #     "metadata": s_meta,
        #     "aesthetic": s_aes
        # }
        return final

