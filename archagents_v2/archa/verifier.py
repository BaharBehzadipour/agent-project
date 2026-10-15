"""Per-attribute verifier built on the PHOTO-domain ArchCLIP heads."""
import numpy as np

from . import archclip as A


class Verifier:
    def __init__(self, archclip_photo_dir, device=None):
        import torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = A.ArchCLIP.load(archclip_photo_dir, device)
        self.maps = self.model.maps

    def probs(self, image):
        return {h: v[0] for h, v in self.model.predict([image], domain="photo").items()}

    def score(self, image, spec, fields=None):
        """probability the image shows the spec'd value, per field (only fields present in spec)."""
        p = self.probs(image)
        out = {}
        for h in (fields or spec):
            val = spec.get(h)
            if not val or h not in p:
                continue
            if h in A.MULTI:
                idx = [self.maps[h][v] for v in val if v in self.maps[h]]
                if idx:
                    out[h] = float(np.mean(p[h][idx]))
            elif val in self.maps[h]:
                out[h] = float(p[h][self.maps[h][val]])
        return out
