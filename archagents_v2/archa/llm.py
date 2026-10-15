"""Optional VLM used by the agents' `decide` step (Gemini REST, image + text -> JSON).

Set GEMINI_API_KEY in the environment (never hard-code it). Without a key the agents fall back to
ArchCLIP + the consistency prior, which is also the ablation baseline.
"""
import base64
import io
import json
import os
import re

import requests


class NoLLM:
    available = False

    def choose(self, *a, **k):
        return None


class GeminiVLM:
    available = True

    def __init__(self, model=None, api_key=None, temperature=0.2, timeout=60):
        self.model = model or os.environ.get("ARCH_VLM_MODEL", "gemini-2.5-flash")
        self.key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise ValueError("set GEMINI_API_KEY (or run with --no_llm)")
        self.temperature, self.timeout = temperature, timeout
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    @staticmethod
    def _b64(img):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def ask_json(self, system, user, images=()):
        parts = [{"text": user}] + [{"inline_data": {"mime_type": "image/png", "data": self._b64(i)}}
                                    for i in images]
        payload = {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": parts}],
                   "generationConfig": {"temperature": self.temperature,
                                        "responseMimeType": "application/json"}}
        r = requests.post(self.url, headers={"x-goog-api-key": self.key, "Content-Type": "application/json"},
                          data=json.dumps(payload), timeout=self.timeout)
        r.raise_for_status()
        cands = r.json().get("candidates", [])
        if not cands:
            raise RuntimeError("no candidates from VLM")
        txt = "".join(p.get("text", "") for p in cands[0]["content"]["parts"])
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0) if m else txt)

    def choose(self, field, candidates, sketch, locked, neighbours, multi=False):
        """candidates: [(value, prob)] ; returns chosen value (or list) among them, or None."""
        system = ("You are an expert architectural historian. You look at an architect's line sketch and pick "
                  "the single best value for ONE attribute, using only the allowed candidates. "
                  "Answer with JSON only.")
        user = (f"Attribute: {field}\n"
                f"Allowed candidates with the classifier's probabilities: "
                f"{json.dumps([[v, round(float(p), 3)] for v, p in candidates])}\n"
                f"Already decided attributes (must stay consistent): {json.dumps(locked)}\n"
                f"Attributes of the 5 most similar sketches in the dataset: {json.dumps(neighbours)}\n"
                + ("Return JSON {\"values\": [up to 3 candidates], \"evidence\": \"...\"}."
                   if multi else
                   "Return JSON {\"value\": \"<one candidate>\", \"evidence\": \"<what in the sketch supports it>\"}."))
        out = self.ask_json(system, user, [sketch])
        allowed = {v for v, _ in candidates}
        if multi:
            vals = [v for v in out.get("values", []) if v in allowed]
            return vals or None
        v = out.get("value")
        return v if v in allowed else None
