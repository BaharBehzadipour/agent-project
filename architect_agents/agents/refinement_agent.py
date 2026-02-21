class RefinementAgent:
    def refine(self, prompt: str, score: float) -> str:
        if score < 0.75:
            prompt += " Increase architectural detail and stylistic fidelity."
        if score < 0.6:
            prompt += " Stronger regional and historical characteristics."
        if score < 0.5:
            prompt += " More dramatic lighting and composition."
        return prompt