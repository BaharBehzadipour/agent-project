class StyleAgent:
    def act(self, metadata: dict) -> str:
        style = metadata.get("style", "")
        if not style:
            return ""
        return (
            f"Architectural style: {style}. "
            f"Preserve characteristic forms, ornaments, façade language, and proportions of {style} architecture."
        )