class FunctionAgent:
    def act(self, metadata: dict) -> str:
        purpose = metadata.get("purpose", "")
        if not purpose:
            return ""
        return (
            f"Building function: {purpose}. "
            f"Ensure spatial organization, scale, and elements appropriate for a {purpose}."
        )