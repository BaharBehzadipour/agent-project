class MaterialAgent:
    def act(self, metadata: dict) -> str:
        materials = metadata.get("primary_materials", "")
        if not materials:
            return ""
        return (
            f"Primary materials: {materials}. "
            f"Emphasize realistic texture, structural expression, and weathering of {materials}."
        )