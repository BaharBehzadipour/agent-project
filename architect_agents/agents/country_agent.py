class CountryAgent:
    def act(self, metadata: dict) -> str:
        country = metadata.get("country_normalized") or metadata.get("region")
        if not country:
            return ""
        return (
            f"Geographical context: {country}. "
            f"Include regional identity, climate adaptation, and cultural architectural motifs."
        )