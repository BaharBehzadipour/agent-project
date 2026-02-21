class PeriodAgent:
    def act(self, metadata: dict) -> str:
        period = metadata.get("period", "")
        if not period:
            return ""
        return (
            f"Historical period: {period}. "
            f"Reflect construction techniques, structural logic, and aesthetics of the {period} era."
        )