class PromptComposer:
    def __init__(self, agents):
        self.agents = agents

    def act(self, metadata: dict) -> str:
        parts = []
        for agent in self.agents:
            text = agent.act(metadata)
            if text:
                parts.append(text)
        return " ".join(parts)