# class PromptComposer:
#     def __init__(self, agents):
#         self.agents = agents

#     def act(self, metadata: dict) -> str:
#         parts = []
#         for agent in self.agents:
#             text = agent.act(metadata)
#             if text:
#                 parts.append(text)

#         return " ".join(parts)


class PromptComposer:
    def __init__(self, agents):
        self.agents = agents

    def act(self, metadata: dict) -> str:
        parts = []
        for agent in self.agents:
            text = agent.act(metadata)
            if text:
                parts.append(text)
        prompt = " ".join(parts)
        print("@@@@@@@@@@@@@@")
        print(prompt)
        print("@@@@@@@@@@@@@@")
        print(metadata)
        print("@@@@@@@@@@@@@@")
        print(parts)
        # Ensure that the combined prompt is not too long for the model
        prompt = self.shorten_prompt(prompt)
        print("@@@@@@@@@@@@@@")
        print(prompt)
        return prompt

    def shorten_prompt(self, prompt, max_length=77):
        # tokens = prompt.split()  # Split into words
        # if len(tokens) > max_length:
        #     tokens = tokens[:max_length]  # Cut it to the allowed length
        # return ' '.join(tokens)
        return prompt[:max_length]
