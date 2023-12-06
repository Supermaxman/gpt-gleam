import dataclasses


@dataclasses.dataclass
class ChatCompletionConfig:
    seed: int
    delay: int
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    system_prompt: str
    user_prompt: str
