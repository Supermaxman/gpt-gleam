from tqdm import tqdm
from openai.types.chat import ChatCompletion

from gpt_gleam.costs import CompletionUsageEstimator


class ChatCompletionProgress(tqdm):
    def __init__(self, total: int, seen: int, *args, **kwargs):
        super().__init__(total=total, *args, **kwargs)
        self.chat_usage = CompletionUsageEstimator()
        self._total_chat = total
        self._seen_chat = seen

    def __enter__(self):
        self.update(self._total_chat - self._seen_chat)
        return super().__enter__()

    def update(self, completion: ChatCompletion) -> bool | None:
        self.chat_usage.update(completion)
        res = super().update(1)
        use = self.chat_usage.estimate()
        self.set_postfix(
            {
                "cost": f"${use.total_cost:.2f}",
            }
        )
        return res
