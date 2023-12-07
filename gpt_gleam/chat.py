from abc import ABC, abstractmethod
import time
from gpt_gleam.configuration import ChatCompletionConfig
from gpt_gleam.data import Frame, Post, Stance

from openai import OpenAI, BadRequestError
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential


class MinimumDelay:
    def __init__(self, delay: float | int):
        self.delay = delay
        self.start = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        seconds = end - self.start
        if self.delay > seconds:
            time.sleep(self.delay - seconds)


@retry(wait=wait_random_exponential(min=1, max=90), stop=stop_after_attempt(3))
def chat(client: OpenAI, delay: float | int, **kwargs) -> ChatCompletion | None:
    try:
        with MinimumDelay(delay):
            return client.chat.completions.create(**kwargs)
    except BadRequestError as e:
        print(f"Bad Request: {e}")
        if "safety" in e.message:
            return None
        raise e
    except Exception as e:
        print(f"Exception: {e}")
        raise e


def print_messages(messages):
    for message in messages:
        if isinstance(message["content"], list):
            print(f"{message['role']}:")
            for content in message["content"]:
                if content["type"] == "text":
                    print(content["text"])
                elif content["type"] == "image_url":
                    print("[IMAGE]")
        else:
            print(f"{message['role']}: {message['content']}")
        print()
    print("=========================================")


class ChatContextCreator:
    def __init__(self, config: ChatCompletionConfig):
        self.config = config
        self.system_prompt = self.config.system_prompt.strip()
        self.user_prompt = self.config.user_prompt.strip()

    def create_text_prompt(self, content: str):
        return {"role": "user", "content": content}

    def create_image_prompt(self, content: str, image_url: str):
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": content},
                {
                    "type": "image_url",
                    "image_url": image_url,
                },
            ],
        }

    def build_context(self):
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        return messages

    def build_prompt(self, post: Post, frame: Frame, stance: Stance) -> str:
        content = self.user_prompt.format(
            post=post.text,
            frame=frame.text,
            stance=stance.value,
        )
        return content

    def create_prompt(self, post: Post, frame: Frame, stance: Stance):
        content = self.build_prompt(post, frame, stance)
        if post.image_url is None:
            return self.create_text_prompt(content)
        else:
            return self.create_image_prompt(content, post.image_url)

    # def create_demo(self, ex: Post):
    #     if ex.demonstrations is None:
    #         raise ValueError(f"Example {ex.id} has no demonstrations")
    #     messages = []
    #     for f_id, reason in ex.demonstrations.items():
    #         frame = self.frames[f_id]
    #         messages.append(self.create_prompt(ex, frame))
    #         messages.append({"role": "assistant", "content": reason})
    #     return messages

    def create_context(self, post: Post, frame: Frame, stance: Stance):
        messages = self.build_context()
        # demos = self.find_demos(*args, **kwargs)
        # for demo in demos:
        #     messages.extend(self.create_demo(demo))
        messages.append(self.create_prompt(post, frame, stance))
        return messages
