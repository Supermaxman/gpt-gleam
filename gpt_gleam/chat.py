import dataclasses
import time
from typing import Optional
from gpt_gleam.configuration import ChatCompletionConfig
from gpt_gleam.data import Demonstration, Frame, Post, Problem, Stance

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
                    "image_url": {
                        "url": image_url,
                        "detail": "high",
                    },
                },
            ],
        }

    def create_response_prompt(self, content: str):
        return {"role": "assistant", "content": content}

    def build_context(self):
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        return messages

    def build_prompt(
        self,
        post: Optional[Post] = None,
        frame: Optional[Frame] = None,
        stance: Optional[Stance] = None,
        problems: Optional[dict[str, Problem]] = None,
        **kwargs,
    ) -> str:
        values = {}
        if post is not None:
            values["post"] = post.text
        if frame is not None:
            values["frame"] = frame.text

            if frame.problems is not None and problems is not None:
                values["problems"] = "\n".join(
                    [f"{problems[p_id].id}: {problems[p_id].claim}" for p_id in frame.problems]
                )

        if stance is not None:
            values["stance"] = stance.value
        values = {**values, **kwargs}
        content = self.user_prompt.format(**values)
        return content

    def create_prompt(
        self,
        post: Optional[Post] = None,
        frame: Optional[Frame] = None,
        stance: Optional[Stance] = None,
        problems: Optional[dict[str, Problem]] = None,
        **kwargs,
    ):
        content = self.build_prompt(post, frame, stance, problems, **kwargs)
        if post is not None and post.image_url is not None:
            return self.create_image_prompt(content, post.image_url)
        else:
            return self.create_text_prompt(content)

    def create_context(
        self,
        post: Optional[Post] = None,
        frame: Optional[Frame] = None,
        stance: Optional[Stance] = None,
        problems: Optional[dict[str, Problem]] = None,
        demos: Optional[list[Demonstration]] = None,
        **kwargs,
    ):
        messages = self.build_context()
        if demos is not None:
            for demo in demos:
                messages.append(self.create_prompt(demo.post, demo.frame, demo.stance, demo.problems, **kwargs))
                messages.append(self.create_response_prompt(demo.response))
        messages.append(self.create_prompt(post, frame, stance, problems, **kwargs))
        return messages
