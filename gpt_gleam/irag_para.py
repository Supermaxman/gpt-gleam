import argparse
import os
from typing import Optional
import yaml
import ujson as json
from openai import OpenAI
from tqdm import tqdm
from gpt_gleam.chat import ChatContextCreator, chat, print_messages
from gpt_gleam.configuration import ChatCompletionConfig

from gpt_gleam.data import (
    Frame,
    Stance,
    TweetPreprocessConfig,
    iterate_posts,
    load_frames,
    load_problems,
    preprocess_tweet,
)
from gpt_gleam.predictions import JsonlPredictionsWriter
from gpt_gleam.progress import ChatCompletionProgress


def main(
    config: ChatCompletionConfig,
    pred_path: str,
    known_path: str,
    problem_path: str,
    output_path: str,
    total: Optional[int] = None,
    debug: bool = False,
    keep_known: bool = False,
):
    preprocess_config = TweetPreprocessConfig(
        do_lower_case=False,
        replace_usernames=False,
        replace_urls=True,
        asciify_emojis=False,
        replace_multiple_usernames=False,
        replace_multiple_urls=False,
        standardize_punctuation=True,
        remove_unicode_symbols=False,
        remove_accented_characters=False,
    )
    problems = load_problems(problem_path, preprocess_config)
    problems_text = "\n".join([f"{k}: {v.claim}" for k, v in problems.items()])
    known_frames = load_frames(known_path, preprocess_config)
    with open(pred_path, "r") as f:
        all_new_frames = json.load(f)

    new_frames = {f_key: f_data for f_key, f_data in all_new_frames.items() if f_data["known_id"] is None}

    creator = ChatContextCreator(config)
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=os.getenv("OPENAI_TIMEOUT", 90),
    )
    total = len(new_frames)
    next_frame_id = max(int(k[1:]) for k in known_frames.keys()) + 1 if known_frames else 1

    with (
        JsonlPredictionsWriter(output_path) as preds,
        ChatCompletionProgress(total=total, seen=len(preds), disable=debug) as bar,
    ):
        for new_frame_id, new_frame in new_frames.items():
            ex_id = f"{new_frame_id}"
            if ex_id in preds:
                continue
            f_text = preprocess_tweet(new_frame["frame"], preprocess_config)
            f_problems = []
            total_count = len(new_frame["posts"])
            for problem, problem_data in sorted(
                new_frame["problems"].items(), key=lambda x: len(x[1]["posts"]), reverse=True
            ):
                fp_count = len(problem_data["posts"])
                f_problems.append(f"{problem}: {100 * fp_count / total_count:.0f}%")
            f_problems_text = "\n".join(f_problems)
            known_frame_texts = []
            # TODO determine if I want to keep adding to known_frames
            for f_id, f in known_frames.items():
                kfp_problems_text = ", ".join(f.problems)
                known_frame_texts.append(f"{f_id}: {f.text} ({kfp_problems_text})")
            known_text = "\n".join(known_frame_texts)
            messages = creator.create_context(
                problem_definitions=problems_text,
                known_frames=known_text,
                novel_frame=f_text,
                novel_problems=f_problems_text,
            )
            completion = chat(
                client,
                delay=config.delay,
                model=config.model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=config.seed,
                response_format=config.response_format,
            )
            if completion is None:
                print(f"Skipping example due to API safety error: {new_frame_id}")
                continue
            content = completion.choices[0].message.content
            preds.add({"id": ex_id, "new_frame_id": new_frame_id, "content": content})
            messages.append({"role": "assistant", "content": content})
            if debug:
                print_messages(messages)
            try:
                p = json.loads(content)
            except json.JSONDecodeError:
                print(f"Failed to parse completion for {new_frame_id}")
                continue
            known_frame_id = p["frame_id"]
            if known_frame_id is None and keep_known:
                problems = []
                for problem, problem_data in sorted(
                    new_frame["problems"].items(), key=lambda x: len(x[1]["posts"]), reverse=True
                ):
                    fp_count = len(problem_data["posts"])
                    fp_percent = fp_count / total_count
                    # must be a problem if it's more than 50% of the time
                    if fp_percent > 0.5:
                        problems.append(problem)
                known_frames[f"F{next_frame_id}"] = Frame(
                    id=f"F{next_frame_id}",
                    text=f_text,
                    problems=problems,
                )
                next_frame_id += 1

            bar.update(completion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--pred_path", type=str, required=True, help="path to preds json file")
    parser.add_argument("--known_path", type=str, required=True, help="path to known json file")
    parser.add_argument("--problem_path", type=str, required=True, help="path to problems json file")
    parser.add_argument("--output_path", type=str, required=True, help="path to output jsonl file")
    parser.add_argument("--total", type=int, help="total number of examples to process")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--keep_known", action="store_true", help="keep known frames", default=False)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = ChatCompletionConfig(**config)
    main(
        config=config,
        pred_path=args.pred_path,
        known_path=args.known_path,
        problem_path=args.problem_path,
        output_path=args.output_path,
        total=args.total,
        debug=args.debug,
        keep_known=args.keep_known,
    )
