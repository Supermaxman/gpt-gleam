import argparse
import os
from typing import Optional
import yaml
import ujson as json
from openai import OpenAI
from tqdm import tqdm
from gpt_gleam.chat import ChatContextCreator, chat, print_messages
from gpt_gleam.configuration import ChatCompletionConfig

from gpt_gleam.data import Stance, TweetPreprocessConfig, iterate_posts, load_frames, load_problems, preprocess_tweet
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
        # "F4": {
        # "frame": "COVID-19 vaccine is an unsafe poison, no one should take it.",
        # "problems": {
        # "Conspiracy": {
        #     "locations": {
        #     "Text": 21,
        #     "Image": 19
        #     },
        #     "count": 21
        # },
        # "Confidence": {
        #     "locations": {
        #     "Text": 15,
        #     "Image": 11
        #     },
        #     "count": 15
        # }
        # },
        new_frames = json.load(f)

    creator = ChatContextCreator(config)
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=os.getenv("OPENAI_TIMEOUT", 90),
    )
    total = len(new_frames)

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
            total_count = new_frame["count"]
            for problem, problem_data in sorted(
                new_frame["problems"].items(), key=lambda x: x[1]["count"], reverse=True
            ):
                fp_count = problem_data["count"]
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
    )
