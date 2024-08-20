from collections import defaultdict
import dataclasses
import html
import os
import re
from typing import Iterator, Optional
import unicodedata
import base64

import emoji
import ujson as json
import unidecode
from enum import Enum


# compile regexes
username_regex = re.compile(r"(^|[^@\w])@(\w{1,15})\b")
url_regex = re.compile(r"((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))")
control_char_regex = re.compile(r"[\r\n\t]+")
# translate table for punctuation
transl_table = dict([(ord(x), ord(y)) for x, y in zip("‘’´“”–-", "'''\"\"--")])


@dataclasses.dataclass
class TweetPreprocessConfig:
    do_lower_case: bool = True
    url_filler: str = "[URL]"
    username_filler: str = "[USER]"
    replace_usernames: bool = True
    replace_urls: bool = True
    asciify_emojis: bool = True
    replace_multiple_usernames: bool = True
    replace_multiple_urls: bool = True
    standardize_punctuation: bool = True
    remove_unicode_symbols: bool = True
    remove_accented_characters: bool = False


def preprocess_tweet(text: str, args: TweetPreprocessConfig):
    """Preprocesses tweet."""
    # standardize
    text = standardize_text(text)
    # replace usernames/urls
    if args.replace_usernames:
        text = replace_usernames(text, filler=args.username_filler)
    if args.replace_urls:
        text = replace_urls(text, filler=args.url_filler)
    if args.asciify_emojis:
        text = asciify_emojis(text)
    if args.standardize_punctuation:
        text = standardize_punctuation(text)
    if args.do_lower_case:
        text = text.lower()
    if args.replace_multiple_usernames:
        text = replace_multi_occurrences(text, args.username_filler)
    if args.replace_multiple_urls:
        text = replace_multi_occurrences(text, args.url_filler)
    if args.remove_unicode_symbols:
        text = remove_unicode_symbols(text)
    if args.remove_accented_characters:
        text = remove_accented_characters(text)
    return text


def remove_accented_characters(text):
    text = unidecode.unidecode(text)
    return text


def remove_unicode_symbols(text):
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "So")
    return text


def replace_multi_occurrences(text, filler):
    """Replaces multiple occurrences of filler with n filler."""
    # only run if we have multiple occurrences of filler
    if text.count(filler) <= 1:
        return text
    # pad fillers with whitespace
    text = text.replace(f"{filler}", f" {filler} ")
    # remove introduced duplicate whitespaces
    text = " ".join(text.split())
    # find indices of occurrences
    indices = []
    for m in re.finditer(r"{}".format(filler), text):
        index = m.start()
        indices.append(index)
    # collect merge list
    merge_list = []
    old_index = None
    for i, index in enumerate(indices):
        if i > 0 and index - old_index == len(filler) + 1:
            # found two consecutive fillers
            if len(merge_list) > 0 and merge_list[-1][1] == old_index:
                # extend previous item
                merge_list[-1][1] = index
                merge_list[-1][2] += 1
            else:
                # create new item
                merge_list.append([old_index, index, 2])
        old_index = index
    # merge occurrences
    if len(merge_list) > 0:
        new_text = ""
        pos = 0
        for start, end, count in merge_list:
            new_text += text[pos:start]
            new_text += f"{count} {filler}"
            pos = end + len(filler)
        new_text += text[pos:]
        text = new_text
    return text


def asciify_emojis(text):
    """Converts emojis into text aliases.

    E.g. 👍 becomes :thumbs_up: For a full list of text aliases see: https://www.webfx.com/tools/emoji-cheat-sheet/
    """
    text = emoji.demojize(text)
    return text


def standardize_text(text):
    """1) Escape HTML 2) Replaces some non-standard punctuation with standard versions.

    3) Replace \r, \n and \t with white spaces 4) Removes all other control characters and the NULL byte 5) Removes
    duplicate white spaces
    """
    # escape HTML symbols
    text = html.unescape(text)
    # standardize punctuation
    text = text.translate(transl_table)
    text = text.replace("…", "...")
    # replace \t, \n and \r characters by a whitespace
    text = re.sub(control_char_regex, " ", text)
    # remove all remaining control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    # replace multiple spaces with single space
    text = " ".join(text.split())
    return text.strip()


def standardize_punctuation(text):
    return "".join([unidecode.unidecode(t) if unicodedata.category(t)[0] == "P" else t for t in text])


def replace_usernames(text, filler="user"):
    # @<user> is a marker used internally. use filler instead
    text = text.replace("@<user>", f"{filler}")
    # replace other user handles by filler
    text = re.sub(username_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f" {filler} ")
    text = " ".join(text.split())
    return text


def replace_urls(text, filler="url"):
    # <url> is a marker used internally. use filler instead
    text = text.replace("<url>", filler)
    # replace other urls by filler
    text = re.sub(url_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f" {filler} ")
    text = " ".join(text.split())
    return text


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex


def write_jsonl(path, data):
    with open(path, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image_url(image_path):
    if image_path.endswith(".png"):
        image_type = "png"
    elif image_path.endswith(".jpg"):
        image_type = "jpeg"
    else:
        raise ValueError(f"Image type not supported: {image_path}")

    base64_image = encode_image(image_path)
    return f"data:image/{image_type};base64,{base64_image}"


def batch(iterable, n=1):
    """Yield successive n-sized batches from an iterable."""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


class Stance(str, Enum):
    Accept = "Accept"
    Reject = "Reject"
    No_Stance = "No Stance"
    Not_Relevant = "Not Relevant"


STANCE_ORDER = [Stance.Accept, Stance.Reject, Stance.No_Stance, Stance.Not_Relevant]


@dataclasses.dataclass
class Frame:
    id: str
    text: str
    problems: Optional[list[str]]


@dataclasses.dataclass
class Problem:
    id: str
    claim: str
    counter_claim: str


@dataclasses.dataclass
class StanceCounterFactual:
    stance: Stance
    rationale: str


@dataclasses.dataclass
class Post:
    id: str
    text: str
    image_url: Optional[str] = None
    labels: Optional[dict[str, Stance]] = None
    demonstrations: Optional[list["Demonstration"]] = None
    cfacts: Optional[dict[str, list[StanceCounterFactual]]] = None


@dataclasses.dataclass
class Demonstration:
    post: Post
    response: str
    frame: Optional[Frame] = None
    stance: Optional[Stance] = None
    problems: Optional[dict[str, Problem]] = None


def load_frames(frame_path: str, preprocess_config: Optional[TweetPreprocessConfig] = None) -> dict[str, Frame]:
    if preprocess_config is None:
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
    with open(frame_path, "r") as f:
        frames = json.load(f)

    frame_objs = {}
    for f_id, frame in frames.items():
        frame["text"] = preprocess_tweet(frame["text"], preprocess_config)
        frame_objs[f_id] = Frame(
            id=f_id, text=frame["text"], problems=frame["problems"] if "problems" in frame else None
        )

    return frame_objs


def load_problems(problem_path: str, preprocess_config: Optional[TweetPreprocessConfig] = None) -> dict[str, Problem]:
    if preprocess_config is None:
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
    with open(problem_path, "r") as f:
        problems = json.load(f)

    problem_objs = {}
    for p_id, problem in problems.items():
        problem["claim"] = preprocess_tweet(problem["claim"], preprocess_config)
        problem["counter_claim"] = preprocess_tweet(problem["counter_claim"], preprocess_config)
        problem_objs[p_id] = Problem(id=p_id, claim=problem["claim"], counter_claim=problem["counter_claim"])

    return problem_objs


def load_post(
    ex: dict,
    data_path: str,
    preprocess_config: TweetPreprocessConfig,
    cfacts: Optional[dict] = None,
    demo_data_path: Optional[str] = None,
) -> Post:
    ex_id = ex["id"]
    ex_text = ex["text"]
    ex_text = ex_text.strip().replace("\r", " ").replace("\n", " ")
    ex_text = preprocess_tweet(ex_text, preprocess_config)
    result = {"id": ex_id, "text": ex_text}
    if "images" in ex:
        image_relative_path = ex["images"][0]
        data_folder = os.path.dirname(data_path)
        image_path = os.path.join(data_folder, image_relative_path)
        image_url = encode_image_url(image_path)
        result["image_url"] = image_url
    if "labels" in ex:
        result["labels"] = {f_id: Stance[stance.replace(" ", "_")] for f_id, stance in ex["labels"].items()}
    if "demonstrations" in ex:
        demonstrations = ex["demonstrations"]
        result["demonstrations"] = [
            Demonstration(
                post=load_post(d["post"], demo_data_path, preprocess_config, cfacts=cfacts), response=d["response"]
            )
            for d in demonstrations
        ]
    if cfacts is not None and ex_id in cfacts:
        # f_id -> list[CounterFactual]
        ex_cfacts = cfacts[ex_id]
        result["cfacts"] = ex_cfacts
    result = Post(**result)
    return result


def iterate_posts(
    data_path: str,
    preprocess_config: Optional[TweetPreprocessConfig] = None,
    cfact_path: Optional[str] = None,
    demo_data_path: Optional[str] = None,
) -> Iterator[Post]:
    if preprocess_config is None:
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

    cfacts = defaultdict(lambda: defaultdict(list))
    if cfact_path is not None:
        for cf in read_jsonl(cfact_path):
            post_id = cf["post_id"]
            f_id = cf["f_id"]
            stance = cf["stance"]
            content = cf["content"]
            stance = Stance[stance.replace(" ", "_")]
            cfacts[post_id][f_id].append(StanceCounterFactual(stance=stance, rationale=content))

        for post_id, pf_cfacts in cfacts.items():
            for f_id, f_cfacts in pf_cfacts.items():
                f_cfacts.sort(key=lambda x: STANCE_ORDER.index(x.stance))

    for ex in read_jsonl(data_path):
        post = load_post(ex, data_path, preprocess_config, cfacts=cfacts, demo_data_path=demo_data_path)
        yield post


def iterate_post_frame_labeled_pairs(
    data_path: str,
    frame_path: str,
    preprocess_config: Optional[TweetPreprocessConfig] = None,
    skip_stances: Optional[list[Stance]] = None,
    cfact_path: Optional[str] = None,
):
    if preprocess_config is None:
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
    if skip_stances is None:
        skip_stances = []
    frames = load_frames(frame_path, preprocess_config)
    for post in iterate_posts(data_path, preprocess_config, cfact_path):
        if post.labels is None:
            continue
        for f_id, stance in post.labels.items():
            if stance in skip_stances:
                continue
            frame = frames[f_id]
            yield post, frame, stance


def iterate_post_frame_problem_labeled_pairs(
    data_path: str,
    frame_path: str,
    problem_path: str,
    preprocess_config: Optional[TweetPreprocessConfig] = None,
    skip_stances: Optional[list[Stance]] = None,
    cfact_path: Optional[str] = None,
):
    problems = load_problems(problem_path, preprocess_config)
    for post, frame, stance in iterate_post_frame_labeled_pairs(
        data_path, frame_path, preprocess_config=preprocess_config, skip_stances=skip_stances, cfact_path=cfact_path
    ):
        if frame.problems is not None and len(frame.problems) > 0:
            for p_id in frame.problems:
                problem = problems[p_id]
                yield post, frame, stance, problem


def iterate_post_frame_problems_labeled_pairs(
    data_path: str,
    frame_path: str,
    problem_path: str,
    preprocess_config: Optional[TweetPreprocessConfig] = None,
    skip_stances: Optional[list[Stance]] = None,
    cfact_path: Optional[str] = None,
):
    problems = load_problems(problem_path, preprocess_config)
    for post, frame, stance in iterate_post_frame_labeled_pairs(
        data_path, frame_path, preprocess_config=preprocess_config, skip_stances=skip_stances, cfact_path=cfact_path
    ):
        ex_problems: dict[str, Problem] = {}
        if frame.problems is not None and len(frame.problems) > 0:
            for p_id in frame.problems:
                problem = problems[p_id]
                ex_problems[p_id] = problem
        yield post, frame, stance, ex_problems


def iterate_post_problem_labeled_pairs(
    data_path: str,
    frame_path: str,
    problem_path: str,
    preprocess_config: Optional[TweetPreprocessConfig] = None,
    skip_stances: Optional[list[Stance]] = None,
    cfact_path: Optional[str] = None,
):
    seen_pairs = set()
    for post, _, _, problem in iterate_post_frame_problem_labeled_pairs(
        data_path,
        frame_path,
        problem_path,
        preprocess_config=preprocess_config,
        skip_stances=skip_stances,
        cfact_path=cfact_path,
    ):
        pair_id = f"{post.id}-{problem.id}"
        if pair_id in seen_pairs:
            continue
        seen_pairs.add(pair_id)
        yield post, problem


def iterate_post_frame_unlabeled_pairs(
    data_path: str,
    frame_path: str,
    preprocess_config: Optional[TweetPreprocessConfig] = None,
    skip_stances: Optional[list[Stance]] = None,
):
    if preprocess_config is None:
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
    if skip_stances is None:
        skip_stances = []
    frames = load_frames(frame_path, preprocess_config)
    for post in iterate_posts(data_path, preprocess_config):
        if post.labels is None:
            continue
        for f_id, f_stance in post.labels.items():
            if f_stance in skip_stances:
                continue
            frame = frames[f_id]
            for stance in Stance:
                if stance in skip_stances:
                    continue
                yield post, frame, stance


def load_demos(demo_path: str):
    demos: dict[str, str] = {}
    for d in read_jsonl(demo_path):
        ex_id = d["id"]
        demos[ex_id] = d["content"]
    return demos


def load_demos(
    demo_data_path: str,
    demo_path: str,
    frame_path: str,
    problem_path: str,
    preprocess_config: Optional[TweetPreprocessConfig] = None,
    skip_stances: Optional[list[Stance]] = None,
    cfact_path: Optional[str] = None,
):
    problems = load_problems(problem_path, preprocess_config)
    for post, frame, stance in iterate_post_frame_labeled_pairs(
        demo_data_path,
        frame_path,
        preprocess_config=preprocess_config,
        skip_stances=skip_stances,
        cfact_path=cfact_path,
    ):
        ex_problems: dict[str, Problem] = {}
        if frame.problems is not None and len(frame.problems) > 0:
            for p_id in frame.problems:
                problem = problems[p_id]
                ex_problems[p_id] = problem
        yield post, frame, stance, ex_problems
