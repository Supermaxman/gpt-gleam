import argparse

import ujson as json

from gpt_gleam.data import read_jsonl


def key_fn(frame_text):
    frame_text = frame_text.replace(" ", "")
    frame_text = frame_text.replace("\n", "")
    frame_text = frame_text.replace("\t", "")
    frame_text = frame_text.lower()
    frame_text = frame_text.strip()
    # remove all punctuation
    frame_text = "".join([c for c in frame_text if c.isalnum()])
    return frame_text


def add_pred_locations(locations: list[dict[str, str]]):
    new_locations = []
    seen_text = False
    seen_image = False
    for location in locations:
        location = location["location"]
        if location == "Text" and seen_text:
            continue
        if location == "Image" and seen_image:
            continue
        if location == "Text":
            seen_text = True
        elif location == "Image":
            seen_image = True
        else:
            continue
        new_location = {"location": location}
        new_locations.append(new_location)
    if seen_text and seen_image:
        new_locations.append({"location": "Both"})
    elif seen_text:
        new_locations.append({"location": "Text-Only"})
    elif seen_image:
        new_locations.append({"location": "Image-Only"})
    return new_locations


def main(
    pred_path: str,
    known_path: str,
    output_path: str,
):
    known_lookup = {}
    with open(known_path, "r") as f:
        known_frames = json.load(f)
    for f_id, f in known_frames.items():
        known_lookup[key_fn(f["text"])] = f_id
    unique_frames = {}
    post_count = 0
    count = 0
    known = 0
    for pred in read_jsonl(pred_path):
        post_count += 1
        pred_frames = json.loads(pred["content"])
        for f_pred in pred_frames["frames"]:
            post_id = pred["post_id"]
            count += 1
            f_key = key_fn(f_pred["frame"])
            if f_key not in unique_frames:
                known_f_id = known_lookup.get(f_key)
                if known_f_id is not None:
                    known += 1
                unique_frames[f_key] = {
                    "frame": f_pred["frame"],
                    "problems": {
                        p["problem"]: {
                            "locations": {l["location"]: [post_id] for l in add_pred_locations(p["locations"])},
                            "posts": [post_id],
                        }
                        for p in f_pred["problems"]
                    },
                    "posts": [post_id],
                    "known_id": known_f_id,
                }
            else:
                unique_frames[f_key]["posts"].append(post_id)
                for p in f_pred["problems"]:
                    if p["problem"] not in unique_frames[f_key]["problems"]:
                        unique_frames[f_key]["problems"][p["problem"]] = {
                            "locations": {l["location"]: [post_id] for l in add_pred_locations(p["locations"])},
                            "posts": [post_id],
                        }
                    else:
                        unique_frames[f_key]["problems"][p["problem"]]["posts"].append(post_id)
                        for l in add_pred_locations(p["locations"]):
                            if l["location"] not in unique_frames[f_key]["problems"][p["problem"]]["locations"]:
                                unique_frames[f_key]["problems"][p["problem"]]["locations"][l["location"]] = [post_id]
                            else:
                                unique_frames[f_key]["problems"][p["problem"]]["locations"][l["location"]].append(
                                    post_id
                                )

    print(f"Posts: {post_count:,}")
    print(f"Frames: {count:,}")
    print(f"Known Frames: {known:,}")
    frames = {
        f"F{i}": frame
        for i, (_, frame) in enumerate(
            sorted(unique_frames.items(), key=lambda x: len(x[1]["posts"]), reverse=True), start=1
        )
    }
    print(f"Unique Frames: {len(frames):,}")
    with open(output_path, "w") as f:
        json.dump(frames, f, indent=2)

    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True, help="path to data jsonl file")
    parser.add_argument("--known_path", type=str, required=True, help="path to known json file")
    parser.add_argument("--output_path", type=str, required=True, help="path to output jsonl file")
    args = parser.parse_args()

    main(
        pred_path=args.pred_path,
        known_path=args.known_path,
        output_path=args.output_path,
    )
