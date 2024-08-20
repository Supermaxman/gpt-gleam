import argparse
import ujson as json

from gpt_gleam.data import Stance, read_jsonl, load_frames


def main(
    data_path: str,
    pred_path: str,
    frame_path: str,
    output_path: str,
):
    frames = load_frames(frame_path)
    preds: dict[str, str] = {}
    for pred in read_jsonl(pred_path):
        preds[pred["id"]] = pred["content"]
    with open(output_path, "w") as f:
        for ex in read_jsonl(data_path):
            post_id = ex["id"]
            f_demos = []
            for f_id, f_stance in ex["labels"].items():
                if f_stance != Stance.Accept.value:
                    continue
                frame = frames[f_id]
                pred_id = f"{post_id}-{f_id}"
                if pred_id not in preds:
                    continue
                ex_pred_content = preds[pred_id]
                # problem_rationales:
                #  - problem: str
                #  - explanation: str
                #  - locations:
                #    - explanation: str
                #    - location: Enum["Text" | "Image"]
                #
                # : str
                p = json.loads(ex_pred_content)
                f_problem_demos = []
                for pr in p["problem_rationales"]:
                    f_p_locations = []
                    for loc in pr["locations"]:
                        f_p_locations.append(
                            {
                                "explanation": loc["explanation"],
                                "location": loc["location"],
                            }
                        )
                    f_problem_demos.append(
                        {
                            "explanation": pr["explanation"],
                            "locations": f_p_locations,
                            "problem": pr["problem"],
                        }
                    )
                f_demos.append(
                    {
                        "problems": f_problem_demos,
                        "frame_rationale": p["frame_rationale"],
                        "frame": frame.text,
                    }
                )
            if len(f_demos) == 0:
                continue
            ex["f_demo"] = {
                "frames": f_demos,
            }
            ex["f_demo_content"] = json.dumps(ex["f_demo"])
            f.write(json.dumps(ex) + "\n")

    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to data jsonl file")
    parser.add_argument("--pred_path", type=str, required=True, help="path to predictions jsonl file")
    parser.add_argument("--frame_path", type=str, required=True, help="path to frames json file")
    parser.add_argument("--output_path", type=str, required=True, help="path to output jsonl file")
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        pred_path=args.pred_path,
        frame_path=args.frame_path,
        output_path=args.output_path,
    )
