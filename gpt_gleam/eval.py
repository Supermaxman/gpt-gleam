import argparse
from collections import defaultdict

from gpt_gleam.data import Stance, iterate_post_frame_labeled_pairs, read_jsonl
from gpt_gleam.results import TabularResultsWriter


def main(
    data_path: str,
    pred_path: str,
    frame_path: str,
    output_path: str,
    full_text: bool = False,
    debug: bool = False,
):
    with TabularResultsWriter(output_path) as writer:
        # "post_id": post.id, "f_id": frame.id, "content": content
        preds = {(p["post_id"], p["f_id"]): p["content"] for p in read_jsonl(pred_path)}
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        for post, frame, stance in iterate_post_frame_labeled_pairs(
            data_path, frame_path, skip_stances=[Stance.Not_Relevant]
        ):
            if (post.id, frame.id) not in preds:
                pred = Stance.No_Stance.value.lower()
                # TODO track missing predictions
            else:
                pred = preds[(post.id, frame.id)].lower().strip()
                if not full_text:
                    pred = pred.split("\n")[-1]

            if stance == Stance.No_Stance:
                # actual negative
                if stance.value.lower() in pred:
                    # actual negative, predicted negative
                    # no need to count true negatives
                    pass
                else:
                    # actual negative, predicted positive OR just did not predict stance
                    for s in [Stance.Accept, Stance.Reject]:
                        if s.value.lower() in pred:
                            fp[s.value] += 1
                            break
                    # ok if none of these match, same as true negative
            else:
                # actual positive
                if stance.value.lower() in pred:
                    # actual positive, predicted positive
                    tp[stance.value] += 1
                else:
                    # actual positive, predicted negative OR just did not predict stance
                    fn[stance.value] += 1

        results = {}
        for sv in [Stance.Accept, Stance.Reject]:
            s = sv.value
            # guard against divide by zeros
            p = tp[s] / max(tp[s] + fp[s], 1)
            r = tp[s] / max(tp[s] + fn[s], 1)
            denom = p + r
            num = 2 * p * r
            if denom == 0:
                f1 = 0.0
            else:
                f1 = num / denom
            results[f"{s} F1"] = f1
            results[f"{s} P"] = p
            results[f"{s} R"] = r

        final_results = {}
        final_results["Macro F1"] = sum(results[f"{s} F1"] for s in [Stance.Accept.value, Stance.Reject.value]) / 2
        final_results["Macro P"] = sum(results[f"{s} P"] for s in [Stance.Accept.value, Stance.Reject.value]) / 2
        final_results["Macro R"] = sum(results[f"{s} R"] for s in [Stance.Accept.value, Stance.Reject.value]) / 2
        # reorder to have macro in front of table
        for k, v in results.items():
            final_results[k] = v

        writer.write(final_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to data jsonl file")
    parser.add_argument("--frame_path", type=str, required=True, help="path to frames json file")
    parser.add_argument("--pred_path", type=str, help="path to prediction jsonl file")
    parser.add_argument("--output_path", type=str, required=True, help="path to output jsonl file")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--full_text", action="store_true", help="use full text instead of rationale")
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        frame_path=args.frame_path,
        pred_path=args.pred_path,
        output_path=args.output_path,
        full_text=args.full_text,
        debug=args.debug,
    )
