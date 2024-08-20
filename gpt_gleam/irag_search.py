import argparse
import os
import ujson as json

from gpt_gleam.data import TweetPreprocessConfig, preprocess_tweet, read_jsonl, batch

from open_clip import create_model_from_pretrained, get_tokenizer
import torch
from PIL import Image

from tqdm import tqdm


def main(
    top_k: int,
    data_path: str,
    demo_data_path: str,
    output_path: str,
):
    model_name = "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
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
    print(f"Loading Model: {model_name}")
    batch_size = 64
    model, preprocess = create_model_from_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)
    model.cuda()
    model.eval()

    context_length = 77

    def get_text_embeddings(text_list, progress=True):
        all_features = []
        for b in tqdm(
            batch(text_list, batch_size), total=(len(text_list) + batch_size - 1) // batch_size, disable=not progress
        ):
            texts = tokenizer(b, context_length=context_length).cuda()
            text_features = model.encode_text(texts)
            all_features.append(text_features)

        return torch.cat(all_features, dim=0)

    def get_image_embeddings(image_list, progress=True):
        # images = torch.stack([preprocess(img) for img in image_list]).cuda()
        # image_features = model.encode_image(images)
        all_features = []
        for b in tqdm(
            batch(image_list, batch_size), total=(len(image_list) + batch_size - 1) // batch_size, disable=not progress
        ):
            images = torch.stack([preprocess(img) for img in b]).cuda()
            image_features = model.encode_image(images)
            all_features.append(image_features)
        return torch.cat(all_features, dim=0)

    print(f"Loading Demo Data: {demo_data_path}")
    examples = []
    # first, load inverted predictions
    # next, load data to index
    for ex in read_jsonl(demo_data_path):
        ex_id = ex["id"]
        ex_text = ex["text"]
        ex_text = ex_text.strip().replace("\r", " ").replace("\n", " ")
        ex_text = preprocess_tweet(ex_text, preprocess_config)
        # if "images" in ex:
        # Assume multimodal data
        image_relative_path = ex["images"][0]
        data_folder = os.path.dirname(data_path)
        image_path = os.path.join(data_folder, image_relative_path)
        # image_url = encode_image_url(image_path)
        image = Image.open(image_path)

        f_demos = ex["f_demo"]
        f_demos_content = ex["f_demo_content"]
        examples.append(
            {
                "id": ex_id,
                "text": ex_text,
                "image": image,
                "f_demos": f_demos,
                "f_demos_content": f_demos_content,
                "post": ex,
                "response": f_demos_content,
            }
        )

    print(f"Building Index: {len(examples):,} examples")
    # next, build index
    with torch.inference_mode():
        text_embeddings = get_text_embeddings([ex["text"] for ex in examples])
        image_embeddings = get_image_embeddings([ex["image"] for ex in examples])
        combined_embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)

        def search_index(text: str, image_path: str, top_k: int):
            image = Image.open(image_path)
            text_features = get_text_embeddings([text], progress=False)
            image_features = get_image_embeddings([image], progress=False)
            combined_features = torch.cat([text_features, image_features], dim=1)

            similarities = torch.einsum("ij,kj->i", combined_embeddings, combined_features)
            top_k_indices = similarities.argsort(descending=True)[:top_k]
            return [examples[i] for i in top_k_indices]

        total = sum(1 for _ in read_jsonl(data_path))
        print(f"Searching Index: {total:,} to search")
        # next, search index
        with open(output_path, "w") as f:
            for ex in tqdm(read_jsonl(data_path), total=total):
                ex_id = ex["id"]
                ex_text = ex["text"]
                ex_text = ex_text.strip().replace("\r", " ").replace("\n", " ")
                ex_text = preprocess_tweet(ex_text, preprocess_config)
                # if "images" in ex:
                # Assume multimodal data
                image_relative_path = ex["images"][0]
                data_folder = os.path.dirname(data_path)
                image_path = os.path.join(data_folder, image_relative_path)

                top_k_examples = search_index(ex_text, image_path, top_k)
                ex["demonstrations"] = [
                    {
                        "post": d["post"],
                        "response": d["response"],
                    }
                    # reverse to show most similar last in chat
                    for d in reversed(top_k_examples)
                ]
                f.write(json.dumps(ex) + "\n")

        print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, required=True, help="top k examples to retrieve")
    parser.add_argument("--data_path", type=str, required=True, help="path to data jsonl file")
    parser.add_argument("--demo_data_path", type=str, required=True, help="path to demo data jsonl file")
    parser.add_argument("--output_path", type=str, required=True, help="path to output jsonl file")
    args = parser.parse_args()

    main(
        top_k=args.top_k,
        data_path=args.data_path,
        demo_data_path=args.demo_data_path,
        output_path=args.output_path,
    )
