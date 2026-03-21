# Copyright (c) 2025 Jun Zhang. Licensed under the BSD-3-Clause License.
# Original code copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import os

import nncore
import torch
from nncore.engine import set_random_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.utils import GroundingDataset
from timelens.dataset.timelens_data import DATASET_DICT
from timelens.utils import extract_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True, help="Output prediction path")
    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument("--min_tokens", type=int, default=16)
    parser.add_argument("--total_tokens", type=int, default=3584)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max tokens to generate for each sample. Lower is faster.",
    )

    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--chunk",
        type=int,
        default=1,
        help="Number of chunks to split the dataset for distributed evaluation. Default is 1.",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default is 42.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If > 0, only evaluate the first N samples (after chunking). Useful for quick sanity checks.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    args.seed = set_random_seed(args.seed)
    print(f"Setting random seed to {args.seed}")

    pred_path = f"{args.pred_path}_{args.index}.jsonl"

    # Ensure output directory exists (pred_path may include nested dirs)
    out_dir = os.path.dirname(os.path.abspath(pred_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(
        f"Dataset: {args.dataset}({args.split}) | Chunk: {args.chunk} | "
        f"Index: {args.index} | Output Path: {pred_path}"
    )

    assert args.device == "auto", (
        'Device should be set to "auto" for multi-GPU evaluation.'
    )

    # Load model
    # NOTE:
    # - `transformers.from_pretrained` uses `torch_dtype`, not `dtype`.
    # - Qwen models need `trust_remote_code=True`.
    # - If args.model_path points to a PEFT adapter dir, try loading base + adapter.
    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(args.model_path)
        base_model = AutoModelForImageTextToText.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=args.device,
            trust_remote_code=True,
        ).eval()
        model = PeftModel.from_pretrained(base_model, args.model_path).eval()
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=args.device,
            trust_remote_code=True,
        ).eval()

    # Load processor (model-specific)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        padding_side="left",
        do_resize=False,  # For Video Processing, we do not need to resize the video frames again in the processor
        trust_remote_code=True,
    )
    # Load dataset
    dataset_class = DATASET_DICT[args.dataset]
    annos = dataset_class.load_annos(split=args.split)

    # Sort by video length in descending order
    # 1. balance the video length for each GPU
    # 2. long videos are more likely to cause OOM, so we put them first
    annos.sort(key=lambda x: x["duration"], reverse=True)
    annos = annos[args.index :: args.chunk]
    if args.max_samples and args.max_samples > 0:
        annos = annos[: args.max_samples]

    dataset = GroundingDataset(annos, processor, args)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=10,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=lambda x: x[0],
    )

    dumps = []
    for data in nncore.ProgressBar(data_loader):
        inputs = data["inputs"].to("cuda", non_blocking=True)
        anno = data["anno"]

        video_path = anno["video_path"]
        query = anno["query"]
        duration = anno["duration"]
        span = anno["span"]  # ground truth time span

        output_ids = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=args.max_new_tokens,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        answers = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answers = answers[0]

        # Parse the answer
        timestamps = extract_time(answers)
        if len(timestamps) != 0:
            print(f"Extracted timestamps: {timestamps}")
        else:
            print("No timestamps extracted, answer might be invalid. Answer:", answers)
            timestamps = [[duration + 10, duration + 20]]

        # Round timestamps to units
        unit = getattr(dataset_class, "UNIT", 1.0)
        timestamps = [
            [
                round(start / unit) * unit,
                round(end / unit) * unit,
            ]
            for start, end in timestamps
        ]

        # Save the inference results
        video_name = os.path.basename(video_path)
        if type(span[0]) is list or type(span[0]) is tuple:
            span = span[0]

        dump = {
            f"{video_name}>>>{query}>>>{span}": {
                "timestamps": timestamps,  # the extracted time span prediction from the model
                "answers": answers,  # the full answer from the model
                "duration": duration,  # save the video duration
            }
        }

        print(
            f"video_path: {video_path}, query: {query}, duration: {duration}, "
            f"answer: {answers}, extracted timestamps: {timestamps}"
        )

        dumps.append(dump)

    nncore.dump(dumps, pred_path)
