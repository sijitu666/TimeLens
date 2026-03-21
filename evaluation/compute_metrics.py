# Copyright (c) 2025 Jun Zhang. Licensed under the BSD-3-Clause License.
# Modified from https://github.com/RenShuhuai-Andy/TimeChat/blob/master/utils/format_tvg.py
import argparse
import json
from pathlib import Path
from collections import defaultdict

from timelens.utils import extract_time, iou, read_json


# read JSONL file
def read_jsonl_return_dict(file_path):
    """Read JSONL file and merge into a single dictionary"""
    data = {}
    with open(file_path, "r") as reader:
        for line in reader:
            item = json.loads(line)
            if isinstance(item, dict):
                data.update(item)
            else:
                raise ValueError("Each line in the JSONL file should be a dictionary.")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", default="your_result.json")
    parser.add_argument(
        "--group_by_query_type",
        action="store_true",
        help="Report metrics grouped by query type (perception/causal/comparison/counting/temporal_reasoning).",
    )
    args = parser.parse_args()

    if args.f.endswith(".json"):
        datas = read_json(args.f)
    elif args.f.endswith(".jsonl"):
        datas = read_jsonl_return_dict(args.f)
    else:
        raise ValueError(
            "Unsupported file format. Please provide a .json or .jsonl file."
        )

    num_annos = len(datas)
    ious = []
    grouped_ious = defaultdict(list)

    def _classify_query_type(query: str) -> str:
        try:
            from training.train.query_complexity import QueryComplexityEstimator

            return QueryComplexityEstimator(mode="rule_based").estimate(query).query_type
        except Exception:
            q = (query or "").lower()
            if any(k in q for k in ["between", "while", "during", "同时", "期间", "之间"]):
                return "temporal_reasoning"
            if any(k in q for k in ["compared", "more than", "less than", "faster", "slower", "different", "similar", "相比", "不同于", "更"]):
                return "comparison"
            if any(k in q for k in ["first", "second", "third", "fourth", "nth", "last", "第一", "第二", "第三", "第", "最后"]):
                return "counting"
            if any(k in q for k in ["after", "before", "because", "caused", "led to", "resulted", "之后", "之前", "因为", "导致", "由于"]):
                return "causal"
            return "perception"

    # Example of annotation format:
    # anno = {
    #     f'{video_name}>>>{query}>>>{ground_truth_span}': {
    #         "timestamps": [(0, 10)],  # predicted time spans
    #         "answers": "The event happens from 0 - 10 seconds." # raw text answer from the model
    #     }
    # }
    for key, pred in datas.items():
        video_id, query, gt_span = key.split(">>>")
        gt_span = eval(gt_span)

        if type(pred) is dict:
            if "timestamps" in pred:
                timestamps = pred["timestamps"]
            elif "answers" in pred:  # parse the raw answer
                timestamps = extract_time(pred["answers"])
            else:
                raise ValueError(f"Unexpected key in prediction: {pred}")
        else:
            raise ValueError(
                f"Unexpected type for prediction: {type(pred)}. Expected dict or str."
            )

        if len(timestamps) > 1:
            print(
                f"Warning: Multiple timestamp pairs found for prediction '{pred}', using the first pair: {timestamps[0]}"
            )
        elif len(timestamps) == 0:
            print(
                f"Timestamp extraction failed: pred={pred},timestamps={timestamps}, IoU will be 0"
            )
            timestamps = [(-100, -100)]

        timestamps = timestamps[0]  # only use the first pair of timestamps
        if timestamps[0] >= timestamps[1]:
            print(
                f"Warning: Invalid timestamp found in prediction '{pred}', start timestamp >= end timestamp, IoU will be 0"
            )

        cur_iou = iou(gt_span, timestamps)
        ious.append(cur_iou)
        if args.group_by_query_type:
            grouped_ious[_classify_query_type(query)].append(cur_iou)

    recall = {0.3: 0, 0.5: 0, 0.7: 0}
    for iou_threshold in [0.3, 0.5, 0.7]:
        for cur_iou in ious:
            if cur_iou >= iou_threshold:
                recall[iou_threshold] += 1

    RESULT_STR = "IOU 0.3: {0}\nIOU 0.5: {1}\nIOU 0.7: {2}\nmIOU: {3}".format(
        recall[0.3] * 100 / num_annos,
        recall[0.5] * 100 / num_annos,
        recall[0.7] * 100 / num_annos,
        sum(ious) * 100 / num_annos,  # mean IoU (mIoU)
    )
    print(RESULT_STR)

    if args.group_by_query_type:
        lines = ["\n--- Grouped by query_type ---"]
        for qt in ["perception", "causal", "comparison", "counting", "temporal_reasoning"]:
            arr = grouped_ious.get(qt, [])
            if not arr:
                continue
            n = len(arr)
            r03 = sum(1 for x in arr if x >= 0.3) * 100 / n
            r05 = sum(1 for x in arr if x >= 0.5) * 100 / n
            r07 = sum(1 for x in arr if x >= 0.7) * 100 / n
            miou = sum(arr) * 100 / n
            lines.append(
                f"{qt}: N={n} | IOU0.3={r03:.2f} IOU0.5={r05:.2f} IOU0.7={r07:.2f} mIOU={miou:.2f}"
            )
        GROUPED_STR = "\n".join(lines)
        print(GROUPED_STR)

    # Save the result to a .log file
    log_file_path = Path(args.f).with_suffix(".log")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Processed file: {args.f}\n")
        log_file.write(RESULT_STR)
        if args.group_by_query_type:
            log_file.write("\n")
            log_file.write(GROUPED_STR)
