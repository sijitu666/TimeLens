import copy
import os
import random
from pathlib import Path

import nncore
import numpy as np
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from timelens.dataset.timelens_data import TimeLens100KDataset, parse_query
from training.data.preprocess import preprocess

GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

# Evidence-Grounded Chain-of-Thought prompt.
#
# - 复杂查询：允许/鼓励输出 <think> 中带 [Evidence@start-end_s] 的证据锚定推理链
# - 简单查询：允许直接输出 <answer>
PROMPT_EGCOT = """Given a video and a textual query, your task is to identify the relevant temporal segment in the video that corresponds to the given query.

For complex queries that require reasoning (causal, comparative, counting, or temporal logic), you should first analyze the video evidence step by step, anchoring each observation to specific timestamps using [Evidence@start-end_s] format, then provide your answer.

For simple perception queries, you can directly provide the answer.

Query: {query}

Respond in one of two formats:

Format A (for queries requiring reasoning):
<think>
[Evidence@X.X-Y.Ys] What you observe at this timestamp
[Evidence@X.X-Y.Ys] Another observation
[Reasoning] Your logical chain connecting evidence to the answer
</think>
<answer>start_time - end_time</answer>

Format B (for simple perception queries):
<answer>start_time - end_time</answer>"""

AUDIO_QUERY_KEYWORDS = {
    "hear",
    "heard",
    "hears",
    "hearing",
    "sound",
    "sounded",
    "sounds",
    "sounding",
    "audio",
}


def _is_audio_related_query(query: str) -> bool:
    words = query.strip("?").lower().split()
    return any(keyword in words for keyword in AUDIO_QUERY_KEYWORDS)


def _normalize_spans(span):
    if isinstance(span, tuple):
        return [list(span)]
    if isinstance(span, list) and len(span) > 0 and isinstance(span[0], (list, tuple)):
        return [list(s) for s in span]
    if isinstance(span, list) and len(span) == 2 and isinstance(span[0], (int, float)):
        return [span]
    raise ValueError(f"Unsupported span format: {span}")


def _format_response(spans):
    return (
        "The event happens in "
        + ", ".join([f"{s:.1f} - {e:.1f} seconds" for s, e in spans])
        + "."
    )


def _format_answer_tag(spans):
    # 训练数据 span 可能是多个区间；EG-CoT/TimeLens 评测默认取第一个
    s, e = spans[0]
    return f"<answer>{s:.1f} - {e:.1f}</answer>"


def _select_prompt_text(query: str, data_args) -> str:
    prompt_template = getattr(data_args, "prompt_template", "legacy")
    if prompt_template == "egcot":
        return PROMPT_EGCOT.format(query=query)
    return GROUNDING_PROMPT.format(query)


def _build_video_content(anno, data_args, include_video_range=False):
    content = {
        "type": "video",
        "video": anno["video_path"],
        "min_pixels": int(data_args.min_tokens * 32 * 32),
        "total_pixels": int(data_args.total_tokens * 32 * 32),
        "fps": float(data_args.fps),
    }
    if include_video_range:
        content["video_start"] = anno.get("video_start")
        content["video_end"] = anno.get("video_end")
    if getattr(data_args, "fps_max_frames", None) is not None:
        content["max_frames"] = int(data_args.fps_max_frames)
    return content


def _load_filtered_annos(path: str):
    loaded = nncore.load(path)
    if isinstance(loaded, dict):
        loaded = [loaded]
    if loaded is None:
        return []
    annos = []
    for raw in loaded:
        if "source" not in raw or "query" not in raw:
            continue
        annos.append(
            {
                "source": raw["source"],
                "data_type": raw.get("data_type", "grounding"),
                "video_path": raw["video_path"],
                "duration": raw["duration"],
                "query": parse_query(raw["query"]),
                "span": raw["span"],
                "iou": raw.get("iou"),
                "pred": raw.get("pred"),
                "answer": raw.get("answer"),
            }
        )
    return annos


def _resolve_video_path(video_path: str) -> str:
    """Resolve possibly-relative video paths to real files.

    EG-CoT jsonl built by scripts/build_egcot_data.py may store paths like
    "cosmo_cap/xxx.mp4" (without the TimeLens-100K VIDEO_ROOT prefix). In that
    case, we try to prepend TimeLens100KDataset.VIDEO_ROOT.
    """
    if not video_path:
        return video_path

    p = Path(str(video_path))
    if p.is_absolute():
        return str(p)
    if p.exists():
        return str(p)

    # Try TimeLens-100K video root first.
    candidate = Path(TimeLens100KDataset.VIDEO_ROOT) / str(video_path)
    if candidate.exists():
        return str(candidate)

    # As a fallback, also try TimeLens-Bench root (useful if users reuse loaders).
    bench_root = Path("data/TimeLens-Bench/videos") / str(video_path)
    if bench_root.exists():
        return str(bench_root)

    # Return the 100K-resolved path for better error messages.
    return str(candidate)


def _load_egcot_annos(path: str):
    """Load EG-CoT SFT jsonl produced by `scripts/build_egcot_data.py`.

    约定输入字段：
    - video / video_path
    - query
    - timestamps 或 span
    - duration
    - response（可选；perception 可只包含 <answer>）
    """
    loaded = nncore.load(path)
    if isinstance(loaded, dict):
        loaded = [loaded]
    if loaded is None:
        return []

    annos = []
    for raw in loaded:
        if not isinstance(raw, dict):
            continue
        if "query" not in raw:
            continue

        video_path = raw.get("video_path") or raw.get("video")
        if not video_path:
            continue

        video_path = _resolve_video_path(str(video_path))

        # timestamps -> span
        span = raw.get("span")
        if span is None and isinstance(raw.get("timestamps"), list) and len(raw["timestamps"]) == 2:
            span = raw["timestamps"]

        if span is None:
            continue

        annos.append(
            {
                "source": raw.get("source", raw.get("dataset", "egcot")),
                "data_type": raw.get("data_type", "grounding"),
                "video_path": str(video_path),
                "duration": float(raw.get("duration", 0.0) or 0.0),
                "query": parse_query(str(raw["query"])),
                "span": span,
                # EG-CoT fields
                "response": raw.get("response"),
                "query_type": raw.get("query_type"),
                "complexity_level": raw.get("complexity_level"),
            }
        )
    return annos


class GroundingDataset(Dataset):
    def __init__(
        self,
        processor,
        model_args,
        data_args,
        training_args,
        dataset_name: str,
        filter_args=None,
        training_mode: str = "sft",
    ):
        super().__init__()
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.training_mode = training_mode
        self.max_data_retries = int(getattr(data_args, "max_data_retries", 20))
        self._bad_video_paths = set()

        if dataset_name in ("gemini_refined_data", "timelens-100k"):
            base_annos = TimeLens100KDataset.load_annos(split="train")
            if dataset_name == "gemini_refined_data":
                raw_annos = [
                    anno
                    for anno in base_annos
                    if not _is_audio_related_query(anno["query"])
                ]
            else:
                raw_annos = base_annos
        elif dataset_name == "filtered_hybrid":
            if not data_args.raw_anno_path:
                raise ValueError(
                    "raw_anno_path is required for filtered_hybrid dataset."
                )
            if not Path(data_args.raw_anno_path).exists():
                raise FileNotFoundError(
                    f"raw_anno_path does not exist: {data_args.raw_anno_path}"
                )
            raw_annos = _load_filtered_annos(data_args.raw_anno_path)
        elif dataset_name == "egcot_jsonl":
            if not getattr(data_args, "data_path", None):
                raise ValueError("data_path is required for egcot_jsonl dataset.")
            if not Path(data_args.data_path).exists():
                raise FileNotFoundError(f"data_path does not exist: {data_args.data_path}")
            raw_annos = _load_egcot_annos(data_args.data_path)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        annos = []
        for anno in raw_annos:
            num_words = len(anno["query"].split(" "))
            if data_args.min_num_words >= 0 and num_words < data_args.min_num_words:
                continue
            if data_args.max_num_words >= 0 and num_words > data_args.max_num_words:
                continue
            if (
                data_args.min_video_len >= 0
                and anno.get("duration", float("inf")) < data_args.min_video_len
            ):
                continue
            if (
                data_args.max_video_len >= 0
                and anno.get("duration", 0) > data_args.max_video_len
            ):
                continue
            duration = anno.get("duration")
            spans = _normalize_spans(anno["span"])
            if duration and not any(0 <= s <= e <= duration for s, e in spans):
                continue
            anno = dict(anno)
            anno["span"] = spans
            annos.append(anno)

        if filter_args is not None:
            annos = self._filter_annos(annos, filter_args)

        self.annos = annos
        self.raw_length = len(raw_annos)

    def _filter_annos(self, annos, filter_args):
        unique_videos = filter_args.get("unique_videos", False)
        if unique_videos:
            seen = set()
            uniq = []
            for anno in annos:
                vpath = anno["video_path"]
                if vpath in seen:
                    continue
                seen.add(vpath)
                uniq.append(anno)
            annos = uniq

        filter_ratio = filter_args.get("filter_ratio")
        filter_target_size = filter_args.get("filter_target_size")
        if filter_ratio is None and filter_target_size is None:
            return annos

        gaussian_filter_mean = getattr(self.data_args, "gaussian_filter_mean", None)
        gaussian_filter_std = getattr(self.data_args, "gaussian_filter_std", None)
        if (gaussian_filter_mean is None) != (gaussian_filter_std is None):
            raise ValueError(
                "gaussian_filter_mean and gaussian_filter_std should be provided together."
            )
        if gaussian_filter_mean is not None and not annos:
            return annos
        if gaussian_filter_mean is not None and "iou" not in annos[0]:
            raise ValueError("Gaussian filtering requires 'iou' in annotations.")

        seed = getattr(self.training_args, "seed", 42)
        rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        buckets = {duration_range: [] for duration_range in filter_args["filter_range"]}
        kept_indices = []
        for idx, anno in enumerate(annos):
            matched = False
            for duration_range in buckets:
                min_duration, max_duration = duration_range
                if min_duration <= anno["duration"] <= max_duration:
                    buckets[duration_range].append(idx)
                    matched = True
                    break
            if not matched:
                kept_indices.append(idx)

        for i, (duration_range, indices) in enumerate(buckets.items()):
            if len(indices) == 0:
                continue
            num_to_select = (
                int(len(indices) * filter_ratio[i])
                if filter_ratio is not None
                else int(filter_target_size[i])
            )
            num_to_select = min(num_to_select, len(indices))

            if gaussian_filter_mean is not None:
                iou_list = np.array(
                    [annos[idx]["iou"] for idx in indices], dtype=np.float64
                )
                weights = np.exp(
                    -0.5
                    * ((iou_list - gaussian_filter_mean) / gaussian_filter_std) ** 2
                )
                if getattr(self.data_args, "fixed_gaussian_sampling", False):
                    num_bins = 20
                    counts, bin_edges = np.histogram(
                        iou_list, bins=num_bins, range=(0, 1)
                    )
                    bin_indices = np.digitize(iou_list, bins=bin_edges)
                    bin_indices = np.clip(bin_indices, 1, num_bins) - 1
                    inverse_density = 1.0 / (counts + 1e-6)
                    weights *= inverse_density[bin_indices]
                weights = weights / weights.sum()
                selected_indices = rng.choice(
                    indices, size=num_to_select, replace=False, p=weights
                ).tolist()
            else:
                selected_indices = py_rng.sample(indices, num_to_select)
            kept_indices.extend(selected_indices)

        return [annos[i] for i in range(len(annos)) if i in kept_indices]

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        if len(self.annos) == 0:
            raise RuntimeError("Empty dataset: no annotations available after filtering.")

        last_exc = None
        base_idx = int(idx) % len(self.annos)
        tries = max(1, self.max_data_retries)

        for offset in range(tries):
            cur_idx = (base_idx + offset) % len(self.annos)
            try:
                if self.training_mode == "sft":
                    return self._getitem_sft(cur_idx)
                if self.training_mode == "grpo":
                    return self._getitem_grpo(cur_idx)
                raise ValueError(f"Unsupported training_mode: {self.training_mode}")
            except Exception as e:
                last_exc = e
                video_path = None
                try:
                    video_path = self.annos[cur_idx].get("video_path")
                except Exception:
                    video_path = None

                if video_path and video_path not in self._bad_video_paths:
                    self._bad_video_paths.add(video_path)
                    rank = os.environ.get("RANK")
                    if rank is None:
                        rank = os.environ.get("LOCAL_RANK")
                    prefix = f"[rank{rank}]" if rank is not None else "[rank?]"
                    print(
                        f"{prefix} Skip bad sample idx={cur_idx} video={video_path} "
                        f"err={type(e).__name__}: {e}",
                        flush=True,
                    )
                continue

        raise RuntimeError(
            f"Failed to fetch a valid sample after {tries} tries. Last error: {last_exc}"
        ) from last_exc

    def _getitem_sft(self, idx):
        anno = copy.deepcopy(self.annos[idx])
        spans = _normalize_spans(anno["span"])

        messages = [
            {
                "role": "user",
                "content": [
                    _build_video_content(anno, self.data_args),
                    {"type": "text", "text": _select_prompt_text(anno["query"], self.data_args)},
                ],
            }
        ]

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        prompt_template = getattr(self.data_args, "prompt_template", "legacy")
        if prompt_template == "egcot":
            # 若使用 scripts/build_egcot_data.py 生成的数据，优先读取其 response 字段
            response = anno.get("response") or _format_answer_tag(spans)
        else:
            response = _format_response(spans)
        messages.append({"role": "assistant", "content": response})

        text = self.processor.apply_chat_template(messages, tokenize=False)
        text = [text.strip()]
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["labels"] = preprocess(
            inputs["input_ids"],
            text[0],
            self.processor.tokenizer,
            self.model_args.conv_type,
        )
        return inputs

    def _getitem_grpo(self, idx):
        anno = copy.deepcopy(self.annos[idx])

        messages = [
            {
                "role": "user",
                "content": [
                    _build_video_content(
                        anno, self.data_args, include_video_range=True
                    ),
                    {"type": "text", "text": _select_prompt_text(anno["query"], self.data_args)},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        text = [text]

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["prompt"] = messages
        inputs["prompt_text"] = text[0]
        inputs["anno"] = anno
        return inputs
