"""Build EG-CoT (Evidence-Grounded Chain-of-Thought) SFT data from TimeLens-100K jsonl.

第一优先级目标（对应你的科研实现计划 1.1）：
- 对每条样本做规则复杂度分类（perception/causal/comparison/counting/temporal_reasoning）
- perception 样本输出仅 `<answer>`（跳过推理链）
- reasoning 样本输出 EG-CoT：`<think> ... [Evidence@start-end_s] ... </think> <answer>...`

脚本支持两种推理链生成方式：
1) `--llm_provider none`：启用可复现的启发式 EG-CoT（不依赖外部 API，先跑通管线）
2) `--llm_provider openai|gemini`：用外部 LLM 生成推理链（可选；支持缓存、重试、校验）

输入（TimeLens-100K jsonl 常见字段）：
{
  "video": "path/to/video.mp4"  # 或 video_path
  "query": "...",
  "timestamps": [12.5, 18.3]    # 或 span
  "duration": 45.0,
  "dataset": "activitynet"
}

输出字段（新增 query_type/complexity_level/response）：
{
  ...,
  "query_type": "causal",
  "complexity_level": 2,
  "response": "<think>...<answer>12.5 - 18.3</answer>"
}

用法示例：
  python scripts/build_egcot_data.py \
    --input_jsonl data/TimeLens-100K/timelens-100k.jsonl \
    --output_jsonl output/TimeLens-8B/sft/egcot_timelens100k.jsonl \
    --llm_provider none \
    --target_reasoning_ratio 0.4

  # 若你要接入外部 LLM（可选，建议先用 none 跑通）
  OPENAI_API_KEY=... python scripts/build_egcot_data.py --llm_provider openai --llm_model gpt-4o-mini ...

注意：本脚本会强制让最终 `<answer>` 与 GT 一致（避免 LLM 漂移破坏监督信号）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


EGCOT_GENERATION_PROMPT = """You are annotating training data for a video temporal grounding model.

Given a video query and its ground-truth timestamps, generate a structured reasoning chain in EG-CoT (Evidence-Grounded Chain-of-Thought) format.

Rules:
1. Each reasoning step MUST be anchored to a specific timestamp using [Evidence@start-end_s] format
2. The [Evidence@...] timestamps should be approximate segments within the video that support the reasoning
3. The final <answer> must match the ground-truth timestamps
4. Keep reasoning concise (2-4 steps for causal queries, 3-5 for complex ones)
5. The evidence timestamps should be BEFORE or OVERLAPPING with the ground-truth, representing the clues the model would observe

Query: {query}
Query Type: {query_type}
Ground-truth timestamps: {start}s - {end}s
Video duration: {duration}s

Generate the EG-CoT response in this exact format:
<think>
[Evidence@X.X-Y.Ys] Description of what is observed at this timestamp
[Evidence@X.X-Y.Ys] Description of another observation
[Reasoning] Logical chain connecting evidence to the answer
</think>
<answer>{start} - {end}</answer>
"""


EVIDENCE_RE = re.compile(r"\[Evidence@(\d+\.?\d*)-(\d+\.?\d*)s?\]", re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*</answer>", re.IGNORECASE)


def classify_query_complexity(query: str) -> Tuple[str, int]:
    """规则复杂度分类：返回 (query_type, complexity_level)。

    设计目标：
    - 速度快、可解释
    - 先覆盖常见模式，后续可替换为 QueryComplexityEstimator
    """
    q = (query or "").strip().lower()

    # 时序推理（最高优先级）
    temporal_patterns = [
        r"after .+ (and |then |,\s*)(before|while|during)",
        r"between .+ and",
        r"while .+ then",
        r"介于.+之间",
        r"在.+之后.+之前",
    ]
    for pat in temporal_patterns:
        if re.search(pat, q):
            return ("temporal_reasoning", 4)

    # 比较型
    comparison_keywords = [
        r"compared to",
        r"more than",
        r"less than",
        r"faster",
        r"slower",
        r"different from",
        r"similar to",
        r"相比",
        r"不同于",
        r"比.+更",
    ]
    for kw in comparison_keywords:
        if re.search(kw, q):
            return ("comparison", 3)

    # 因果型
    causal_keywords = [
        "after",
        "before",
        "because",
        "caused",
        "led to",
        "resulted",
        "in response to",
        "之后",
        "之前",
        "因为",
        "导致",
        "由于",
    ]
    if any(kw in q for kw in causal_keywords):
        return ("causal", 2)

    # 计数型
    counting_keywords = [
        "first time",
        "second time",
        "third",
        "nth",
        "last time",
        "第一次",
        "第二次",
        "第三次",
        "最后一次",
    ]
    if any(kw in q for kw in counting_keywords):
        return ("counting", 2)

    return ("perception", 0)


def classify_query_complexity_via_estimator(query: str) -> Tuple[str, int]:
    """优先复用训练侧统一实现（QueryComplexityEstimator），避免规则漂移。

    若 import 失败则回退到本文件内置规则。
    """
    try:
        from training.train.query_complexity import QueryComplexityEstimator

        qc = QueryComplexityEstimator(mode="rule_based").estimate(query)
        return (qc.query_type, qc.complexity_level)
    except Exception:
        return classify_query_complexity(query)


def _fmt_ts(x: float) -> str:
    # 用 1 位小数以和 TimeLens 数据/评测保持一致
    return f"{x:.1f}"


def _extract_gt_span(obj: Dict[str, Any]) -> Tuple[float, float]:
    if "timestamps" in obj and isinstance(obj["timestamps"], list) and len(obj["timestamps"]) == 2:
        s, e = obj["timestamps"]
        return float(s), float(e)
    if "span" in obj:
        span = obj["span"]
        # span 可能是 [s,e] 或 [[s,e]]
        if isinstance(span, list) and len(span) == 2 and all(isinstance(x, (int, float)) for x in span):
            return float(span[0]), float(span[1])
        if isinstance(span, list) and span and isinstance(span[0], (list, tuple)):
            s, e = span[0]
            return float(s), float(e)
    raise ValueError("Cannot find ground-truth timestamps/span in input record")


def _get_video_path(obj: Dict[str, Any]) -> str:
    if "video_path" in obj:
        return str(obj["video_path"])
    if "video" in obj:
        return str(obj["video"])
    raise ValueError("Missing video/video_path")


def _stable_id(video_path: str, query: str, start: float, end: float) -> str:
    raw = f"{video_path}\n{query}\n{start:.3f}\n{end:.3f}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def extract_evidence_timestamps(text: str) -> List[Tuple[float, float]]:
    return [(float(a), float(b)) for a, b in EVIDENCE_RE.findall(text or "")]


def extract_answer_timestamps(text: str) -> Optional[Tuple[float, float]]:
    m = ANSWER_RE.search(text or "")
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def has_thinking(text: str) -> bool:
    t = text or ""
    return "<think>" in t and "</think>" in t


def _clamp_span(span: Tuple[float, float], duration: float) -> Tuple[float, float]:
    s, e = span
    s = max(0.0, min(float(s), float(duration)))
    e = max(0.0, min(float(e), float(duration)))
    if e < s:
        s, e = e, s
    return s, e


def _heuristic_egcot(query: str, query_type: str, gt: Tuple[float, float], duration: float) -> str:
    """不依赖外部 LLM 的启发式 EG-CoT：先把端到端数据管线跑通。"""
    s, e = gt
    s, e = _clamp_span((s, e), duration)
    span_len = max(e - s, 0.1)

    # 证据 1：尽量取一个“事件前因/上下文”窗口
    ctx_end = s
    ctx_start = max(0.0, ctx_end - min(5.0, max(1.0, span_len)))
    ctx = (ctx_start, ctx_end)

    # 证据 2：事件主体（覆盖 GT 的前半段）
    ev_end = min(duration, s + min(span_len, 3.0))
    ev = (s, max(ev_end, s + 0.1))

    evidence_lines = []
    if ctx[1] - ctx[0] >= 0.1 and query_type != "perception":
        evidence_lines.append(
            f"[Evidence@{_fmt_ts(ctx[0])}-{_fmt_ts(ctx[1])}s] Context related to the query happens before the target event"
        )
    evidence_lines.append(
        f"[Evidence@{_fmt_ts(ev[0])}-{_fmt_ts(ev[1])}s] The key action described in the query is observed"
    )

    reasoning = (
        f"[Reasoning] This is a {query_type} query. "
        f"Use the evidence timestamps to locate the target segment, and output the final span."
    )

    return (
        "<think>\n"
        + "\n".join(evidence_lines)
        + "\n"
        + reasoning
        + "\n</think>\n"
        + f"<answer>{_fmt_ts(s)} - {_fmt_ts(e)}</answer>"
    )


def _force_answer(text: str, gt: Tuple[float, float]) -> str:
    """强制 `<answer>` 与 GT 一致；若不存在则追加。"""
    gt_s, gt_e = gt
    answer = f"<answer>{_fmt_ts(gt_s)} - {_fmt_ts(gt_e)}</answer>"

    if ANSWER_RE.search(text or ""):
        return ANSWER_RE.sub(answer, text)
    # 没有 <answer>：尽量不破坏已有内容
    return (text.rstrip() + "\n" + answer).strip() + "\n"


def _normalize_llm_response(text: str, query_type: str, gt: Tuple[float, float], duration: float) -> str:
    """对 LLM 生成的 EG-CoT 做最小修复：

    - 没有 <think> 时补齐
    - 没有 evidence 时补 1 条
    - evidence 超界时 clamp
    - 强制 <answer> == GT
    """
    text = (text or "").strip()
    if not text:
        return ""

    # 补齐 think
    if not has_thinking(text):
        # 若模型只输出了 <answer>，则用启发式补一段 think
        if ANSWER_RE.search(text):
            text = _heuristic_egcot("", query_type, gt, duration)
        else:
            text = "<think>\n" + text + "\n</think>\n" + ""

    # evidence 至少 1 条
    evidence = extract_evidence_timestamps(text)
    if len(evidence) == 0:
        s, e = gt
        text = text.replace(
            "<think>",
            f"<think>\n[Evidence@{_fmt_ts(s)}-{_fmt_ts(e)}s] Evidence supporting the answer\n",
            1,
        )

    # clamp evidence
    def _clamp_evidence(m: re.Match) -> str:
        s = float(m.group(1))
        e = float(m.group(2))
        s, e = _clamp_span((s, e), duration)
        return f"[Evidence@{_fmt_ts(s)}-{_fmt_ts(e)}s]"

    text = EVIDENCE_RE.sub(_clamp_evidence, text)
    text = _force_answer(text, gt)
    return text.strip()


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    import requests

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:2000]}")
    return resp.json()


def call_llm(
    provider: str,
    model: str,
    prompt: str,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
) -> str:
    """最小可用的外部 LLM 调用（不引入额外 SDK；用 requests 直连 REST）。"""
    provider = provider.lower()
    if provider == "none":
        raise ValueError("call_llm should not be called when provider=none")

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError("Missing OPENAI_API_KEY env")
                url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                }
                data = _http_post_json(url, headers, payload)
                return data["choices"][0]["message"]["content"]

            if provider == "gemini":
                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise RuntimeError("Missing GEMINI_API_KEY/GOOGLE_API_KEY env")
                base = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
                # Generative Language API v1beta
                url = f"{base}/v1beta/models/{model}:generateContent?key={api_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2},
                }
                data = _http_post_json(url, headers, payload)
                return data["candidates"][0]["content"]["parts"][0]["text"]

            raise ValueError(f"Unsupported llm_provider: {provider}")
        except Exception as e:
            last_err = e
            if attempt + 1 < max_retries:
                time.sleep(retry_sleep * (attempt + 1))
                continue
            raise

    raise RuntimeError(f"LLM call failed: {last_err}")


def _load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load jsonl and normalize into *one sample per record*.

    TimeLens-100K 原始 jsonl 是“每行一个视频 + events 列表”的结构：
    {
      "source": "...",
      "video_path": "...",
      "duration": ...,
      "events": [ {"query": ..., "span": [[s,e]]}, ... ]
    }

    本函数会把其展开成：每个 event 一条记录（包含 video_path/query/span/duration/source）。
    """

    def _expand(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        if isinstance(obj, dict) and "events" in obj and isinstance(obj["events"], list):
            out = []
            for ev in obj["events"]:
                if not isinstance(ev, dict) or "query" not in ev or "span" not in ev:
                    continue
                out.append(
                    {
                        "source": obj.get("source"),
                        "video_path": obj.get("video_path") or obj.get("video"),
                        "duration": obj.get("duration"),
                        "query": ev.get("query"),
                        "span": ev.get("span"),
                    }
                )
            return out
        return [obj]

    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for rec in _expand(obj):
                data.append(rec)
                if max_samples is not None and len(data) >= max_samples:
                    return data
    return data


def _write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_records(
    raw_records: List[Dict[str, Any]],
    llm_provider: str,
    llm_model: str,
    seed: int,
    target_reasoning_ratio: float,
    max_retries: int,
    cache_path: str,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    # 预先分类，决定采样比例
    meta = []
    for r in raw_records:
        query = r.get("query", "")
        qt, lvl = classify_query_complexity_via_estimator(query)
        meta.append((qt, lvl))

    perception_idx = [i for i, (_, lvl) in enumerate(meta) if lvl == 0]
    reasoning_idx = [i for i, (_, lvl) in enumerate(meta) if lvl > 0]

    keep_perception = set(perception_idx)
    keep_reasoning = set(reasoning_idx)

    if 0.0 <= target_reasoning_ratio < 1.0:
        # 目标：通过 subsample 来逼近目标占比。
        # 默认策略：尽量保留全部 reasoning 样本，再下采样 perception。
        R = len(reasoning_idx)
        P = len(perception_idx)
        if R == 0:
            # 全是 perception，无法产生 reasoning 占比
            keep_perception = set(perception_idx)
            keep_reasoning = set()
        else:
            # 若保留全部 reasoning，为达到 ratio，需要的 perception 数：P_needed = R*(1-r)/r
            p_needed = int(round(R * (1.0 - target_reasoning_ratio) / max(1e-6, target_reasoning_ratio)))
            p_keep = min(P, max(0, p_needed))
            keep_perception = set(rng.sample(perception_idx, p_keep)) if p_keep < P else set(perception_idx)
            keep_reasoning = set(reasoning_idx)

            # 如果 perception 太少导致实际 ratio > target（更“推理密集”），保持不动（保留 reasoning 优先）。

    # LLM 缓存：key -> response
    cache: Dict[str, str] = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if "key" in obj and "response" in obj:
                        cache[str(obj["key"])] = str(obj["response"])
        except Exception:
            cache = {}

    cache_f = None
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        cache_f = open(cache_path, "a", encoding="utf-8")

    out: List[Dict[str, Any]] = []
    try:
        for i, r in enumerate(raw_records):
            qt, lvl = meta[i]
            if lvl == 0 and i not in keep_perception:
                continue
            if lvl > 0 and i not in keep_reasoning:
                continue

            query = str(r.get("query", ""))
            duration = float(r.get("duration", 0.0) or 0.0)
            gt = _extract_gt_span(r)
            gt = _clamp_span(gt, duration if duration > 0 else max(gt[1], 1.0))

            # 统一 video_path 字段
            video_path = _get_video_path(r)

            record = dict(r)
            record["video_path"] = video_path
            record.pop("video", None)
            record["query_type"] = qt
            record["complexity_level"] = lvl

            if lvl == 0:
                record["response"] = f"<answer>{_fmt_ts(gt[0])} - {_fmt_ts(gt[1])}</answer>"
                out.append(record)
                continue

            # reasoning：生成 EG-CoT
            key = _stable_id(video_path, query, gt[0], gt[1])
            if key in cache:
                resp = cache[key]
            else:
                if llm_provider == "none":
                    resp = _heuristic_egcot(query, qt, gt, duration)
                else:
                    prompt = EGCOT_GENERATION_PROMPT.format(
                        query=query,
                        query_type=qt,
                        start=_fmt_ts(gt[0]),
                        end=_fmt_ts(gt[1]),
                        duration=_fmt_ts(duration),
                    )
                    raw_text = call_llm(
                        provider=llm_provider,
                        model=llm_model,
                        prompt=prompt,
                        max_retries=max_retries,
                    )
                    resp = _normalize_llm_response(raw_text, qt, gt, duration)

                cache[key] = resp
                if cache_f is not None:
                    cache_f.write(json.dumps({"key": key, "response": resp}, ensure_ascii=False) + "\n")
                    cache_f.flush()

            # 最后做一次最小校验 + 强制 answer
            resp = _normalize_llm_response(resp, qt, gt, duration)
            record["response"] = resp
            out.append(record)
    finally:
        if cache_f is not None:
            cache_f.close()

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--target_reasoning_ratio",
        type=float,
        default=0.4,
        help="输出数据中 reasoning 样本的目标占比；设为 -1 表示不做比例采样（保留全部）。",
    )
    p.add_argument(
        "--llm_provider",
        default="none",
        choices=["none", "openai", "gemini"],
        help="推理链生成方式：none=启发式；openai/gemini=外部 API。",
    )
    p.add_argument(
        "--llm_model",
        default="gpt-4o-mini",
        help="外部 LLM 模型名（openai/gemini 各自语义）。",
    )
    p.add_argument("--llm_max_retries", type=int, default=3)
    p.add_argument(
        "--llm_cache_jsonl",
        default="output/egcot_llm_cache.jsonl",
        help="LLM 生成缓存（jsonl，避免重复计费/重复请求）。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = _load_jsonl(args.input_jsonl, max_samples=args.max_samples)
    ratio = args.target_reasoning_ratio
    if ratio < 0:
        ratio = 1.0  # 不采样：保留全部 reasoning

    built = build_records(
        raw_records=raw,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        seed=args.seed,
        target_reasoning_ratio=ratio if args.target_reasoning_ratio >= 0 else -1,
        max_retries=args.llm_max_retries,
        cache_path=args.llm_cache_jsonl,
    )
    _write_jsonl(args.output_jsonl, built)

    # 简单统计
    counts: Dict[str, int] = {}
    for r in built:
        counts[r.get("query_type", "unknown")] = counts.get(r.get("query_type", "unknown"), 0) + 1
    total = len(built)
    reasoning = sum(v for k, v in counts.items() if k != "perception")
    print(f"Wrote {total} records to {args.output_jsonl}")
    print(f"Reasoning ratio: {reasoning / max(1, total):.3f}")
    print("Breakdown:", counts)


if __name__ == "__main__":
    main()
