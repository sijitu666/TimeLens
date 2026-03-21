"""Build ReasonVTG-Bench candidates (reasoning-intensive VTG evaluation set).

第三/第四优先级任务（你的计划 3/4 的“数据构建脚本”部分）：
- 基于现有 TimeLens-Bench（或其它 VTG 标注）生成“推理型查询”候选
- 输出 jsonl，默认标记 `needs_human_verification=true`，供人工筛选/修正

本脚本的核心策略（先保证可跑通，再迭代质量）：
1) 读取 TimeLens-Bench 的 json（video_id -> {duration, spans, queries}）
2) 规则过滤出原本就“推理型”的 query（after/before/between/比较/序数等）
3) 基于同一视频内多段 span/多条 query 组合生成可验证的推理型 query：
   - causal/later: “Find the moment of B after A” -> gt = span_B（要求 A 结束早于 B 开始）
   - temporal_reasoning: “Find the segment after A and before C” -> gt = span_B（要求 A < B < C）
   - comparison: “Return the earlier of A and B” / “Return the later of A and B” -> gt = earlier/later span

输出记录格式（jsonl，每行一个样本）：
{
  "dataset": "activitynet-timelens",
  "video_id": "v_xxx",
  "duration": 53.44,
  "query": "Find the segment after ... and before ...",
  "span": [20.0, 49.0],
  "query_type": "temporal_reasoning",
  "complexity_level": 4,
  "generation_method": "between_triplet",
  "source_queries": ["...", "...", "..."],
  "source_spans": [[...],[...],[...]],
  "needs_human_verification": true
}

可选：你后续可以用外部 LLM 把模板化 query 改写成更自然的语言，或生成 EG-CoT rationale。
本脚本暂时不强依赖外部 API（先把 pipeline 建起来）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any, Dict, List, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _span_order(a: Tuple[float, float], b: Tuple[float, float]) -> int:
    """Return -1 if a before b, +1 if a after b, 0 if overlap/unclear."""
    if a[1] <= b[0]:
        return -1
    if b[1] <= a[0]:
        return 1
    return 0


def _to_span(x) -> Tuple[float, float]:
    return float(x[0]), float(x[1])


def _classify_query(query: str) -> Tuple[str, int]:
    try:
        from training.train.query_complexity import QueryComplexityEstimator

        qc = QueryComplexityEstimator(mode="rule_based").estimate(query)
        return qc.query_type, qc.complexity_level
    except Exception:
        q = (query or "").lower()
        if re.search(r"after .+ (and |then |,\s*)(before|while|during)", q) or "between" in q:
            return "temporal_reasoning", 4
        if any(k in q for k in ["compared", "more than", "less than", "faster", "slower", "different", "similar", "相比", "不同于", "更"]):
            return "comparison", 3
        if any(k in q for k in ["first", "second", "third", "fourth", "nth", "last", "第一", "第二", "第三", "第", "最后"]):
            return "counting", 2
        if any(k in q for k in ["after", "before", "because", "caused", "led to", "resulted", "之后", "之前", "因为", "导致", "由于"]):
            return "causal", 2
        return "perception", 0


def build_candidates_from_timelens_bench(
    bench_json: Dict[str, Any],
    dataset_name: str,
    rng: random.Random,
    max_per_video: int,
    keep_existing_reasoning: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for vid, item in bench_json.items():
        duration = float(item.get("duration", 0.0) or 0.0)
        spans = [list(map(float, s)) for s in item.get("spans", [])]
        queries = [str(q).strip() for q in item.get("queries", [])]
        if not spans or not queries or len(spans) != len(queries):
            continue

        # 1) 保留原本就带推理特征的 query（作为 baseline reasoning set）
        if keep_existing_reasoning:
            for q, s in zip(queries, spans):
                qt, lvl = _classify_query(q)
                if lvl > 0:
                    out.append(
                        {
                            "dataset": dataset_name,
                            "video_id": vid,
                            "duration": duration,
                            "query": q,
                            "span": s,
                            "query_type": qt,
                            "complexity_level": lvl,
                            "generation_method": "keep_existing",
                            "source_queries": [q],
                            "source_spans": [s],
                            "needs_human_verification": True,
                        }
                    )

        # 2) 组合生成：pair / triplet
        indices = list(range(len(spans)))
        rng.shuffle(indices)
        generated = 0

        # pair: earlier/later
        for i in range(len(indices)):
            if generated >= max_per_video:
                break
            for j in range(i + 1, len(indices)):
                if generated >= max_per_video:
                    break
                a = indices[i]
                b = indices[j]
                sa = _to_span(spans[a])
                sb = _to_span(spans[b])
                order = _span_order(sa, sb)
                if order == 0:
                    continue
                qa, qb = queries[a], queries[b]
                if order == -1:
                    earlier_idx, later_idx = a, b
                else:
                    earlier_idx, later_idx = b, a

                q_earlier = (
                    f"Between the following two events, return the EARLIER one: "
                    f"(1) {queries[earlier_idx]} (2) {queries[later_idx]}."
                )
                out.append(
                    {
                        "dataset": dataset_name,
                        "video_id": vid,
                        "duration": duration,
                        "query": q_earlier,
                        "span": spans[earlier_idx],
                        "query_type": "comparison",
                        "complexity_level": 3,
                        "generation_method": "compare_earlier_pair",
                        "source_queries": [queries[earlier_idx], queries[later_idx]],
                        "source_spans": [spans[earlier_idx], spans[later_idx]],
                        "needs_human_verification": True,
                    }
                )
                generated += 1
                if generated >= max_per_video:
                    break

                q_later = (
                    f"Between the following two events, return the LATER one: "
                    f"(1) {queries[earlier_idx]} (2) {queries[later_idx]}."
                )
                out.append(
                    {
                        "dataset": dataset_name,
                        "video_id": vid,
                        "duration": duration,
                        "query": q_later,
                        "span": spans[later_idx],
                        "query_type": "comparison",
                        "complexity_level": 3,
                        "generation_method": "compare_later_pair",
                        "source_queries": [queries[earlier_idx], queries[later_idx]],
                        "source_spans": [spans[earlier_idx], spans[later_idx]],
                        "needs_human_verification": True,
                    }
                )
                generated += 1
                if generated >= max_per_video:
                    break

        # triplet: after A and before C -> B
        # 仅当 span 有清晰时序关系（不重叠）时生成
        if len(spans) >= 3 and generated < max_per_video:
            triples = []
            for a in range(len(spans)):
                for b in range(len(spans)):
                    if b == a:
                        continue
                    for c in range(len(spans)):
                        if c == a or c == b:
                            continue
                        sa = _to_span(spans[a])
                        sb = _to_span(spans[b])
                        sc = _to_span(spans[c])
                        if _span_order(sa, sb) == -1 and _span_order(sb, sc) == -1:
                            triples.append((a, b, c))
            rng.shuffle(triples)
            for a, b, c in triples[: max(0, max_per_video - generated)]:
                q_between = (
                    f"Find the segment that happens AFTER: {queries[a]} "
                    f"and BEFORE: {queries[c]}."
                )
                out.append(
                    {
                        "dataset": dataset_name,
                        "video_id": vid,
                        "duration": duration,
                        "query": q_between,
                        "span": spans[b],
                        "query_type": "temporal_reasoning",
                        "complexity_level": 4,
                        "generation_method": "between_triplet",
                        "source_queries": [queries[a], queries[b], queries[c]],
                        "source_spans": [spans[a], spans[b], spans[c]],
                        "needs_human_verification": True,
                    }
                )
                generated += 1
                if generated >= max_per_video:
                    break

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bench_json",
        nargs="+",
        required=True,
        help="One or more TimeLens-Bench json files, e.g. data/TimeLens-Bench/activitynet-timelens.json",
    )
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_per_video", type=int, default=2)
    p.add_argument(
        "--keep_existing_reasoning",
        action="store_true",
        help="Keep existing reasoning-like queries from the source bench.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    all_rows: List[Dict[str, Any]] = []
    for path in args.bench_json:
        dataset_name = os.path.basename(path).replace(".json", "")
        bench = _load_json(path)
        rows = build_candidates_from_timelens_bench(
            bench,
            dataset_name=dataset_name,
            rng=rng,
            max_per_video=args.max_per_video,
            keep_existing_reasoning=args.keep_existing_reasoning,
        )
        all_rows.extend(rows)

    _write_jsonl(args.output_jsonl, all_rows)
    print(f"Wrote {len(all_rows)} candidates to {args.output_jsonl}")


if __name__ == "__main__":
    main()

