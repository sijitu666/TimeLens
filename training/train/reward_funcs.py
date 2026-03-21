import re

from training.utils.parser import extract_time, extract_answer, iou


EVIDENCE_TS_RE = re.compile(r"\[Evidence@(\d+\.?\d*)-(\d+\.?\d*)s?\]", re.IGNORECASE)
ANSWER_TS_RE = re.compile(r"<answer>\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*</answer>", re.IGNORECASE)


def format_reward(completions, **kwargs):
    """格式奖励：支持两种输出。

    - Format A: `<think> ... </think> <answer>...</answer>`（含推理链）
    - Format B: `<answer>...</answer>`（感知型可跳过推理）
    """
    pattern_a = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
    pattern_b = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern_a, content) or re.fullmatch(pattern_b, content) for content in completion_contents]

    for i, match in enumerate(matches):
        if not match:
            print(f"Completion {i} does not match the required format: {completion_contents[i]}")

    return [1.0 if match else 0.0 for match in matches]


def extract_evidence_timestamps(text: str):
    """从推理链中提取所有 `[Evidence@start-end_s]` 时间戳。"""
    return [(float(s), float(e)) for s, e in EVIDENCE_TS_RE.findall(text or "")]


def extract_answer_timestamps(text: str):
    """从 `<answer>` 标签中提取时间戳。"""
    m = ANSWER_TS_RE.search(text or "")
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def has_thinking(text: str) -> bool:
    t = text or ""
    return "<think>" in t and "</think>" in t


def classify_query_complexity_simple(query: str) -> int:
    """简单版复杂度分类，返回 ref_steps（用于效率惩罚）。

    规则来源：QueryComplexityEstimator(rule_based)。
    这里保持零依赖（仅 stdlib），并在可用时复用统一实现。
    """
    try:
        from training.train.query_complexity import QueryComplexityEstimator

        return QueryComplexityEstimator(mode="rule_based").estimate(query).ref_steps
    except Exception:
        # fallback（避免 import 路径/环境问题影响训练）
        q = (query or "").lower()
        if any(kw in q for kw in ["between", "while", "during", "同时", "期间", "之间"]):
            return 4
        if any(kw in q for kw in ["after", "before", "之后", "之前"]):
            if any(kw in q for kw in ["and then", "before", "之后", "之前", "between", "之间"]):
                return 4
            return 2
        if any(kw in q for kw in ["compared", "more than", "less than", "faster", "slower", "different", "similar", "相比", "不同", "更"]):
            return 3
        if any(kw in q for kw in ["first", "second", "third", "fourth", "nth", "last", "第一", "第二", "第三", "第", "最后"]):
            return 2
        return 0


def evidence_aware_reward(
    prompts,
    completions,
    completion_ids,
    anno,
    prompt_text,
    alpha: float = 0.3,
    lambda_eff: float = 0.1,
    delta: float = 0.05,
    **kwargs,
):
    """Evidence-Aware Reward (EAR).

    公式：
      R = R_answer × (1 + α × R_evidence) × γ_efficiency + δ × 𝟙(Perception)

    - R_answer: tIoU(pred, gt)
    - R_evidence: evidence 时间戳与 gt 的相关性（IoU + proximity）
    - γ_efficiency: 推理步数冗余惩罚（仅对 ref_steps>0 且 num_steps>ref_steps 生效）
    - δ bonus: 感知型查询且未输出 `<think>` 时给小奖励

    兼容 GRPO trainer：参数签名与 `tiou_reward` 一致。
    """
    # 清理 prompt_text（保持与 tiou_reward 一致的日志/解析体验）
    pattern = r'<\|(video_pad|image_pad|vision_start|vision_end)\|>'
    prompt_text = [re.sub(pattern, '', text) for text in prompt_text]

    completion_contents = [completion[0]["content"] for completion in completions]

    rewards = []
    for i, completion in enumerate(completion_contents):
        gt = anno[i]["span"]
        if isinstance(gt[0], list):
            gt = gt[0]
        gt = (float(gt[0]), float(gt[1]))

        query = anno[i].get("query", "")
        duration = float(anno[i].get("duration", 0.0) or 0.0)
        if duration <= 0:
            duration = max(gt[1], 1.0)

        pred = extract_answer_timestamps(completion)
        if pred is None:
            # fallback：如果没有 answer tag，尝试从内容里抽取 time
            answer_text = extract_answer(completion)
            ts = extract_time(answer_text)
            pred = ts[0] if ts else None

        if pred is None:
            rewards.append(0.0)
            continue

        if pred[0] >= pred[1]:
            rewards.append(0.0)
            continue

        r_answer = float(iou(gt, pred))

        ref_steps = classify_query_complexity_simple(query)
        is_perception = ref_steps == 0

        evidence_ts = extract_evidence_timestamps(completion)
        if len(evidence_ts) > 0:
            relevance_scores = []
            gt_center = (gt[0] + gt[1]) / 2.0
            for ev_s, ev_e in evidence_ts:
                # 无效范围
                if ev_e <= 0 or ev_s >= duration:
                    relevance_scores.append(0.0)
                    continue
                # clamp 到视频范围
                ev_s = max(0.0, min(float(ev_s), duration))
                ev_e = max(0.0, min(float(ev_e), duration))
                if ev_e <= ev_s:
                    relevance_scores.append(0.0)
                    continue

                ev_iou = float(iou((ev_s, ev_e), gt))
                ev_center = (ev_s + ev_e) / 2.0
                distance_ratio = abs(gt_center - ev_center) / max(duration, 1.0)
                proximity_score = max(0.0, 1.0 - distance_ratio * 2.0)
                relevance_scores.append(max(ev_iou, proximity_score * 0.5))
            r_evidence = sum(relevance_scores) / max(1, len(relevance_scores))
        else:
            r_evidence = 0.0

        num_steps = len(evidence_ts)
        if ref_steps > 0 and num_steps > ref_steps:
            gamma = max(0.0, 1.0 - lambda_eff * float(num_steps - ref_steps))
        else:
            gamma = 1.0

        adaptive_bonus = delta if (is_perception and not has_thinking(completion)) else 0.0

        reward = r_answer * (1.0 + alpha * r_evidence) * gamma + adaptive_bonus
        rewards.append(float(reward))

    return rewards


def tiou_reward(prompts, completions, completion_ids, anno, prompt_text, **kwargs):
    """Reward function that returns temporal IoU between predicted and ground truth spans."""
    pattern = r'<\|(video_pad|image_pad|vision_start|vision_end)\|>'
    prompt_text = [re.sub(pattern, '', text) for text in prompt_text]

    completions = [completion[0]["content"] for completion in completions]
    answers = [extract_answer(completion) for completion in completions]
    timestamps_list = [extract_time(answer) for answer in answers]

    rewards = []
    for i, timestamps in enumerate(timestamps_list):
        gt = anno[i]["span"]
        if isinstance(gt[0], list):
            gt = gt[0]

        pred = answers[i]

        if len(timestamps) == 0:
            print(f"Timestamp extraction failed: pred={pred}, IoU will be 0")
            rewards.append(0)
        elif timestamps[0][0] >= timestamps[0][1]:
            print(f"Warning: Invalid timestamp in prediction '{pred}', IoU will be 0")
            rewards.append(0)
        else:
            if len(timestamps) > 1:
                print(f"Warning: Multiple timestamps for '{pred}', using first: {timestamps[0]}")
            rewards.append(iou(gt, timestamps[0]))
            print(f"prompt: {prompt_text[i]}, completion: {completions[i]}, answer: {pred}, gt: {gt}, tIoU: {rewards[i]}")

    return rewards


REWARD_FUNCS_DICT = {
    "tiou": tiou_reward,
    "format": format_reward,
    "ear": evidence_aware_reward,
}


def load_reward_funcs(reward_func_names):
    return [
        REWARD_FUNCS_DICT[func_name.strip()]
        for func_name in reward_func_names.split(",")
    ]
