"""Generate golden_set.json v1 from RouterBench peek.csv (stratified sample).
读取golden_set.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PEEK = ROOT / "data" / "peek.csv"
OUT = Path(__file__).resolve().parent / "golden_set.json"

# Curated eval_name list for v1 golden set (diverse task families)
CURATED_EVALS = [
    "grade-school-math",
    "mbpp",
    "hellaswag",
    "arc-challenge",
    "winogrande",
    "mmlu-high-school-mathematics",
    "mmlu-professional-law",
    "mmlu-clinical-knowledge",
    "mmlu-high-school-biology",
    "mmlu-elementary-mathematics",
    "Chinese_character_riddles",
    "chinese_idioms",
    "consensus_summary",
    "bias_detection",
    "abstract2title",
    "mmlu-moral-scenarios",
    "mmlu-philosophy",
    "mmlu-security-studies",
    "chinese_ancient_poetry",
    "mmlu-prehistory",
    "grade-school-math",  # duplicate ok for more samples
    "gsm8k",  # may not exist in peek
]

SAMPLES_PER_EVAL = 2
RANDOM_SEED = 42

TASK_FAMILY_RULES: list[tuple[str, list[str], dict]] = [
    (
        "numeric_reasoning",
        ["grade-school-math", "mtbench-math", "gsm8k", "mmlu-elementary-mathematics", "mmlu-high-school-mathematics", "mmlu-college-mathematics"],
        {
            "expected_behavior": "Extract final numeric answer; match reference exactly.",
            "rubric": "Parse model output for final number (after #### or last line). Success iff normalized numeric value equals reference. Ignore reasoning text.",
            "success_criteria": "deterministic",
            "scorer": "exact_match_numeric",
            "judge_protocol": None,
        },
    ),
    (
        "code_generation",
        ["mbpp"],
        {
            "expected_behavior": "Generated code passes unit tests for the prompt.",
            "rubric": "Execute candidate code against MBPP hidden tests. Success iff all tests pass. Timeout 10s per sample.",
            "success_criteria": "deterministic",
            "scorer": "code_exec",
            "judge_protocol": None,
        },
    ),
    (
        "multiple_choice",
        ["hellaswag", "winogrande", "arc-challenge"] + [f"mmlu-{x}" for x in []],
        {
            "expected_behavior": "Single letter choice A/B/C/D matches reference.",
            "rubric": "Extract first valid choice letter from model output. Success iff matches gold letter (case-insensitive).",
            "success_criteria": "deterministic",
            "scorer": "choice_match",
            "judge_protocol": None,
        },
    ),
    (
        "open_generation",
        ["consensus_summary", "bias_detection", "abstract2title", "chinese_hard_translations"],
        {
            "expected_behavior": "Response satisfies task-specific rubric via dual judges.",
            "rubric": "Score 1-5 on task completion, factuality, and conciseness. Pass if both judges >=4 after swap-and-aggregate.",
            "success_criteria": "llm_judge",
            "scorer": "g_eval",
            "judge_protocol": "dual_judge_swap_aggregate",
        },
    ),
    (
        "chinese_cultural",
        ["Chinese_character_riddles", "chinese_idioms", "chinese_ancient_poetry", "chinese-lantern-riddles", "chinese_famous_novel"],
        {
            "expected_behavior": "Final answer in brackets matches reference character/word/poem line.",
            "rubric": "Extract content inside [] or 【】 as final answer. Normalize traditional/simplified. Success iff exact match to reference.",
            "success_criteria": "deterministic",
            "scorer": "bracket_extract_match",
            "judge_protocol": None,
        },
    ),
]


def _family_for_eval(eval_name: str) -> tuple[str, dict]:
    en_lower = eval_name.lower()
    if en_lower.startswith("chinese") or "chinese_" in en_lower or eval_name.startswith("Chinese"):
        return "chinese_cultural", TASK_FAMILY_RULES[4][2]
    if eval_name.startswith("mmlu-"):
        return "multiple_choice", TASK_FAMILY_RULES[2][2]
    for family, names, cfg in TASK_FAMILY_RULES:
        if eval_name in names:
            return family, cfg
    if eval_name in ("hellaswag", "winogrande", "arc-challenge"):
        return "multiple_choice", TASK_FAMILY_RULES[2][2]
    if eval_name in ("consensus_summary", "bias_detection", "abstract2title", "chinese_hard_translations"):
        return "open_generation", TASK_FAMILY_RULES[3][2]
    return "open_generation", TASK_FAMILY_RULES[3][2]


def _prompt_text(raw: str) -> str:
    if len(raw) > 2000:
        return raw[:2000] + "…"
    return raw


def main() -> None:
    df = pd.read_csv(PEEK)
    available = set(df["eval_name"].unique())

    evals_to_sample = []
    for e in CURATED_EVALS:
        if e in available and e not in evals_to_sample:
            evals_to_sample.append(e)
    # fill up to ~24 evals from available curated-like names
    for e in sorted(available):
        if len(evals_to_sample) >= 24:
            break
        if e not in evals_to_sample and (
            e.startswith("mmlu-") or "chinese" in e.lower() or e in ("mbpp", "hellaswag", "arc-challenge", "winogrande", "grade-school-math")
        ):
            evals_to_sample.append(e)

    samples = []
    for eval_name in evals_to_sample:
        sub = df[df["eval_name"] == eval_name]
        if sub.empty:
            continue
        picked = sub.sample(n=min(SAMPLES_PER_EVAL, len(sub)), random_state=RANDOM_SEED)
        family, cfg = _family_for_eval(eval_name)
        for _, row in picked.iterrows():
            sid = str(row["sample_id"])
            samples.append(
                {
                    "id": f"{eval_name}.golden.{sid.split('.')[-1] if '.' in sid else sid}",
                    "sample_id": sid,
                    "prompt": _prompt_text(str(row["prompt"])),
                    "task_type": family,
                    "eval_name": eval_name,
                    "expected_behavior": cfg["expected_behavior"],
                    "rubric": cfg["rubric"],
                    "success_criteria": cfg["success_criteria"],
                    "scorer": cfg["scorer"],
                    "judge_protocol": cfg["judge_protocol"],
                    "source_split": "routerbench_golden_holdout",
                    "reference_answer": None,
                    "oracle_model": str(row.get("oracle_model_to_route_to", "")) if pd.notna(row.get("oracle_model_to_route_to")) else None,
                    "failure_tags": [],
                    "notes": "Auto-sampled v1; reference_answer to be filled during manual audit.",
                }
            )

    doc = {
        "schema_version": "1.0",
        "name": "RouteAlpha Golden Set v1",
        "description": "Held-out audit set for eval protocol and routing decision quality. Never use for training or hyperparameter tuning.",
        "protocol": {
            "data_discipline": {
                "train": "RouterBench minus golden/RouterArena",
                "calibration": "Separate held-out slice; never train",
                "test": "RouterBench test slice for backtest",
                "golden": "This file only; audit and failure attribution",
                "routerarena": "Final leaderboard only; no training",
            },
            "judge_debias": {
                "method": "swap_and_aggregate",
                "description": "For llm_judge samples: run dual judges; swap A/B order; accept only if consistent.",
                "metrics": ["position_flip_rate", "disputed_rate"],
                "implementations_note": "Judge models configurable (e.g. Kimi + Fable); v1 may use RouterBench 0/1 labels for deterministic scorers.",
            },
            "inspired_by": [
                {"repo": "promptfoo/promptfoo", "use": "versioned regression eval cases"},
                {"repo": "confident-ai/deepeval", "use": "pytest-style metric modules"},
                {"repo": "EleutherAI/lm-evaluation-harness", "use": "task adapter and metric naming"},
                {"repo": "LesterALeong/llm-evalgate", "use": "deterministic gate before LLM judge"},
                {"repo": "macamiri/judgebias", "use": "position flip rate diagnostics"},
                {"repo": "DaoyuanLi2816/pairjudge", "use": "swap_debias naming"},
            ],
        },
        "rubrics_by_task_family": {
            fam: {"eval_examples": names, **cfg}
            for fam, names, cfg in TASK_FAMILY_RULES
        },
        "samples": samples,
        "stats": {
            "total_samples": len(samples),
            "eval_name_count": len({s["eval_name"] for s in samples}),
            "deterministic_count": sum(1 for s in samples if s["success_criteria"] == "deterministic"),
            "llm_judge_count": sum(1 for s in samples if s["success_criteria"] == "llm_judge"),
        },
    }

    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {OUT}")


if __name__ == "__main__":
    main()
