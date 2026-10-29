#!/usr/bin/env python3
"""Compare deterministic text output and first-token logprobs for two servers."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from urllib.request import Request, urlopen


PROMPTS = [
    "用一句话解释为什么天空通常呈蓝色。",
    "计算 17 * 23，并只给出结果。",
    "Write a Python function that returns the larger of two numbers.",
    "A box contains 3 red and 2 blue balls. What is the probability of drawing a red ball?",
    "请把这句话翻译成英文：性能优化必须同时验证精度。",
    "List three practical ways to reduce neural-network inference latency.",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain briefly.",
    "续写一句：人工智能的价值不只在于速度，还在于",
]


def request(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": max_tokens,
            "logprobs": 5,
            "seed": 20260915,
        }
    ).encode()
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=180) as response:
        return json.load(response)["choices"][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=27)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for prompt in PROMPTS:
        baseline = request(args.baseline_url, args.model, prompt, args.max_tokens)
        candidate = request(args.candidate_url, args.model, prompt, args.max_tokens)
        b_tokens = baseline.get("logprobs", {}).get("tokens", [])
        c_tokens = candidate.get("logprobs", {}).get("tokens", [])
        matched = sum(a == b for a, b in zip(b_tokens, c_tokens))
        denom = max(len(b_tokens), len(c_tokens), 1)

        b_top = (baseline.get("logprobs", {}).get("top_logprobs") or [{}])[0]
        c_top = (candidate.get("logprobs", {}).get("top_logprobs") or [{}])[0]
        b_keys, c_keys = set(b_top), set(c_top)
        rows.append(
            {
                "prompt": prompt,
                "baseline_text": baseline["text"],
                "candidate_text": candidate["text"],
                "exact_text_match": baseline["text"] == candidate["text"],
                "text_similarity": SequenceMatcher(
                    None, baseline["text"], candidate["text"]
                ).ratio(),
                "token_position_match_rate": matched / denom,
                "first_token_match": bool(b_tokens and c_tokens and b_tokens[0] == c_tokens[0]),
                "first_token_top5_overlap": len(b_keys & c_keys) / max(len(b_keys | c_keys), 1),
                "baseline_finish_reason": baseline.get("finish_reason"),
                "candidate_finish_reason": candidate.get("finish_reason"),
            }
        )

    payload = {
        "method": "temperature=0, seed fixed; 8 bilingual/reasoning prompts; top-5 first-token logprobs",
        "summary": {
            "prompts": len(rows),
            "exact_text_match_rate": sum(r["exact_text_match"] for r in rows) / len(rows),
            "mean_text_similarity": sum(r["text_similarity"] for r in rows) / len(rows),
            "mean_token_position_match_rate": sum(
                r["token_position_match_rate"] for r in rows
            )
            / len(rows),
            "first_token_match_rate": sum(r["first_token_match"] for r in rows)
            / len(rows),
            "mean_first_token_top5_overlap": sum(
                r["first_token_top5_overlap"] for r in rows
            )
            / len(rows),
        },
        "cases": rows,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

