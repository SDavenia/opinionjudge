"""
Score every same-situation opinion pair with a judge served by OpenRouter (same prompts and
output format as score.py). Needs OPENROUTER_API_KEY, e.g. in a .env file.

Output: {output_dir}/{dataset_id}/closed/[{generator}_{prompt}_]{judge_model_id}_{scoring_prompt_version}.csv

Usage:
    python score_openrouter.py --dataset_id habermas --judge_model_id openai/gpt-5.4-nano
    python score_openrouter.py --dataset_id habermas --judge_model_id openai/gpt-5.4-nano --num_examples 2
"""
import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tqdm import tqdm

from utils.scoring_utils import (
    add_scoring_args, build_direction_prompts, check_scoring_args, expand_pairs_for_variations, fill_scores,
    load_generation_df, parse_generation_scoring, resolve_output_path, select_pairs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge_model_id", required=True, help="OpenRouter model slug, e.g. openai/gpt-5.4-nano.")
    add_scoring_args(parser)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_concurrency", type=int, default=8, help="Requests in flight.")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=50, help="Save every N completed requests.")
    args = parser.parse_args()
    check_scoring_args(parser, args)
    return args


def make_client():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (add it to a .env file).")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def call_openrouter(client, model, prompt, max_tokens, temperature, max_retries):
    """One chat completion, with exponential backoff on transient errors ("" if it keeps failing)."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                                  max_tokens=max_tokens, temperature=temperature)
            content = (resp.choices[0].message.content or "").strip()
            if not content and attempt == 0:
                print(f"[warn] empty answer (finish reason: {resp.choices[0].finish_reason})")
            return content
        except (RateLimitError, APITimeoutError, APIError) as e:
            if attempt == max_retries - 1:
                print(f"[error] giving up after {max_retries} attempts: {e}")
                return ""
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 30)
    return ""


def main():
    args = parse_args()
    client = make_client()
    gen_df = load_generation_df(args)
    pairs_df = select_pairs(args, gen_df)
    output_path = resolve_output_path(args, "closed")

    prompts = build_direction_prompts(pairs_df, args.scoring_prompt_version)
    pairs_df = expand_pairs_for_variations(pairs_df, args.scoring_prompt_version)
    print(f"Judge {args.judge_model_id}: {len(prompts)} OpenRouter requests "
          f"({len(pairs_df)} pair x variation rows x 2 orders).")
    generations = [None] * len(prompts)

    def save():
        fill_scores(pairs_df, generations, args, args.judge_model_id, decimals=1).to_csv(output_path, index=False)

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        futures = {pool.submit(call_openrouter, client, args.judge_model_id, prompt, args.max_new_tokens,
                               args.temperature, args.max_retries): i for i, prompt in enumerate(prompts)}
        for completed, future in enumerate(tqdm(as_completed(futures), total=len(futures),
                                                desc=f"Scoring ({args.judge_model_id})"), start=1):
            idx = futures[future]
            try:
                generations[idx] = future.result()
            except Exception as e:
                print(f"[error] prompt {idx} failed: {e}")
                generations[idx] = ""
            if completed <= 3:
                parsed = parse_generation_scoring(generations[idx], option_setting=args.option_setting)
                print(f"[sample {idx}] {generations[idx]!r} -> {parsed}")
            if completed % args.checkpoint_every == 0:
                save()

    save()
    print(f"Saved {len(pairs_df)} rows to {output_path}")
    print(f"{sum(g == '' for g in generations)} of {len(generations)} requests returned no answer.")


if __name__ == "__main__":
    main()
