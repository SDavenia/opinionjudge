"""
Score every same-situation opinion pair with a local judge model: for each pair, each
prompt variation of --scoring_prompt_version and both orders of the two opinions.

Output: {output_dir}/{dataset_id}/open/[{generator}_{prompt}_]{judge_model_id}_{scoring_prompt_version}.csv

Usage:
    python score.py --dataset_id habermas --judge_model_id gemma3
    python score.py --dataset_id valueprism --judge_model_id qwen2.5-72b --generation_model_id llama-3.3-70b
"""
import argparse

import torch
from tqdm import tqdm

from utils.generation_utils import batches, chat_text, effective_batch_size, generate_batch, sampling_kwargs
from utils.models_utils import REGISTRY, load_model, tokenizer_of
from utils.scoring_utils import (
    add_scoring_args, build_direction_prompts, check_scoring_args, expand_pairs_for_variations, fill_scores,
    load_generation_df, resolve_output_path, select_pairs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge_model_id", required=True, choices=list(REGISTRY))
    add_scoring_args(parser)
    parser.add_argument("--batch_size", type=int, default=32, help="Divided by the model's batch_size_divide.")
    parser.add_argument("--save_every", type=int, default=1000,
                        help="Save the partial results every N generations (0: only at the end).")
    args = parser.parse_args()
    check_scoring_args(parser, args)
    return args


def main():
    args = parse_args()
    gen_df = load_generation_df(args)
    pairs_df = select_pairs(args, gen_df)

    spec = REGISTRY[args.judge_model_id]
    model, proc = load_model(spec)
    tok = tokenizer_of(proc, spec)
    output_path = resolve_output_path(args, "open")

    prompts = build_direction_prompts(pairs_df, args.scoring_prompt_version)
    texts = [chat_text(tok, spec, p) for p in prompts]
    # One row per (pair, prompt variation): generations 2i and 2i + 1 are its two orders.
    pairs_df = expand_pairs_for_variations(pairs_df, args.scoring_prompt_version)
    generations = [None] * len(texts)
    size = effective_batch_size(args.batch_size, spec)
    print(f"Judge {spec.name} | {len(pairs_df)} pair x variation rows | {len(texts)} prompts | batch size {size}")

    def save():
        fill_scores(pairs_df, generations, args, spec.name, decimals=2).to_csv(output_path, index=False)

    n_done = 0
    for batch in tqdm(list(batches(texts, size)), desc=f"Scoring ({args.judge_model_id})"):
        generations[n_done:n_done + len(batch)] = generate_batch(model, proc, spec, batch, args.max_new_tokens,
                                                                 sampling_kwargs(0.0))
        if args.save_every > 0 and (n_done + len(batch)) // args.save_every > n_done // args.save_every:
            save()
        n_done += len(batch)

    save()
    print(f"Saved {len(pairs_df)} rows to {output_path}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: max allocated {torch.cuda.max_memory_allocated(i) / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    main()
