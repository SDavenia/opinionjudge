"""
Generate the synthetic opinions.

valueprism: one opinion per sampled ValuePrism row, written from the situation, the target
stance (Supports / Opposes / Either), the value and its explanation.
habermas_hybrid: every human Habermas opinion, plus one synthetic opinion per situation
and stance (agree / disagree / neutral); score.py pairs each synthetic opinion with the
human ones of its situation.

Output: {output_dir}/{dataset_id}/{generation_model_id}_{generation_prompt_version}.csv

Usage:
    python generate_opinions.py --dataset_id valueprism --generation_model_id llama-3.3-70b
    python generate_opinions.py --dataset_id habermas_hybrid --generation_model_id llama-3.3-70b
"""
import argparse
from pathlib import Path

import pandas as pd

from utils.data_utils import DEFAULT_GENERATION_PROMPT, situations_path
from utils.generation_utils import generate
from utils.models_utils import REGISTRY, load_model
from utils.paths import GENERATIONS_DIR
from utils.prompts_utils import GENERATION_PROMPTS, GENERATION_PROMPTS_BY_DATASET, REFUSAL_STANCE_INSTRUCTIONS

# Synthetic Habermas stance group -> (agreement label, stance instruction).
HYBRID_STANCES = {"agree": ("AGREE", "support"), "disagree": ("DISAGREE", "oppose"), "neutral": ("NEUTRAL", "neutral")}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_id", required=True, choices=list(GENERATION_PROMPTS_BY_DATASET))
    parser.add_argument("--generation_model_id", required=True, choices=list(REGISTRY))
    parser.add_argument("--generation_prompt_version", default=None, choices=list(GENERATION_PROMPTS),
                        help="Default: reflective_person (valueprism), hb_reflective_person (habermas_hybrid).")
    parser.add_argument("--batch_size", type=int, default=32, help="Divided by the model's batch_size_divide.")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_situations", type=int, default=None, help="Only the first N situations, for tests.")
    parser.add_argument("--output_dir", type=Path, default=GENERATIONS_DIR)
    args = parser.parse_args()
    if args.generation_prompt_version is None:
        args.generation_prompt_version = DEFAULT_GENERATION_PROMPT[args.dataset_id]
    elif args.generation_prompt_version not in GENERATION_PROMPTS_BY_DATASET[args.dataset_id]:
        parser.error(f"--generation_prompt_version must be one of {GENERATION_PROMPTS_BY_DATASET[args.dataset_id]}.")
    return args


def valueprism_prompt(row, prompt_version):
    value = row["vp_vrd_value"]
    if row["stance"] == "Supports":
        instruction = f"Write an opinion that firmly ADVOCATES FOR or JUSTIFIES the action in the situation, using the principle of {value}."
    elif row["stance"] == "Opposes":
        instruction = f"Write an opinion that firmly CRITICIZES or ARGUES AGAINST the action in the situation, using the principle of {value}."
    else:
        instruction = f"Write an opinion that highlights the AMBIGUITY, CONDITIONALITY, or internal tension of the situation regarding the principle of {value}."
    return GENERATION_PROMPTS[prompt_version].format(situation=row["situation"], value=value, valence=row["stance"],
                                                     explanation=row["vp_explanation"], stance_instruction=instruction)


def limit_situations(df, n):
    if n is None:
        return df
    return df[df["situation_id"].isin(df["situation_id"].drop_duplicates().head(n))]


def generate_valueprism(args, model, proc, spec):
    df = limit_situations(pd.read_csv(situations_path("valueprism")), args.num_situations)
    prompts = [valueprism_prompt(row, args.generation_prompt_version) for _, row in df.iterrows()]
    df["text"] = generate(model, proc, spec, prompts, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                          temperature=args.temperature, top_p=args.top_p, desc=f"Generating ({args.generation_model_id})")
    df["generation_model"] = args.generation_model_id
    df["generation_prompt_version"] = args.generation_prompt_version
    return df


def generate_habermas_hybrid(args, model, proc, spec):
    human = limit_situations(pd.read_csv(situations_path("habermas")), args.num_situations)
    template = GENERATION_PROMPTS[args.generation_prompt_version]
    rows = []
    for _, sit in human.drop_duplicates("situation_id")[["situation_id", "situation"]].iterrows():
        for group, (stance, instruction) in HYBRID_STANCES.items():
            rows.append({"situation_id": sit["situation_id"],
                         "text_id": f"HBS_{sit['situation_id'].removeprefix('HB_')}_{group}",
                         "situation": sit["situation"], "stance": stance, "stance_group": group,
                         "prompt": template.format(situation=sit["situation"],
                                                   stance_instruction=REFUSAL_STANCE_INSTRUCTIONS["habermas"][instruction])})
    synthetic = pd.DataFrame(rows)
    synthetic["text"] = generate(model, proc, spec, synthetic["prompt"].tolist(), max_new_tokens=args.max_new_tokens,
                                 batch_size=args.batch_size, temperature=args.temperature, top_p=args.top_p,
                                 desc=f"Generating ({args.generation_model_id})")
    synthetic = synthetic.drop(columns="prompt").assign(origin="synthetic", generation_model=args.generation_model_id,
                                                        generation_prompt_version=args.generation_prompt_version)
    human = human.assign(origin="human", generation_model="human", generation_prompt_version=None)
    hybrid = pd.concat([human, synthetic], ignore_index=True)
    hybrid["dataset"] = "habermas_hybrid"
    # Human opinions first, then the synthetic ones, each grouped by situation.
    order = (hybrid["origin"] == "synthetic").astype(int)
    return hybrid.assign(_order=order).sort_values(["_order", "situation_id"], kind="stable").drop(columns="_order")


def main():
    args = parse_args()
    spec = REGISTRY[args.generation_model_id]
    model, proc = load_model(spec)
    build = generate_valueprism if args.dataset_id == "valueprism" else generate_habermas_hybrid
    df = build(args, model, proc, spec)
    output_path = args.output_dir / args.dataset_id / f"{args.generation_model_id}_{args.generation_prompt_version}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} opinions to {output_path}")


if __name__ == "__main__":
    main()
