"""The judge's probability of refusing to write an opinion with a given stance on each
situation. Completions are flagged by substring matching, and the flagged ones are then
re-classified by an LLM (classify), since many of them are opinions against the action
("I can't support ...") rather than refusals."""
import pandas as pd

from utils.data_utils import SITUATION_DATASETS, load_situations
from utils.generation_utils import generate
from utils.prompts_utils import REFUSAL_STANCES
from utils.refusal_utils import (
    aggregate_refusal_probabilities, apply_classifier_labels, build_classifier_prompts, build_generations_df,
    build_sampling_plan, summarize_classification, summarize_refusal_matches,
)

DATASETS = SITUATION_DATASETS


def output_dir(dataset_id, judge, args):
    return args.features_dir / "refusal" / dataset_id / judge


def classified_dir(dataset_id, judge, args):
    return output_dir(dataset_id, judge, args) / "refusal_classified" / args.refusal_classifier


def done(dataset_id, judge, args):
    return (output_dir(dataset_id, judge, args) / "refusal_generations.csv").exists()


def classified(dataset_id, judge, args):
    return (classified_dir(dataset_id, judge, args) / "refusal_prob.csv").exists()


def extract(model, proc, spec, judge, dataset_id, args):
    situations = load_situations(dataset_id, limit=args.limit)
    records = build_sampling_plan(situations, dataset_id, args.refusal_samples)
    print(f"{len(situations)} situations x {len(REFUSAL_STANCES)} stances x {args.refusal_samples} samples "
          f"= {len(records)} completions.")
    generations = generate(model, proc, spec, [r["prompt"] for r in records],
                           max_new_tokens=args.refusal_max_new_tokens, batch_size=args.batch_size,
                           temperature=1.0, top_p=1.0, desc=f"Refusal ({judge}, {dataset_id})")
    generations_df = build_generations_df(records, generations)
    out_dir = output_dir(dataset_id, judge, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_refusal_probabilities(generations_df).to_csv(out_dir / "refusal_prob.csv", index=False)
    generations_df.to_csv(out_dir / "refusal_generations.csv", index=False)
    print(f"Saved {out_dir}/refusal_{{prob,generations}}.csv")
    print("\n".join(summarize_refusal_matches(generations_df)))


def classify(model, proc, spec, judge, dataset_id, args):
    generations_df = pd.read_csv(output_dir(dataset_id, judge, args) / "refusal_generations.csv")
    row_indices, prompts = build_classifier_prompts(generations_df, dataset_id, args.classify_scope)
    print(f"{judge}/{dataset_id}: {len(prompts)} of {len(generations_df)} completions to classify.")
    raw = generate(model, proc, spec, prompts, max_new_tokens=16, batch_size=args.batch_size, temperature=0.0,
                   desc=f"Classifying refusals ({judge}, {dataset_id})") if prompts else []
    classified_df = apply_classifier_labels(generations_df, row_indices, raw)
    out_dir = classified_dir(dataset_id, judge, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    classified_df.to_csv(out_dir / "refusal_generations.csv", index=False)
    aggregate_refusal_probabilities(classified_df).to_csv(out_dir / "refusal_prob.csv", index=False)
    print(f"Saved {out_dir}/refusal_{{prob,generations}}.csv")
    print("\n".join(summarize_classification(classified_df)))
