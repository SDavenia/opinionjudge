"""
Extract the features used in the analyses, by default all of them, for every dataset and
every open judge model.

Judge features (computed under each judge, open models only):
    perplexity  perplexity of each opinion
    alignment   the judge's own position on each situation
    refusal     the judge's probability of refusing to write each stance on each situation,
                re-classified by --refusal_classifier
Opinion features:
    safety      Llama-Guard unsafe score
    sentiment   RoBERTa sentiment score
    vad         valence, arousal and dominance
    vader       VADER sentiment
    embeddings  sentence embedding

Situation-level features (alignment, refusal) exist for valueprism and habermas only:
habermas_hybrid uses the habermas ones. For habermas_hybrid, opinion features are computed
on its synthetic opinions only, the human ones being the habermas opinions.

Outputs already present are skipped unless --overwrite is given.

Usage:
    python extract_features.py
    python extract_features.py --features perplexity alignment --models gemma3 qwen2.5-72b
    python extract_features.py --features sentiment vader --datasets habermas_hybrid
    python extract_features.py --features refusal --models qwen2.5-05b --datasets habermas --limit 5
"""
import argparse
import gc
from pathlib import Path

import torch

from feature_scripts import JUDGE_FEATURES, TEXT_FEATURES, refusal
from utils.analysis_utils import DEFAULT_EMBEDDING_MODEL
from utils.data_utils import DATASETS, DEFAULT_GENERATOR
from utils.models_utils import REGISTRY, load_model
from utils.paths import FEATURES_DIR
from utils.refusal_utils import CLASSIFY_SCOPES

FEATURES = {**JUDGE_FEATURES, **TEXT_FEATURES}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", nargs="+", default=list(FEATURES), choices=list(FEATURES))
    parser.add_argument("--models", nargs="+", default=list(REGISTRY),
                        help="Judges for the judge features (default: every open model).")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--generation_model_id", default=DEFAULT_GENERATOR,
                        help="Generator of the synthetic opinions (valueprism, habermas_hybrid).")
    parser.add_argument("--features_dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--batch_size", type=int, default=32, help="Divided by the model's batch_size_divide.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N situations / opinions, for tests.")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--perplexity_batch_size", type=int, default=2)
    parser.add_argument("--perplexity_max_length", type=int, default=512, help="Max tokens of prompt + opinion.")
    parser.add_argument("--alignment_samples", type=int, default=30, help="Samples per situation and prompt.")
    parser.add_argument("--alignment_max_new_tokens", type=int, default=20)
    parser.add_argument("--refusal_samples", type=int, default=30, help="Samples per situation and stance.")
    parser.add_argument("--refusal_max_new_tokens", type=int, default=50)
    parser.add_argument("--refusal_classifier", default="llama-3.3-70b", choices=list(REGISTRY))
    parser.add_argument("--classify_scope", default="flagged", choices=CLASSIFY_SCOPES,
                        help="Re-classify only the completions flagged by substring matching, or all of them.")
    parser.add_argument("--embedding_model_id", default=DEFAULT_EMBEDDING_MODEL)
    return parser.parse_args()


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_judge_features(args, names):
    for model_id in args.models:
        if model_id not in REGISTRY:
            print(f"[skip] {model_id}: judge features are only computed for open models.")
            continue
        jobs = [(name, dataset) for name in names for dataset in args.datasets
                if dataset in JUDGE_FEATURES[name].DATASETS
                and (args.overwrite or not JUDGE_FEATURES[name].done(dataset, model_id, args))]
        if not jobs:
            continue
        spec = REGISTRY[model_id]
        model, proc = load_model(spec)
        for name, dataset in jobs:
            print(f"== {name} | {model_id} | {dataset}")
            JUDGE_FEATURES[name].extract(model, proc, spec, model_id, dataset, args)
        del model, proc
        free_memory()


def run_refusal_classification(args):
    jobs = [(model_id, dataset) for model_id in args.models if model_id in REGISTRY
            for dataset in args.datasets if dataset in refusal.DATASETS
            and refusal.done(dataset, model_id, args) and (args.overwrite or not refusal.classified(dataset, model_id, args))]
    if not jobs:
        return
    spec = REGISTRY[args.refusal_classifier]
    model, proc = load_model(spec)
    for model_id, dataset in jobs:
        print(f"== refusal classification | {model_id} | {dataset}")
        refusal.classify(model, proc, spec, model_id, dataset, args)
    del model, proc
    free_memory()


def run_text_features(args, names):
    for name in names:
        module = TEXT_FEATURES[name]
        datasets = [d for d in args.datasets if args.overwrite or not module.done(d, args)]
        if not datasets:
            continue
        scorer = module.load(args)
        for dataset in datasets:
            print(f"== {name} | {dataset}")
            module.extract(scorer, dataset, args)
        del scorer
        free_memory()


def main():
    args = parse_args()
    judge_names = [f for f in args.features if f in JUDGE_FEATURES]
    if judge_names:
        run_judge_features(args, judge_names)
    if "refusal" in judge_names:
        run_refusal_classification(args)
    run_text_features(args, [f for f in args.features if f in TEXT_FEATURES])


if __name__ == "__main__":
    main()
