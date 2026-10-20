"""The judge's own position on each situation: n samples per alignment prompt, parsed into the
prompt's labels, with the label distribution per situation."""
import json
import re

import pandas as pd

from utils.data_utils import SITUATION_DATASETS, load_situations
from utils.generation_utils import generate
from utils.prompts_utils import ALIGNMENT_PROMPTS, ALIGNMENT_PROMPTS_BY_DATASET, LABEL_MAPPING

DATASETS = SITUATION_DATASETS


def output_dir(dataset_id, judge, args):
    return args.features_dir / "alignment" / dataset_id / judge


def done(dataset_id, judge, args):
    return all((output_dir(dataset_id, judge, args) / f"{p}.csv").exists()
               for p in ALIGNMENT_PROMPTS_BY_DATASET[dataset_id])


def parse_labels(generations, prompt_version):
    labels = LABEL_MAPPING[prompt_version]
    # Longest label first, so that e.g. "I WOULD NOT" wins over "I WOULD".
    pattern = re.compile(r"\b(" + "|".join(re.escape(l) for l in sorted(labels, key=len, reverse=True)) + r")\b",
                         re.IGNORECASE)
    canonical = {label.upper(): label for label in labels}
    parsed = []
    for generation in generations:
        match = pattern.search(generation)
        parsed.append(canonical.get(match.group(1).upper(), match.group(1).upper()) if match else None)
    return parsed


def label_distribution(situations, parsed, labels):
    rows = []
    for situation_id, situation in dict.fromkeys(situations):
        evaluations = [p for s, p in zip(situations, parsed) if s == (situation_id, situation)]
        counts = {label: evaluations.count(label) for label in labels}
        total = sum(counts.values())
        probs = {f"p({label})": (counts[label] / total if total else 0.0) for label in labels}
        rows.append({"situation_id": situation_id, "situation": situation,
                     "alignment_distribution": json.dumps({**probs, "total_evaluations": total})})
    return pd.DataFrame(rows)


def extract(model, proc, spec, judge, dataset_id, args):
    situations = load_situations(dataset_id, limit=args.limit)
    repeated = [s for s in situations for _ in range(args.alignment_samples)]
    out_dir = output_dir(dataset_id, judge, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    for prompt_version in ALIGNMENT_PROMPTS_BY_DATASET[dataset_id]:
        path = out_dir / f"{prompt_version}.csv"
        if path.exists() and not args.overwrite:
            continue
        prompts = [ALIGNMENT_PROMPTS[prompt_version].format(situation=situation) for _, situation in repeated]
        generations = generate(model, proc, spec, prompts, max_new_tokens=args.alignment_max_new_tokens,
                               batch_size=args.batch_size, temperature=1.0, top_p=1.0, top_k=0,
                               desc=f"Alignment {prompt_version} ({judge})")
        parsed = parse_labels(generations, prompt_version)
        pd.DataFrame({"situation": [s for _, s in repeated], "generation": generations,
                      "parsed_evaluations": parsed}).to_csv(out_dir / f"{prompt_version}_generations.csv", index=False)
        label_distribution(repeated, parsed, LABEL_MAPPING[prompt_version]).to_csv(path, index=False)
        print(f"Saved {path}")
