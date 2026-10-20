"""
Sample the opinion pairs of the human symmetry study: --n_valueprism ValuePrism pairs and
--n_habermas Habermas pairs, built as in score.py so that the judges have scored all of them.
Within each dataset: one pair per situation, stratified over the (unordered) stance
combination of the two opinions; Habermas opinions limited to --min_words..--max_words.
"1to2" is the order of the scoring files (A = text_1, B = text_2).

Usage:
    python -m human_annotation.symmetric.sample_pairs
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_utils import opinions_path
from utils.scoring_utils import build_pairs

HERE = Path(__file__).resolve().parent
STANCE_COLUMN = {"valueprism": "stance", "habermas": "stance_group"}
CONTEXT_LABEL = {"valueprism": "Situation", "habermas": "Statement under discussion"}
PAIR_PREFIX = {"valueprism": "VP", "habermas": "HB"}
PAIR_COLUMNS = ["pair_id", "dataset", "situation_id", "situation", "context_label", "stance_combo",
                "id_1", "text_id_1", "stance_1", "text_1", "id_2", "text_id_2", "stance_2", "text_2"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_valueprism", type=int, default=25)
    parser.add_argument("--n_habermas", type=int, default=25)
    parser.add_argument("--valueprism_csv", type=Path, default=opinions_path("valueprism"))
    parser.add_argument("--habermas_csv", type=Path, default=opinions_path("habermas"))
    parser.add_argument("--min_words", type=int, default=20, help="Habermas only.")
    parser.add_argument("--max_words", type=int, default=150, help="Habermas only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=Path, default=HERE / "samples" / "pairs.csv")
    return parser.parse_args()


def load_pairs(dataset_id, source, args):
    pairs = build_pairs(pd.read_csv(source))
    stance_col = STANCE_COLUMN[dataset_id]
    pairs["stance_1"], pairs["stance_2"] = pairs[f"{stance_col}_1"], pairs[f"{stance_col}_2"]
    pairs["stance_combo"] = ["-".join(sorted(s)) for s in zip(pairs["stance_1"], pairs["stance_2"])]
    pairs["situation_id"] = pairs["situation_id_1"]
    if dataset_id == "habermas":
        keep = pairs["text_1"].notna() & pairs["text_2"].notna()
        for col in ("text_1", "text_2"):
            keep &= pairs[col].astype(str).str.split().str.len().between(args.min_words, args.max_words)
        print(f"habermas length filter: {keep.sum()} of {len(pairs)} pairs kept.")
        pairs = pairs[keep]
    for col in ("text_1", "text_2"):
        pairs[col] = pairs[col].astype(str).str.strip()
    return pairs.reset_index(drop=True)


def stratified_sample(pairs, n_pairs, rng):
    """n_pairs pairs, as even as possible over stance_combo, at most one per situation."""
    combos = sorted(pairs["stance_combo"].unique())
    quota = {c: n_pairs // len(combos) for c in combos}
    for c in rng.permutation(combos)[: n_pairs % len(combos)]:
        quota[c] += 1
    used, chosen = set(), []
    # The rarest combination first: it has the fewest situations to choose from.
    for combo in sorted(combos, key=lambda c: pairs.loc[pairs["stance_combo"] == c, "situation_id"].nunique()):
        candidates = pairs[pairs["stance_combo"] == combo]
        candidates = candidates.iloc[rng.permutation(len(candidates))]
        candidates = candidates[~candidates["situation_id"].isin(used)].drop_duplicates("situation_id")
        if len(candidates) < quota[combo]:
            raise ValueError(f"Only {len(candidates)} situations left for {combo}, {quota[combo]} needed.")
        picked = candidates.head(quota[combo])
        used.update(picked["situation_id"])
        chosen.append(picked)
    sample = pd.concat(chosen)
    return sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)


def main():
    args = parse_args()
    samples = []
    for k, (dataset_id, n_pairs, source) in enumerate([("valueprism", args.n_valueprism, args.valueprism_csv),
                                                       ("habermas", args.n_habermas, args.habermas_csv)]):
        if n_pairs <= 0:
            continue
        sample = stratified_sample(load_pairs(dataset_id, source, args), n_pairs, np.random.default_rng([args.seed, k]))
        sample["pair_id"] = [f"{PAIR_PREFIX[dataset_id]}-P{i + 1:03d}" for i in range(len(sample))]
        sample["dataset"] = dataset_id
        sample["context_label"] = CONTEXT_LABEL[dataset_id]
        samples.append(sample[PAIR_COLUMNS])
        print(f"{dataset_id}: {len(sample)} pairs\n{sample['stance_combo'].value_counts().to_string()}")
    out = pd.concat(samples, ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved {len(out)} pairs to {args.output_csv}")


if __name__ == "__main__":
    main()
