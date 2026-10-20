"""
Build each annotator's items for the symmetry study. Every pair is rated in both orders:
block 1 has the annotator's pairs shuffled, each in a random order of the two opinions;
block 2 has the same pairs in the other order, reshuffled so that the two ratings of a pair
are at least --min_gap items apart. --n_retest adds exact repeats of block-1 items in block 2.

--design shared (default): every annotator rates the same pairs; disjoint: annotator k rates
pairs [k * n, (k + 1) * n) of each dataset. Annotator k uses np.random.default_rng([seed, k]).

Outputs: assignments/for_annotators/{annotator}.csv (sent to the annotators, without anything
that reveals the pair or the order) and assignments/key.csv (kept: item -> pair, order).

Usage:
    python -m human_annotation.symmetric.assign_annotators --n_annotators 5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANNOTATOR_COLUMNS = ["annotator_id", "dataset", "position", "item_id", "context_label", "situation", "opinion_a",
                     "opinion_b"]
KEY_COLUMNS = ["annotator_id", "dataset", "position", "item_id", "pair_id", "direction", "exposure", "role"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs_csv", type=Path, default=HERE / "samples" / "pairs.csv")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n_annotators", type=int, help="Annotators named ANN1..ANNn.")
    group.add_argument("--annotators", nargs="+", help="Annotator ids.")
    parser.add_argument("--design", choices=["shared", "disjoint"], default="shared")
    parser.add_argument("--n_per_dataset", type=int, default=25, help="Pairs per dataset and annotator.")
    parser.add_argument("--min_gap", type=int, default=20, help="Minimum number of items between two ratings of a pair.")
    parser.add_argument("--n_retest", type=int, default=0, help="Exact repeats per annotator.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", type=Path, default=HERE / "assignments")
    return parser.parse_args()


def spaced_block2(block1_order, min_gap, rng):
    """Block 1's pairs reshuffled so that each comes at least min_gap items after its first rating."""
    n = len(block1_order)
    first_pos = {p: i for i, p in enumerate(block1_order)}
    remaining, block2 = list(block1_order), []
    for j in range(n):
        allowed = [p for p in remaining if (n - first_pos[p]) + j >= min_gap]
        pick = allowed[rng.integers(len(allowed))]
        block2.append(pick)
        remaining.remove(pick)
    return block2


def build_sequence(pair_ids, min_gap, n_retest, rng):
    n = len(pair_ids)
    if min_gap > n:
        raise ValueError(f"--min_gap {min_gap} is larger than the number of pairs per annotator ({n}).")
    block1 = [pair_ids[i] for i in rng.permutation(n)]
    first_dir = {p: rng.choice(["1to2", "2to1"]) for p in block1}
    flip = {"1to2": "2to1", "2to1": "1to2"}
    block2 = spaced_block2(block1, min_gap, rng)
    seq = ([{"pair_id": p, "direction": first_dir[p], "exposure": 1, "role": "main"} for p in block1]
           + [{"pair_id": p, "direction": flip[first_dir[p]], "exposure": 2, "role": "main"} for p in block2])
    for p in rng.choice(block1, size=n_retest, replace=False):
        seen = [k for k, item in enumerate(seq) if item["pair_id"] == p]
        slots = np.arange(n, len(seq) + 1)
        distance = np.min(np.abs(slots[:, None] - np.array(seen)[None, :]), axis=1)
        ok = slots[distance >= min_gap]
        slot = int(rng.choice(ok)) if len(ok) else int(slots[np.argmax(distance)])
        seq.insert(slot, {"pair_id": p, "direction": first_dir[p], "exposure": 3, "role": "retest"})
    return seq


def main():
    args = parse_args()
    pairs = pd.read_csv(args.pairs_csv).set_index("pair_id", drop=False)
    annotators = args.annotators or [f"ANN{k + 1}" for k in range(args.n_annotators)]
    n = args.n_per_dataset
    needed = n * (len(annotators) if args.design == "disjoint" else 1)
    by_dataset = {d: g["pair_id"] for d, g in pairs.groupby("dataset", sort=False)}
    for dataset_id, ids in by_dataset.items():
        if len(ids) < needed:
            raise ValueError(f"{len(ids)} {dataset_id} pairs, {needed} needed: re-run sample_pairs.")

    (args.output_dir / "for_annotators").mkdir(parents=True, exist_ok=True)
    keys = []
    for k, annotator_id in enumerate(annotators):
        rng = np.random.default_rng([args.seed, k])
        start = k * n if args.design == "disjoint" else 0
        share = pd.concat([ids.iloc[start:start + n] for ids in by_dataset.values()])
        rows = []
        for position, item in enumerate(build_sequence(list(share), args.min_gap, args.n_retest, rng), start=1):
            pair = pairs.loc[item["pair_id"]]
            a, b = ("text_1", "text_2") if item["direction"] == "1to2" else ("text_2", "text_1")
            item_id = f"{rng.integers(16 ** 8):08x}"
            rows.append({"annotator_id": annotator_id, "dataset": pair["dataset"], "position": position,
                         "item_id": item_id, "context_label": pair["context_label"], "situation": pair["situation"],
                         "opinion_a": pair[a], "opinion_b": pair[b]})
            keys.append({"annotator_id": annotator_id, "dataset": pair["dataset"], "position": position,
                         "item_id": item_id, **item})
        path = args.output_dir / "for_annotators" / f"{annotator_id}.csv"
        pd.DataFrame(rows, columns=ANNOTATOR_COLUMNS).to_csv(path, index=False)
        print(f"{annotator_id}: {len(rows)} items -> {path}")

    key = pd.DataFrame(keys, columns=KEY_COLUMNS)
    key.to_csv(args.output_dir / "key.csv", index=False)
    pd.Series({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}).to_json(
        args.output_dir / "assignment_config.json", indent=2)
    main_items = key[key["role"] == "main"].pivot_table(index=["annotator_id", "pair_id"], columns="exposure",
                                                        values="position")
    gaps = main_items[2] - main_items[1]
    print(f"Gap between the two orders of a pair: min {gaps.min()}, median {gaps.median():.0f}, max {gaps.max()}.")


if __name__ == "__main__":
    main()
