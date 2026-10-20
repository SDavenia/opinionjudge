"""
Split the sampled opinions of the quality check between the annotators, --n_per_annotator
each, balanced over the target stance, each annotator in their own random order.
--n_overlap items are rated by every annotator, for inter-annotator agreement (0: disjoint).

Outputs: assignments/for_annotators/{annotator}.csv (sent to the annotators) and
assignments/key.csv (kept).

Usage:
    python -m human_annotation.quality.assign_annotators --n_annotators 3
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STANCES = ["Supports", "Opposes", "Either"]
ANNOTATOR_COLUMNS = ["annotator_id", "position", "item_id", "situation", "target_stance", "value_type", "value",
                     "explanation", "opinion"]
KEY_COLUMNS = ["annotator_id", "position", "item_id", "sample_id", "situation_id", "text_id", "stance", "vp_vrd",
               "shared"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items_csv", type=Path, default=HERE / "samples" / "items.csv")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n_annotators", type=int, help="Annotators named ANN1..ANNn.")
    group.add_argument("--annotators", nargs="+", help="Annotator ids.")
    parser.add_argument("--n_per_annotator", type=int, default=50)
    parser.add_argument("--n_overlap", type=int, default=0, help="Items rated by every annotator.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", type=Path, default=HERE / "assignments")
    return parser.parse_args()


def stance_balanced_order(items, rng):
    """sample_ids alternating over the stances, shuffled within each stance."""
    by_stance = [list(rng.permutation(items.loc[items["stance"] == s, "sample_id"])) for s in STANCES]
    order = []
    while any(by_stance):
        for queue in by_stance:
            if queue:
                order.append(queue.pop())
    return order


def main():
    args = parse_args()
    items = pd.read_csv(args.items_csv).set_index("sample_id", drop=False)
    annotators = args.annotators or [f"ANN{k + 1}" for k in range(args.n_annotators)]
    n, k_shared, n_ann = args.n_per_annotator, args.n_overlap, len(annotators)
    if not 0 <= k_shared <= n:
        raise ValueError("--n_overlap must be between 0 and --n_per_annotator.")
    needed = k_shared + n_ann * (n - k_shared)
    if len(items) < needed:
        raise ValueError(f"{len(items)} items, {needed} needed: re-run sample_items with --n_items {needed}.")

    order = stance_balanced_order(items, np.random.default_rng([args.seed, 999]))
    shared, rest = order[:k_shared], order[k_shared:needed]
    # Grouped by stance and dealt round-robin: per stance, the annotators differ by at most one item.
    rest = sorted(rest, key=lambda i: STANCES.index(items.loc[i, "stance"]))
    own = {a: rest[k::n_ann] for k, a in enumerate(annotators)}

    (args.output_dir / "for_annotators").mkdir(parents=True, exist_ok=True)
    keys = []
    for k, annotator_id in enumerate(annotators):
        rng = np.random.default_rng([args.seed, k])
        ids = shared + own[annotator_id]
        ids = [ids[i] for i in rng.permutation(len(ids))]
        rows = []
        for position, sample_id in enumerate(ids, start=1):
            it = items.loc[sample_id]
            item_id = f"{rng.integers(16 ** 8):08x}"
            rows.append({"annotator_id": annotator_id, "position": position, "item_id": item_id,
                         "situation": it["situation"], "target_stance": it["stance"], "value_type": it["vp_vrd"],
                         "value": it["vp_vrd_value"], "explanation": it["vp_explanation"], "opinion": it["text"]})
            keys.append({"annotator_id": annotator_id, "position": position, "item_id": item_id,
                         "sample_id": sample_id, "situation_id": it["situation_id"], "text_id": it["text_id"],
                         "stance": it["stance"], "vp_vrd": it["vp_vrd"], "shared": sample_id in shared})
        path = args.output_dir / "for_annotators" / f"{annotator_id}.csv"
        pd.DataFrame(rows, columns=ANNOTATOR_COLUMNS).to_csv(path, index=False)
        print(f"{annotator_id}: {len(rows)} items ({k_shared} shared) -> {path}")
    pd.DataFrame(keys, columns=KEY_COLUMNS).to_csv(args.output_dir / "key.csv", index=False)
    pd.Series({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}).to_json(
        args.output_dir / "assignment_config.json", indent=2)


if __name__ == "__main__":
    main()
