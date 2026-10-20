"""
Download the Habermas Machine and ValuePrism datasets and sample the situations and
opinions used in the paper.

For both datasets: keep the situations with more than 6 opinions, rank them by how evenly
their opinions are spread over the stances (1 = uniform, 0 = all one stance) and keep the
top --max_situations. Within each situation, sample up to 2 opinions per stance:
  - ValuePrism: 6 opinions spread as evenly as possible over Supports / Opposes / Either
    (a stance short of opinions is made up by the others), chosen to be as different from
    each other as possible (greedy farthest-point sampling on token Jaccard distance);
  - Habermas: up to 2 opinions per agree / disagree / neutral group, preferring distinct
    fine-grained agreement labels (e.g. one AGREE and one SOMEWHAT_AGREE).

ValuePrism is a gated dataset on the Hugging Face Hub: accept its terms and log in first.

Output: {output_dir}/habermas_sample.csv, {output_dir}/valueprism_sample.csv

Usage:
    python prepare_data.py
    python prepare_data.py --datasets habermas
"""
import argparse
import logging
import re
import subprocess
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from utils.paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
log = logging.getLogger("prepare_data")

HABERMAS_BASE_URL = "https://storage.googleapis.com/habermas_machine/datasets"
HABERMAS_RATINGS_FILE = "hm_all_position_statement_ratings.parquet"
HABERMAS_CANDIDATES_FILE = "hm_all_candidate_comparisons.parquet"
AGREE_LABELS = {"STRONGLY_AGREE", "AGREE", "SOMEWHAT_AGREE"}
DISAGREE_LABELS = {"STRONGLY_DISAGREE", "DISAGREE", "SOMEWHAT_DISAGREE"}
NEUTRAL_LABELS = {"NEUTRAL"}
HABERMAS_STANCES = ["agree", "disagree", "neutral"]
HABERMAS_MIN_INSTANCES = 6
HABERMAS_OPINIONS_PER_STANCE = 2

VALUEPRISM_URL = "hf://datasets/allenai/ValuePrism/full/full.csv"
VALUEPRISM_STANCES = ["Supports", "Opposes", "Either"]
VALUEPRISM_MIN_INSTANCES = 6
VALUEPRISM_OPINIONS_PER_STANCE = 2

DEFAULT_MAX_SITUATIONS = 500


def compute_balance_scores(df, situation_col, stance_col, stance_values):
    """Per situation: opinion counts per stance, total n and balance_score in [0, 1]."""
    counts = (df.groupby([situation_col, stance_col]).size().unstack(fill_value=0)
              .reindex(columns=stance_values, fill_value=0))
    counts["n"] = counts[stance_values].sum(axis=1)
    target = 1 / len(stance_values)
    proportions = counts[stance_values].div(counts["n"], axis=0)
    counts["balance_score"] = 1 - proportions.sub(target).abs().sum(axis=1) / (2 * (1 - target))
    return counts


def select_top_situations(counts, min_instances, max_situations, dataset):
    valid = counts[counts["n"] > min_instances]
    log.info("[%s] %d / %d situations have more than %d opinions", dataset, len(valid), len(counts), min_instances)
    if len(valid) < max_situations:
        log.warning("[%s] only %d situations left, fewer than %d: keeping all of them", dataset, len(valid),
                    max_situations)
    return valid.nlargest(max_situations, "balance_score").index


def _tokenize(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _jaccard_distance(a, b):
    union = a | b
    return 0.0 if not union else 1 - len(a & b) / len(union)


def pick_diverse_subset(texts, k):
    """Indices of up to k mutually different texts: the most distant pair first, then
    repeatedly the text farthest from the ones already picked."""
    n = len(texts)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return list(range(n))
    if k == 1:
        return [0]
    token_sets = [_tokenize(t) for t in texts]
    dist = [[_jaccard_distance(token_sets[i], token_sets[j]) for j in range(n)] for i in range(n)]
    best_pair, best_dist = (0, 1), -1.0
    for i, j in combinations(range(n), 2):
        if dist[i][j] > best_dist:
            best_dist, best_pair = dist[i][j], (i, j)
    selected = list(best_pair)
    while len(selected) < k:
        remaining = [i for i in range(n) if i not in selected]
        selected.append(max(remaining, key=lambda i: min(dist[i][s] for s in selected)))
    return selected


def balanced_allocation(capacities, target_total):
    """Split target_total over the groups as evenly as their capacities allow."""
    allocation = {g: 0 for g in capacities}
    remaining = min(target_total, sum(capacities.values()))
    active = [g for g in capacities if capacities[g] > 0]
    while remaining > 0 and active:
        share, extra = divmod(remaining, len(active))
        if share == 0:
            for g in sorted(active, key=lambda g: -(capacities[g] - allocation[g]))[:extra]:
                allocation[g] += 1
                remaining -= 1
            break
        for g in active:
            take = min(share, capacities[g] - allocation[g])
            allocation[g] += take
            remaining -= take
        active = [g for g in active if capacities[g] - allocation[g] > 0]
    return allocation


def sample_diverse_balanced(df, stance_col, stance_values, text_col, target_total):
    capacities = {stance: (df[stance_col] == stance).sum() for stance in stance_values}
    allocation = balanced_allocation(capacities, target_total)
    parts = []
    for stance in stance_values:
        if allocation[stance] == 0:
            continue
        group = df[df[stance_col] == stance]
        parts.append(group.iloc[pick_diverse_subset(group[text_col].fillna("").tolist(), allocation[stance])])
    return pd.concat(parts) if parts else df.iloc[0:0]


def sample_diverse_sublabels(group, raw_label_col, k, random_state=0):
    """Up to k rows, with distinct raw labels first."""
    if len(group) <= k:
        return group
    shuffled = group.sample(frac=1, random_state=random_state)
    selected, seen = [], set()
    for idx, row in shuffled.iterrows():
        if len(selected) >= k:
            break
        if row[raw_label_col] not in seen:
            selected.append(idx)
            seen.add(row[raw_label_col])
    if len(selected) < k:
        selected += [idx for idx in shuffled.index if idx not in selected][:k - len(selected)]
    return group.loc[selected]


def sample_up_to_k_per_stance(df, stance_col, stance_values, k, raw_label_col, random_state=0):
    parts = [sample_diverse_sublabels(df[df[stance_col] == stance], raw_label_col, k, random_state)
             for stance in stance_values if (df[stance_col] == stance).any()]
    return pd.concat(parts) if parts else df.iloc[0:0]


def download_habermas(tmpdir):
    for filename in (HABERMAS_RATINGS_FILE, HABERMAS_CANDIDATES_FILE):
        log.info("Downloading %s", filename)
        subprocess.run(["wget", "-q", "-P", str(tmpdir), f"{HABERMAS_BASE_URL}/{filename}"], check=True)
    return pd.read_parquet(tmpdir / HABERMAS_RATINGS_FILE), pd.read_parquet(tmpdir / HABERMAS_CANDIDATES_FILE)


def clean_habermas_ratings(ratings):
    """First-stage agreement ratings of each participant with each question."""
    cols = ["metadata.participant_id", "question.id", "question.text", "question.affirming_statement",
            "question.negating_statement", "ratings.agreement"]
    ratings = ratings[ratings["rating_index"] == 0][cols].copy()
    ratings["ratings.agreement"] = ratings["ratings.agreement"].apply(lambda x: x[0])
    return ratings[ratings["ratings.agreement"] != "[MOCK]"]


def clean_habermas_candidates(candidates):
    """Completed human-written opinions, one per participant and question."""
    cols = ["question.id", "question.text", "question.split", "question.affirming_statement",
            "question.negating_statement", "question.topic", "own_opinion.text", "metadata.participant_id",
            "own_opinion.metadata.provenance", "own_opinion.metadata.status", "ratings.agreement"]
    candidates = candidates[cols]
    candidates = candidates[candidates["own_opinion.metadata.provenance"] != "BOT_CITIZEN"]
    candidates = candidates[candidates["own_opinion.metadata.status"] == "COMPLETED"]
    candidates = candidates.drop_duplicates(subset=["metadata.participant_id", "question.id"])
    return candidates[["metadata.participant_id", "question.id", "question.text", "question.affirming_statement",
                       "question.negating_statement", "own_opinion.text", "ratings.agreement", "question.topic"]]


def agreement_group(label):
    if label in AGREE_LABELS:
        return "agree"
    if label in DISAGREE_LABELS:
        return "disagree"
    if label in NEUTRAL_LABELS:
        return "neutral"
    return None


def prepare_habermas(output_dir, max_situations):
    log.info("Preparing Habermas Machine")
    with TemporaryDirectory() as tmpdir:
        ratings_raw, candidates_raw = download_habermas(Path(tmpdir))
    merged = pd.merge(
        clean_habermas_ratings(ratings_raw), clean_habermas_candidates(candidates_raw),
        on=["metadata.participant_id", "question.id", "question.text", "question.affirming_statement",
            "question.negating_statement"],
        suffixes=("_position", "_candidate"),
    )
    merged["stance_group"] = merged["ratings.agreement_position"].map(agreement_group)
    merged = merged.dropna(subset=["stance_group"])

    counts = compute_balance_scores(merged, "question.id", "stance_group", HABERMAS_STANCES)
    selected = select_top_situations(counts, HABERMAS_MIN_INSTANCES, max_situations, "habermas")
    merged = merged[merged["question.id"].isin(selected)].copy()
    merged = merged.groupby("question.id", group_keys=False).apply(
        lambda g: sample_up_to_k_per_stance(g, "stance_group", HABERMAS_STANCES, HABERMAS_OPINIONS_PER_STANCE,
                                            raw_label_col="ratings.agreement_position"))

    merged["text_id"] = [f"HB_{idx}" for idx in merged.index]
    merged["question.id"] = merged["question.id"].str.replace("S", "HB_", regex=False)
    merged = merged.rename(columns={"question.id": "situation_id", "metadata.participant_id": "hb_participant_id",
                                    "question.affirming_statement": "situation",
                                    "ratings.agreement_position": "stance", "own_opinion.text": "text"})
    merged["dataset"] = "habermas"
    merged = merged[["situation_id", "text_id", "situation", "stance", "stance_group", "hb_participant_id", "text",
                     "dataset"]]
    out_path = output_dir / "habermas_sample.csv"
    merged.to_csv(out_path, index=False)
    log.info("Wrote %d opinions on %d situations to %s", len(merged), merged["situation_id"].nunique(), out_path)


def prepare_valueprism(output_dir, max_situations):
    log.info("Preparing ValuePrism")
    df = pd.read_csv(VALUEPRISM_URL, dtype={"vrd": str, "text": str})
    counts = compute_balance_scores(df, "situation", "valence", VALUEPRISM_STANCES)
    selected = select_top_situations(counts, VALUEPRISM_MIN_INSTANCES, max_situations, "valueprism")
    df = df[df["situation"].isin(selected)].copy()
    target_total = len(VALUEPRISM_STANCES) * VALUEPRISM_OPINIONS_PER_STANCE
    df = df.groupby("situation", group_keys=False).apply(
        lambda g: sample_diverse_balanced(g, "valence", VALUEPRISM_STANCES, "explanation", target_total))

    df["text_id"] = [f"VP_{i}" for i in df.index]
    situation_ids = {s: f"VPS_{i}" for i, s in enumerate(df["situation"].unique())}
    df["situation_id"] = df["situation"].map(situation_ids)
    df = df.rename(columns={"valence": "stance", "vrd": "vp_vrd", "text": "vp_vrd_value",
                            "explanation": "vp_explanation"})
    df["dataset"] = "valueprism"
    df = df[["situation_id", "text_id", "situation", "stance", "vp_vrd", "vp_vrd_value", "vp_explanation", "dataset"]]
    out_path = output_dir / "valueprism_sample.csv"
    df.to_csv(out_path, index=False)
    log.info("Wrote %d rows on %d situations to %s", len(df), df["situation_id"].nunique(), out_path)


PREPARERS = {"habermas": prepare_habermas, "valueprism": prepare_valueprism}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--datasets", nargs="+", choices=list(PREPARERS), default=list(PREPARERS))
    parser.add_argument("--max_situations", type=int, default=DEFAULT_MAX_SITUATIONS,
                        help="Situations kept per dataset, by stance balance.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        PREPARERS[name](args.output_dir, args.max_situations)


if __name__ == "__main__":
    main()
