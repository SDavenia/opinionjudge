import math
from itertools import combinations

import pandas as pd


def get_den_normalization(n: int, range_size: float):
    return math.floor(pow(n, 2) / 4 * range_size)


def calculate_rs(scores: list[list[float]], range: tuple[float, float] = (0.0, 1.0)) -> float:
    """Repetition stability. scores: one list of repeated judgments per query; queries
    with fewer than two judgments are skipped."""
    tot_sum = 0.0
    n = 0
    size_range = range[1] - range[0]
    for scores_query in scores:
        if len(scores_query) < 2:
            continue
        sum_abs_diff = sum(abs(s - s2) for s, s2 in combinations(scores_query, 2))
        tot_sum += sum_abs_diff / get_den_normalization(len(scores_query), size_range)
        n += 1
    if n == 0:
        return float("nan")
    return 1 - (1 / n) * tot_sum


def calculate_pc(scores: list[list[tuple[float, float]]], range: tuple[float, float] = (0.0, 1.0)) -> float:
    """Positional consistency. scores: per query, a list of (score_1to2, score_2to1)."""
    n = len(scores)
    tot_sum = 0.0
    size_range = range[1] - range[0]
    for scores_couple in scores:
        sum_diff_couples = sum(abs(s - s2) for s, s2 in scores_couple) / size_range
        tot_sum += sum_diff_couples / len(scores_couple)
    return 1 - (1 / n) * tot_sum


def _grouped_scores_rs(df_scoring: pd.DataFrame, score_column: str) -> list:
    return df_scoring.groupby(["id_1", "id_2"]).agg(lambda x: x.tolist())[score_column].tolist()


def calculate_rs_from_df(df_scoring: pd.DataFrame, range: tuple[float, float] = (0.0, 1.0)) -> float:
    all_scores = _grouped_scores_rs(df_scoring, "parsed_score_1to2") + _grouped_scores_rs(df_scoring, "parsed_score_2to1")
    return calculate_rs(all_scores, range=range)


def _grouped_scores_pc(df_scoring: pd.DataFrame) -> list:
    return (df_scoring.groupby(["id_1", "id_2"])
            .agg(lambda x: x.tolist())
            .apply(lambda row: list(zip(row["parsed_score_1to2"], row["parsed_score_2to1"])), axis=1)
            .tolist())


def calculate_pc_from_df(df_scoring: pd.DataFrame, range: tuple[float, float] = (0.0, 1.0)) -> float:
    return calculate_pc(_grouped_scores_pc(df_scoring), range=range)
