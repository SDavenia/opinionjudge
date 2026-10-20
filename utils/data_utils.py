from pathlib import Path

import pandas as pd

from utils.paths import DATA_DIR, GENERATIONS_DIR

DATASETS = ["valueprism", "habermas", "habermas_hybrid"]
# Datasets with their own situations; habermas_hybrid shares the habermas ones.
SITUATION_DATASETS = ["valueprism", "habermas"]

DEFAULT_GENERATOR = "llama-3.3-70b"
DEFAULT_GENERATION_PROMPT = {"valueprism": "reflective_person", "habermas_hybrid": "hb_reflective_person"}


def generator_tag(dataset_id, generation_model_id=DEFAULT_GENERATOR, generation_prompt_version=None):
    """'{generator}_{prompt}' naming the synthetic opinions of a dataset, None for habermas."""
    if dataset_id == "habermas":
        return None
    return f"{generation_model_id}_{generation_prompt_version or DEFAULT_GENERATION_PROMPT[dataset_id]}"


def situations_path(dataset_id, data_dir=DATA_DIR):
    return Path(data_dir) / f"{dataset_id}_sample.csv"


def opinions_path(dataset_id, generation_model_id=DEFAULT_GENERATOR, generation_prompt_version=None,
                  generations_dir=GENERATIONS_DIR, data_dir=DATA_DIR):
    if dataset_id == "habermas":
        return situations_path("habermas", data_dir)
    tag = generator_tag(dataset_id, generation_model_id, generation_prompt_version)
    return Path(generations_dir) / dataset_id / f"{tag}.csv"


def load_situations(dataset_id, data_dir=DATA_DIR, limit=None):
    """(situation_id, situation) pairs, one per distinct situation (first id kept)."""
    df = pd.read_csv(situations_path(dataset_id, data_dir))
    situations = df[["situation_id", "situation"]].drop_duplicates(subset="situation")
    if limit is not None:
        situations = situations.head(limit)
    return list(situations.itertuples(index=False, name=None))


def load_opinions(dataset_id, generation_model_id=DEFAULT_GENERATOR, generation_prompt_version=None,
                  generations_dir=GENERATIONS_DIR, data_dir=DATA_DIR):
    """Opinions to compute per-opinion features on. For habermas_hybrid only the synthetic
    ones: the human ones are the habermas opinions, whose features are computed there."""
    df = pd.read_csv(opinions_path(dataset_id, generation_model_id, generation_prompt_version, generations_dir,
                                   data_dir))
    if dataset_id == "habermas_hybrid":
        df = df[df["origin"] == "synthetic"].reset_index(drop=True)
    return df


def feature_dir(root, dataset_id, generation_model_id=DEFAULT_GENERATOR, generation_prompt_version=None):
    """Output folder of a per-opinion feature: {root}/{dataset}, plus a generator level for
    habermas_hybrid."""
    path = Path(root) / dataset_id
    if dataset_id == "habermas_hybrid":
        path = path / generator_tag(dataset_id, generation_model_id, generation_prompt_version)
    return path
