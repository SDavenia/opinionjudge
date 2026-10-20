import re
from pathlib import Path

import pandas as pd

from utils.data_utils import DEFAULT_GENERATION_PROMPT, generator_tag, opinions_path
from utils.prompts_utils import GENERATION_PROMPTS, GENERATION_PROMPTS_BY_DATASET, SCORING_PROMPTS
from utils.paths import SCORING_DIR


def add_scoring_args(parser):
    parser.add_argument("--dataset_id", required=True, choices=["habermas", "valueprism", "habermas_hybrid"])
    parser.add_argument("--generation_model_id", default=None,
                        help="Generator of the synthetic opinions (valueprism, habermas_hybrid).")
    parser.add_argument("--generation_prompt_version", default=None, choices=list(GENERATION_PROMPTS),
                        help="Default: reflective_person (valueprism), hb_reflective_person (habermas_hybrid).")
    parser.add_argument("--scoring_prompt_version", default="0_1", choices=list(SCORING_PROMPTS))
    parser.add_argument("--option_setting", default="0_1", choices=["four", "0_1"],
                        help="Score scale the judge answers on.")
    parser.add_argument("--generation_csv_path", type=Path, default=None,
                        help="Opinions to score, in place of the default file of the dataset.")
    parser.add_argument("--extract_ids_from", type=Path, default=None,
                        help="CSV with id_1/id_2 columns: score only these pairs.")
    parser.add_argument("--num_examples", type=int, default=None, help="Score only the first N pairs.")
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=SCORING_DIR)


def check_scoring_args(parser, args):
    if args.dataset_id == "habermas":
        return
    if args.generation_prompt_version is None:
        args.generation_prompt_version = DEFAULT_GENERATION_PROMPT[args.dataset_id]
    elif args.generation_prompt_version not in GENERATION_PROMPTS_BY_DATASET[args.dataset_id]:
        parser.error(f"--generation_prompt_version {args.generation_prompt_version} is not valid for "
                     f"{args.dataset_id}; choose from {GENERATION_PROMPTS_BY_DATASET[args.dataset_id]}.")
    if args.generation_model_id is None and (args.dataset_id == "habermas_hybrid" or args.generation_csv_path is None):
        parser.error(f"--generation_model_id is required for {args.dataset_id}.")


def resolve_output_path(args, judge_kind):
    """judge_kind is 'open' (score.py) or 'closed' (score_openrouter.py)."""
    tag = generator_tag(args.dataset_id, args.generation_model_id, args.generation_prompt_version)
    prefix = "" if tag is None else f"{tag}_"
    path = Path(args.output_dir) / args.dataset_id / judge_kind / f"{prefix}{args.judge_model_id}_{args.scoring_prompt_version}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_prompt_variations(prompt_version):
    entry = SCORING_PROMPTS[prompt_version]
    return entry if isinstance(entry, dict) else {prompt_version: entry}


def load_generation_df(args):
    path = args.generation_csv_path or opinions_path(args.dataset_id, args.generation_model_id,
                                                     args.generation_prompt_version)
    df = pd.read_csv(path)

    if args.dataset_id == "habermas":
        print(f"Loaded {path}: {len(df)} opinions.")
        return df.reset_index(drop=True)

    if args.dataset_id == "habermas_hybrid":
        is_human = df["origin"] == "human"
        is_synthetic = ((df["origin"] == "synthetic") & (df["generation_model"] == args.generation_model_id)
                        & (df["generation_prompt_version"] == args.generation_prompt_version))
        print(f"Loaded {path}: {is_human.sum()} human + {is_synthetic.sum()} synthetic opinions.")
        if not is_human.any() or not is_synthetic.any():
            raise ValueError(f"{path} needs both human and synthetic opinions to build hybrid pairs.")
        return df[is_human | is_synthetic].reset_index(drop=True)

    before = len(df)
    if "generation_prompt_version" in df.columns:
        df = df[df["generation_prompt_version"] == args.generation_prompt_version]
    if "generation_model" in df.columns:
        df = df[df["generation_model"] == args.generation_model_id]
    df = df.reset_index(drop=True)
    print(f"Loaded {path}: {len(df)} of {before} opinions from {args.generation_model_id} / "
          f"{args.generation_prompt_version}.")
    if df.empty:
        raise ValueError(f"No opinions of {args.generation_model_id} / {args.generation_prompt_version} in {path}.")
    return df


def build_pairs(gen_df, cross_origin_only=False):
    """Same-situation opinion pairs with id_1 < id_2 (ids are row numbers). With
    cross_origin_only (habermas_hybrid), only human x synthetic pairs, human first."""
    df = gen_df.copy().reset_index().rename(columns={"index": "id"})
    if cross_origin_only:
        human = df[df["origin"] == "human"]
        synthetic = df[df["origin"] == "synthetic"]
        return (human.merge(synthetic, on="situation", suffixes=("_1", "_2"))
                .sort_values(["id_1", "id_2"]).reset_index(drop=True))
    return df.merge(df, on="situation", suffixes=("_1", "_2")).query("id_1 < id_2").reset_index(drop=True)


def select_pairs(args, gen_df):
    pairs_df = build_pairs(gen_df, cross_origin_only=args.dataset_id == "habermas_hybrid")
    if args.num_examples is not None:
        pairs_df = pairs_df.head(args.num_examples)
    if args.extract_ids_from is not None:
        keep = pd.read_csv(args.extract_ids_from)[["id_1", "id_2"]]
        pairs_df = pairs_df.merge(keep, on=["id_1", "id_2"], how="inner")
    return pairs_df


def build_direction_prompts(pairs_df, prompt_version):
    """For each pair and prompt variation, the 1to2 prompt followed by the 2to1 one."""
    variations = get_prompt_variations(prompt_version)
    prompts = []
    for _, row in pairs_df.iterrows():
        for template in variations.values():
            prompts.append(template.format(situation=row["situation"], first_opinion=row["text_1"],
                                           second_opinion=row["text_2"]))
            prompts.append(template.format(situation=row["situation"], first_opinion=row["text_2"],
                                           second_opinion=row["text_1"]))
    return prompts


def expand_pairs_for_variations(pairs_df, prompt_version):
    """One row per (pair, prompt variation), aligned with build_direction_prompts."""
    variations = list(get_prompt_variations(prompt_version))
    expanded = pairs_df.loc[pairs_df.index.repeat(len(variations))].reset_index(drop=True)
    expanded["prompt_variation"] = variations * len(pairs_df)
    return expanded


def fill_scores(pairs_df, generations, args, judge_name, decimals):
    pairs_df["generated_score_1to2"] = pd.Series(generations[0::2])
    pairs_df["generated_score_2to1"] = pd.Series(generations[1::2])
    pairs_df["judge_model"] = judge_name
    pairs_df["generation_model"] = "human" if args.dataset_id == "habermas" else args.generation_model_id
    pairs_df["generation_prompt_version"] = None if args.dataset_id == "habermas" else args.generation_prompt_version
    for direction in ("1to2", "2to1"):
        pairs_df[f"parsed_score_{direction}"] = pairs_df[f"generated_score_{direction}"].apply(
            parse_generation_scoring, option_setting=args.option_setting, decimals=decimals)
    return pairs_df


def parse_generation_scoring(generation, option_setting="four", decimals=1):
    """The score in a judge's answer, or None if there is none or more than one."""
    if generation is None or generation == "":
        return None

    if option_setting == "four":
        found = [option for option in ["1", "2", "3", "4"] if option in generation]
        return found[0] if len(found) == 1 else None

    if option_setting == "0_1":
        # "0.75", ".75", "1.0", "0", "1", but not the "1" of "10"
        matches = re.findall(r"(?<!\d)(?:0?\.\d+|1\.0+|0|1)(?!\d)", generation)
        candidates = [round(float(m), decimals) for m in matches if 0.0 <= float(m) <= 1.0]
        if not candidates or len(set(candidates)) > 1:
            return None
        return candidates[0]

    raise ValueError(f"Unknown option_setting: {option_setting}")
