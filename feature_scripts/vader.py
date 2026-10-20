"""VADER lexicon sentiment of each opinion (neg / neu / pos proportions and compound score)."""
import pandas as pd

from utils.data_utils import feature_dir, load_opinions


def output_path(dataset_id, args):
    return feature_dir(args.features_dir / "vader", dataset_id, args.generation_model_id) / "opinion_vader_scores.csv"


def done(dataset_id, args):
    return output_path(dataset_id, args).exists()


def load(args):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def extract(analyzer, dataset_id, args):
    df = load_opinions(dataset_id, args.generation_model_id)
    if args.limit is not None:
        df = df.head(args.limit)
    rows = []
    for text_id, text in zip(df["text_id"], df["text"]):
        scores = analyzer.polarity_scores(text)
        rows.append({"text_id": text_id, **{k: scores[k] for k in ("neg", "neu", "pos", "compound")}})
    path = output_path(dataset_id, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {path}")
