"""Sentiment of each opinion: probability of POSITIVE vs NEGATIVE under
siebert/sentiment-roberta-large-english."""
import pandas as pd
import torch
from tqdm import tqdm

from utils.data_utils import feature_dir, load_opinions
from utils.generation_utils import batches

MODEL_ID = "siebert/sentiment-roberta-large-english"


def output_path(dataset_id, args):
    return feature_dir(args.features_dir / "sentiment", dataset_id, args.generation_model_id) / "opinion_sentiment_scores.csv"


def done(dataset_id, args):
    return output_path(dataset_id, args).exists()


class SentimentScorer:
    def __init__(self, model_id=MODEL_ID):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()
        id2label = {k: v.upper() for k, v in self.model.config.id2label.items()}
        self.positive_id = next(i for i, label in id2label.items() if label == "POSITIVE")
        self.negative_id = next(i for i, label in id2label.items() if label == "NEGATIVE")

    @torch.no_grad()
    def score_batch(self, texts, max_length=512):
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                             return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits[:, [self.negative_id, self.positive_id]]
        positive = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
        return [{"sentiment_score": p, "label": "POSITIVE" if p >= 0.5 else "NEGATIVE"} for p in positive]


def load(args):
    return SentimentScorer()


def extract(scorer, dataset_id, args):
    df = load_opinions(dataset_id, args.generation_model_id)
    if args.limit is not None:
        df = df.head(args.limit)
    rows = []
    items = list(zip(df["text_id"], df["text"]))
    for batch in tqdm(list(batches(items, args.batch_size)), desc=f"Sentiment ({dataset_id})"):
        text_ids, texts = zip(*batch)
        rows += [{"text_id": t, **r} for t, r in zip(text_ids, scorer.score_batch(list(texts)))]
    path = output_path(dataset_id, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {path}")
