"""Valence, arousal and dominance of each opinion (RobroKools/vad-bert regression head, on the
EmoBank 1-5 scale)."""
import pandas as pd
import torch
from tqdm import tqdm

from utils.data_utils import feature_dir, load_opinions
from utils.generation_utils import batches

MODEL_ID = "RobroKools/vad-bert"


def output_path(dataset_id, args):
    return feature_dir(args.features_dir / "vad", dataset_id, args.generation_model_id) / "opinion_vad_scores.csv"


def done(dataset_id, args):
    return output_path(dataset_id, args).exists()


class VADScorer:
    def __init__(self, model_id=MODEL_ID):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()
        if self.model.config.num_labels != 3:
            raise ValueError(f"Expected a 3-dimensional VAD head, got {self.model.config.num_labels} labels.")

    @torch.no_grad()
    def score_batch(self, texts, max_length=512):
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                             return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits.float().cpu().tolist()
        return [{"valence": v, "arousal": a, "dominance": d} for v, a, d in logits]


def load(args):
    return VADScorer()


def extract(scorer, dataset_id, args):
    df = load_opinions(dataset_id, args.generation_model_id)
    if args.limit is not None:
        df = df.head(args.limit)
    rows = []
    items = list(zip(df["text_id"], df["text"]))
    for batch in tqdm(list(batches(items, args.batch_size)), desc=f"VAD ({dataset_id})"):
        text_ids, texts = zip(*batch)
        rows += [{"text_id": t, **r} for t, r in zip(text_ids, scorer.score_batch(list(texts)))]
    path = output_path(dataset_id, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {path}")
