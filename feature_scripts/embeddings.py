"""Sentence embedding of each opinion, compared by cosine similarity in the analyses."""
import pandas as pd
import torch

from utils.data_utils import feature_dir, load_opinions

# Task prompt of models that need one (nomic-embed-text-v2-moe has no default and embeds
# poorly without it); "STS" matches comparing two texts by cosine similarity.
DEFAULT_PROMPT_NAME = {"nomic-ai/nomic-embed-text-v2-moe": "STS"}


def output_path(dataset_id, args):
    return (feature_dir(args.features_dir / "embeddings", dataset_id, args.generation_model_id)
            / args.embedding_model_id.replace("/", "__") / "opinion_embeddings.parquet")


def done(dataset_id, args):
    return output_path(dataset_id, args).exists()


def _patch_extended_attention_mask(model):
    """nomic-embed-text-v2-moe's remote code calls get_extended_attention_mask, which
    transformers 5 removed from PreTrainedModel: add back the encoder version."""
    model_cls = type(model[0].auto_model)
    if hasattr(model_cls, "get_extended_attention_mask"):
        return

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        dtype = dtype or self.dtype
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
        extended = extended.to(dtype=dtype)
        return (1.0 - extended) * torch.finfo(dtype).min

    model_cls.get_extended_attention_mask = get_extended_attention_mask


def load(args):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.embedding_model_id, trust_remote_code=True)
    _patch_extended_attention_mask(model)
    return model


def extract(model, dataset_id, args):
    df = load_opinions(dataset_id, args.generation_model_id)
    if args.limit is not None:
        df = df.head(args.limit)
    prompt_name = DEFAULT_PROMPT_NAME.get(args.embedding_model_id)
    embeddings = model.encode(df["text"].tolist(), batch_size=args.batch_size, show_progress_bar=True,
                              convert_to_numpy=True, prompt_name=prompt_name)
    path = output_path(dataset_id, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text_id": df["text_id"], "embedding": [e.tolist() for e in embeddings]}).to_parquet(path, index=False)
    print(f"Saved {path}")
