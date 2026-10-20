"""Perplexity of each opinion under a judge, conditioned on a stance-free prompt about the
situation (teacher forcing: mean negative log-likelihood of the opinion tokens only)."""
import torch
from tqdm import tqdm

from utils.data_utils import generator_tag, load_opinions
from utils.generation_utils import batches, chat_text
from utils.models_utils import tokenizer_of
from utils.prompts_utils import DEFAULT_PERPLEXITY_PROMPT, PERPLEXITY_PROMPTS

DATASETS = ["valueprism", "habermas", "habermas_hybrid"]


def output_path(dataset_id, judge, args):
    path = args.features_dir / "perplexity" / dataset_id / judge
    if dataset_id != "habermas":
        path = path / generator_tag(dataset_id, args.generation_model_id)
    return path / f"{DEFAULT_PERPLEXITY_PROMPT[dataset_id]}.csv"


def done(dataset_id, judge, args):
    return output_path(dataset_id, judge, args).exists()


@torch.no_grad()
def compute_perplexities(model, tok, spec, situations, opinions, prompt_key, max_length):
    prefixes = [chat_text(tok, spec, PERPLEXITY_PROMPTS[prompt_key].format(situation=s)) for s in situations]
    # The space keeps the first opinion token from merging with the end of the prefix.
    full_texts = [prefix + " " + opinion for prefix, opinion in zip(prefixes, opinions)]
    prefix_lens = [len(tok(prefix, add_special_tokens=False)["input_ids"]) for prefix in prefixes]

    device = next(model.parameters()).device
    encodings = tok(full_texts, add_special_tokens=False, padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt").to(device)
    input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
    batch_size, seq_len = input_ids.shape

    # Only the (non-padding) opinion tokens count.
    label_mask = attention_mask.clone().bool()
    if tok.padding_side == "left":
        pad_amounts = seq_len - attention_mask.sum(dim=1)
        for i in range(batch_size):
            label_mask[i, :pad_amounts[i].item() + prefix_lens[i]] = False
    else:
        for i in range(batch_size):
            label_mask[i, :prefix_lens[i]] = False

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_label_mask = label_mask[:, 1:].contiguous()

    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1) * shift_label_mask
    token_counts = shift_label_mask.sum(dim=1).clamp(min=1)
    perplexities = torch.exp(-(token_log_probs.sum(dim=1) / token_counts))
    # An opinion truncated away entirely gets NaN rather than exp(0) = 1.
    perplexities[shift_label_mask.sum(dim=1) == 0] = float("nan")
    return perplexities.cpu().tolist()


def extract(model, proc, spec, judge, dataset_id, args):
    df = load_opinions(dataset_id, args.generation_model_id)
    if args.limit is not None:
        df = df.head(args.limit)
    if df["text"].nunique() != len(df):
        raise ValueError(f"Duplicated opinions in the {dataset_id} data.")
    tok = tokenizer_of(proc, spec)
    prompt_key = DEFAULT_PERPLEXITY_PROMPT[dataset_id]
    rows = list(zip(df["situation"], df["text"]))
    perplexities = []
    for batch in tqdm(list(batches(rows, args.perplexity_batch_size)), desc=f"Perplexity ({judge}, {dataset_id})"):
        situations, opinions = zip(*batch)
        perplexities += compute_perplexities(model, tok, spec, list(situations), list(opinions), prompt_key,
                                             args.perplexity_max_length)
    df["perplexity"] = perplexities
    path = output_path(dataset_id, judge, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["situation_id", "situation", "text_id", "text", "perplexity"]].to_csv(path, index=False)
    print(f"Saved {path}")

