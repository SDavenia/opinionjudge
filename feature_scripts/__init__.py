from feature_scripts import alignment, embeddings, perplexity, refusal, safety, sentiment, vad, vader

# Features computed under each judge model (open models only) and features of the opinions alone.
JUDGE_FEATURES = {"perplexity": perplexity, "alignment": alignment, "refusal": refusal}
TEXT_FEATURES = {"safety": safety, "sentiment": sentiment, "vad": vad, "vader": vader, "embeddings": embeddings}
