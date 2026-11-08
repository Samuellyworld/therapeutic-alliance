"""Local transformer inference with complete, token-weighted long-note coverage."""

from importlib.metadata import version

SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOTIONS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")


class TransformerClassifier:
    def __init__(self, model_id, revision="main", batch_size=8, local_files_only=False):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install model support: python -m pip install -e '.[models]'") from error
        self.torch = torch
        self.batch_size = batch_size
        options = {"revision": revision, "local_files_only": local_files_only, "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, use_safetensors=True, **options
        ).to("cpu").eval()
        limit = min(self.tokenizer.model_max_length, self.model.config.max_position_embeddings, 512)
        self.chunk_size = limit - self.tokenizer.num_special_tokens_to_add(pair=False)
        self.metadata = {
            "model": model_id,
            "revision": self.model.config._commit_hash,
            "transformers": version("transformers"),
            "torch": version("torch"),
            "device": "cpu",
            "chunk_tokens": self.chunk_size,
            "aggregation": "non-overlapping chunks; token-count-weighted mean probabilities",
        }

    def predict(self, texts):
        results = []
        for text in texts:
            ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            if not ids:
                raise ValueError("Cannot classify empty text.")
            chunks = [ids[i:i + self.chunk_size] for i in range(0, len(ids), self.chunk_size)]
            weighted = self.torch.zeros(self.model.config.num_labels)
            for offset in range(0, len(chunks), self.batch_size):
                batch = chunks[offset:offset + self.batch_size]
                inputs = [self.tokenizer.prepare_for_model(chunk, return_attention_mask=True)
                          for chunk in batch]
                padded = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
                with self.torch.inference_mode():
                    probabilities = self.model(**padded).logits.softmax(dim=-1)
                weights = self.torch.tensor([len(chunk) for chunk in batch]).unsqueeze(1)
                weighted += (probabilities * weights).sum(dim=0)
            scores = (weighted / len(ids)).tolist()
            results.append({
                "scores": {self.model.config.id2label[i].lower(): score
                           for i, score in enumerate(scores)},
                "chunks": len(chunks),
            })
        return results
