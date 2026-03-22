import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ADIClassifier:
    def __init__(self):
        self.model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # IMPORTANT: read labels from the model itself
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )

        outputs = self.model(**tokens)
        probs = torch.softmax(outputs.logits, dim=1)[0]

        scores = {
            self.id2label[i].upper(): float(probs[i])
            for i in range(len(probs))
        }

        top = max(scores, key=scores.get)

        return {
            "scores": scores,
            "top": top,
            "top_score": scores[top]
        }
