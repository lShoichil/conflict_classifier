from transformers import pipeline


class Hofstede:
    candidate_labels = [
        "дистанция власти",
        "индивидуализм",
        "маскулинность",
        "избегание неопределенности",
        "долгосрочная ориентация",
        "допущение"
    ]

    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli"
        )

    def predict(self, x):
        out = self.classifier(x, self.candidate_labels)

        return {
            "index": out["labels"][0],
            "probability": out["scores"][0]
        }
