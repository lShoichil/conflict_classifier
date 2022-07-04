import numpy as np
from transformers import pipeline


class Hofstede:
    candidate_labels = {
        "pdi": "дистанция власти",
        "idv": "индивидуализм",
        "mas": "маскулинность",
        "uai": "избегание неопределенности",
        "lto": "долгосрочная ориентация",
        "ivr": "допущение"
    }

    def __init__(self):
        pass
        self.classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli"
        )

    @staticmethod
    def __softmax(array) -> np.ndarray:
        return np.exp(array) / np.exp(array).sum()

    def predict(self, x, obj: dict = None):
        a, b = obj["values"]
        a, b = np.array(a), np.array(b)

        probs = self.__softmax(abs(a - b)).round(4)
        probs_classes = np.where(probs > .2)[0]

        print(probs_classes)

        if len(probs_classes) > 1:
            candidate_classes = [self.candidate_labels[obj["order"][idx]] for idx in probs_classes]
            out = self.classifier(x, candidate_classes)

            return {
                "index": out["labels"][0],
                "probability": out["scores"][0]
            }
        else:
            return {
                "index": self.candidate_labels[obj["order"][probs_classes[0]]],
                "probability": probs[probs_classes[0]]
            }
