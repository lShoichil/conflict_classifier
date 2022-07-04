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
        if obj is not None:
            a, b = obj["values"]
            a, b = np.array(a), np.array(b)

            probs = self.__softmax(abs(a - b)).round(4)

            out = self.classifier(x, list(self.candidate_labels.values()))

            out_obj = {
                key: value for key, value in zip(out["labels"], out["scores"])
            }

            probs_obj = {
                self.candidate_labels[key]: value for key, value in zip(obj["order"], probs)
            }

            arr = [out_obj[key] + probs_obj[key] for key in self.candidate_labels.values()]
            arr = np.array(arr)

            arr = self.__softmax(arr)
            idx = arr.argmax(axis=0)

            return {
                "index": list(probs_obj.keys())[idx],
                "probability": arr[idx]
            }

        out = self.classifier(x, list(self.candidate_labels.values()))
        return {
            "index": out["labels"][0],
            "probability": out["scores"][0]
        }
