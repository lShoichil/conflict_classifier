import re
import torch
from navec import Navec
import numpy as np
from .model import ToxicClassifier


class Toxic:
    def __init__(self):
        self.navec = Navec.load("toxificator/navec.tar")
        self.model = ToxicClassifier(self.navec)

        self.model.state_dict(
            torch.load(
                "toxificator/model.pth",
                map_location=torch.device("cpu")
            ),
        )

    @staticmethod
    def __clean_toxic(text):
        text = text.lower()
        text = re.sub(r"[^а-яА-Я]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def __token2idx(self, text):
        return [self.navec.vocab.get(token, self.navec.vocab.get("<unk>")) for token in text.split()]

    def __preprocess_toxic(self, text):
        return self.__token2idx(self.__clean_toxic(text))

    def predict(self, text):
        text = np.array(self.__preprocess_toxic(text))

        self.model.eval()

        with torch.no_grad():
            feature = torch.from_numpy(text).view(1, -1).long()
            feature = feature
            batch_size = feature.size(0)
            h = self.model.init_hidden(batch_size)

            output, h = self.model(feature, h)
            pred = torch.round(output.squeeze())

        return pred.item()
