import torch
from navec import Navec
import numpy as np
from .model import ToxicClassifier


class Toxic:

    eps = .0005

    def __init__(self):
        self.navec = Navec.load("ml/toxificator/navec.tar")
        self.model = ToxicClassifier(self.navec)

        self.model.state_dict(
            torch.load(
                "ml/toxificator/model.pth",
                map_location=torch.device("cpu")
            ),
        )

    def __token2idx(self, text):
        return [self.navec.vocab.get(token, self.navec.vocab.get("<unk>")) for token in text.split()]

    def __preprocess_toxic(self, text):
        return self.__token2idx(text)

    def predict(self, text):
        text = np.array(self.__preprocess_toxic(text))

        self.model.eval()

        with torch.no_grad():
            feature = torch.from_numpy(text).view(1, -1).long()
            output, = self.model(feature)
            # pred = np.round(output.squeeze().item() - self.eps)
            # pred = output.squeeze()

            if abs(output.squeeze().item() - .5) > self.eps:
                pred = torch.round(output.squeeze()).item()
            else:
                pred = 1.0

        return pred
