import json
import torch
from navec import Navec
from models import Toxic, Nation
import spacy
import re
import numpy as np


class NationPredictor:
    def __init__(self):
        with open("models/vectorizer_for_Nation.json") as file:
            content = json.load(file)
            self.vectorizer = Nation.SurnameVectorizer.from_serializable(content)

        self.model = Nation.SurnameClassifier()

        self.model.state_dict(
            torch.load(
                "to_upload/Nation_model.pth", map_location=torch.device("cpu")
            )
        )

    def predict(self, surname):
        vectorized_surname, vec_length = self.vectorizer.vectorize(surname)
        vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(dim=0)
        vec_length = torch.tensor([vec_length], dtype=torch.int64)

        h = self.model.init_hidden(1)

        result, h = self.model(vectorized_surname, h, vec_length, apply_softmax=True)
        probability_values, indices = result.max(dim=1)

        index = indices.item()
        prob_value = probability_values.item()

        predicted_nationality = self.vectorizer.nationality_vocab.lookup_index(index)

        return {'nationality': predicted_nationality, 'probability': prob_value, 'surname': surname}


class ToxicPredictor:
    def __init__(self):
        self.navec = Navec.load("models/navec_for_Toxic.tar")
        self.model = Toxic.Model(self.navec)

        self.model.state_dict(
            torch.load(
                "to_upload/Toxificator-512-Nation_model.pth",
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

        return "Фу, токсик" if pred.item() else "Ты ж мой хороший"


class Predictor:
    def __init__(self):
        self.toxic = ToxicPredictor()
        self.nation = NationPredictor()
        self.nlp = spacy.load("ru_core_news_md")

    def __get_named(self, text):
        return [ent.text for ent in self.nlp(text).ents if ent.label_ == "PER"]

    def predict(self, text):
        answer = {
            "toxic": self.toxic.predict(text)
        }

        entities = self.__get_named(text)

        if entities:
            answer["nationality"] = [self.nation.predict(surname) for surname in entities]

        return answer


if __name__ == '__main__':
    test = "Безос самый богатый"
    pred = Predictor()

    print(pred.predict(test))
