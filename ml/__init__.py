from ml.surname import Nation
from ml.toxificator import Toxic
import spacy
import re


class Predictor:
    def __init__(self):
        self.toxic = Toxic()
        self.nation = Nation()
        self.nlp = spacy.load("ru_core_news_lg")

    @staticmethod
    def __clean(text):
        text = text.lower()
        text = re.sub(r"[^а-яА-Я]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def __get_named(self, text):
        return [ent.text for ent in self.nlp(text).ents if ent.label_ == "PER"]

    def predict(self, text):
        text = self.__clean(text)

        answer = {
            "toxic": self.toxic.predict(text)
        }

        entities = self.__get_named(text)

        if entities:
            answer["nationality"] = [self.nation.predict(surname) for surname in entities]

        return answer
