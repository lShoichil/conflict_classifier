from .surname import Nation
from .toxificator import Toxic
from .hofstedefier import Hofstede
import spacy
import re


class Predictor:
    def __init__(self):
        self.toxic = Toxic()
        self.nation = Nation()
        self.hofstede = Hofstede()
        self.nlp = spacy.load("ru_core_news_md")

    @staticmethod
    def __clean(text):
        text = text.lower()
        text = re.sub(r"[^а-яА-Я]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def __get_named(self, text):
        return [ent.text for ent in self.nlp(text).ents if ent.label_ == "PER"]

    def predict(self, text, nationalities: list[str] = None):
        text = self.__clean(text)

        answer = {
            "toxic": self.toxic.predict(text)
        }

        entities = self.__get_named(text)

        if entities:
            answer["nationality"] = [self.nation.predict(surname) for surname in entities]

        if not nationalities:
            nationalities = [x["nationality"] for x in answer["nationality"]]

        answer["hofstedefier"] = self.hofstede.predict(1, list(map(lambda x: x.lower(), nationalities)))

        return answer


if __name__ == '__main__':
    p = Predictor()
    print(p.predict("Привет, Андрей", ["Бразилия", "США"]))
