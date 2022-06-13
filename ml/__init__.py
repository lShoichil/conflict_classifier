from surname import Nation
from toxificator import Toxic
import spacy


class Predictor:
    def __init__(self):
        self.toxic = Toxic()
        self.nation = Nation()
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
