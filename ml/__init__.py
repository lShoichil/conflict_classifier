from preprocessor import Preprocessor
from surnamer import Nation
from toxificator import Toxic
from hofstedefier import Hofstede
import spacy


class Predictor:
    def __init__(self):
        self.toxic = Toxic()
        self.nation = Nation()
        self.hofstede = Hofstede()
        self.preprocessor = Preprocessor()
        self.nlp = spacy.load("ru_core_news_md")

    def predict(self, raw, obj: dict = None):
        answer = self.preprocessor.preprocess(raw)

        answer["toxic"] = self.toxic.predict(answer["cleaned"])

        if answer["persons"]:
            answer["nationality"] = [self.nation.predict(surname) for surname in answer["persons"]]

        if answer["toxic"]:
            answer["hofstedefier"] = self.hofstede.predict(raw, obj)

        return answer


if __name__ == '__main__':
    p = Predictor()
    print(p.predict(
        "Я встретил француженку у себя в Америке, и когда она подошла ко мне, чтобы отдать поприветсвовать, она поцеловала меня в щеку, я быстро отступил, чтобы нормально поговорить, но она двинулась вперед, приближаясь к моей другой щеке, которую она тоже поцеловала. В тот момент я немного не понимал, что происходит, и еще немного отступил, чтобы попытаться нормально говорить, но она продолжала двигаться вперед, чтобы снова поцеловать меня в другую щеку ... Я был немного смущен, но улыбался, а затем, когда я увидел, как она улыбается и смотрит прямо на меня, я вроде бы хотел, чтобы она хотела целоваться прямо здесь, так что на этот раз это был я, который пошел вперед и поцеловал ее языком на глазах у всех, включая ее бойфренда. Остальное вы можете себе представить.",
        {
            "order": ["pdi", "idv", "mas", "uai", "lto", "ivr"],
            "values": [[68, 71, 43, 86, 63, 48], [40, 91, 62, 46, 26, 68]]
        }
        )
    )
