from typing import Tuple, Dict, Any

from nltk.stem.snowball import RussianStemmer
from difflib import get_close_matches
import pandas as pd
import spacy
import re


class Preprocessor:
    def __init__(self):
        self.stemmer = RussianStemmer()
        self.df = pd.read_csv("preprocessor/counties.csv")
        self.stemmed_countries = self.df["stemming"].tolist()
        self.nlp = spacy.load("ru_core_news_md")

    @staticmethod
    def __clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^а-яА-Я]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def __get_entities(self, text: str, ent_type: str = "PER") -> set:
        return {ent.text if ent_type == "PER" else ent.lemma_ for ent in self.nlp(text).ents if ent.label_ == ent_type}

    def __stemming(self, tokens: list[str]) -> list[str]:
        return [self.stemmer.stem(token) for token in tokens]

    def __replace_stop(self, text):
        return " ".join([token.text for token in self.nlp(text) if not token.is_stop])

    def __tokenize(self, text: str) -> list[str]:
        return [token.text for token in self.nlp(text) if token.pos_ == "NOUN"]

    def __nation_matches(self, stemmed_tokens: list[str]) -> set[str]:
        matched_nations = set()

        for token in stemmed_tokens:
            if len(token) > 3:
                data: list[str] = get_close_matches(token, self.stemmed_countries, n=1, cutoff=.75)

                if data:
                    matched_nations.add(self.df[self.df["stemming"] == data[0]]["country"].item())

        return matched_nations

    def __get_nations(self, text: str) -> dict:
        spacy_nation = self.__get_entities(text, "LOC")

        nltk_nations = self.__nation_matches(
            self.__stemming(
                self.__tokenize(
                    self.__replace_stop(text)
                )
            )
        )

        spacy_persons = self.__get_entities(text)

        return {
            "nations": spacy_nation.union(nltk_nations),
            "persons": spacy_persons
        }

    def preprocess(self, text: str) -> tuple[dict[Any, Any], str]:
        return {
            # "raw": text,
            **self.__get_nations(self.__clean(text))
        }, self.__clean(text),


# if __name__ == '__main__':
#     text = \
#         """
#         Мартина из Бразилии сидела со своими друзьями в столовой, когда мимо прошел американский студент.
#         Их взгляды случайно встретились, он сказал: «Как дела» и прошел дальше.
#         Она чувствовала себя смущенной и растерянной.
#         """
#     preprocessor = Preprocessor()
#
#     print(preprocessor.preprocess(text))
