import pandas as pd
import numpy as np


class Hofstede:
    def __init__(self):
        self.hofstede_indexes = pd.read_csv("ml/hofstedefier/hofstede.csv", index_col=0)
        self.indexes = self.hofstede_indexes.columns[2:]
        self.model = 1

        self.idx = {
            1: "Дистанция власти",
            2: "Индивидуализм",
            3: "Маскулиность",
            4: "Избегание неопределенности",
            5: "Долгосрочная ориентация",
            6: "Допущение",
        }

    def __get_indexes(self, nations):
        countries = self.hofstede_indexes[self.hofstede_indexes["country"].isin(nations)]
        return np.array(countries[self.indexes].to_numpy()).reshape((len(countries), -1))

    @staticmethod
    def __softmax(array):
        return np.exp(array) / np.exp(array).sum()

    def predict(self, x, nations: list[str]):
        nations = self.__get_indexes(nations)

        if len(nations) != 2:
            return {
                "Error": "Select nations"
            }

        a, b = nations
        probs = self.__softmax(abs(a - b)).round(4)
        return {
            self.idx[i]: prob for i, prob in enumerate(probs, 1)
        }


# if __name__ == '__main__':
#     countries = ["сша", "бразилия"]
#     h = Hofstede()
#     print(h.predict(1, countries))
