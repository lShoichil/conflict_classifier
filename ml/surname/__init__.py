import json
import torch
from .model import SurnameClassifier, SurnameVectorizer


class Nation:
    def __init__(self):
        with open("surname/vectorizer.json") as file:
            self.vectorizer = SurnameVectorizer.from_serializable(json.load(file))

        self.model = SurnameClassifier()
        self.model.state_dict(
            torch.load(
                "surname/model.pth",
                map_location=torch.device("cpu")
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
