import torch
import torch.nn as nn
from navec import Navec
from slovnet.model.emb import NavecEmbedding


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.hidden_size = 256
        self.num_layers = 1

        # TODO: enter path to navec
        self.embedding = NavecEmbedding(
            Navec.load(...)
        )

        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                in_features=self.hidden_size*2,
                out_features=64
            ),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(
                in_features=64,
                out_features=32
            ),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=32,
                out_features=1
            ),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size))
        c0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size))

        return h0, c0

    def forward(self, x, hidden):
        batch_size = x.size()

        embedded_x = self.embedding(x)

        output, hidden = self.lstm(embedded_x, hidden)

        output = self.linear(output[:, -1, :])
        output = self.fc(output)

        return output, hidden


# def clean(text):
#     text = text.lower()
#     text = re.sub(r"[^а-яА-Я]+", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()
#
#
# def token2idx(text):
#     return [navec.vocab.get(token, navec.vocab.get("<unk>")) for token in text.split()]
#
#
# def predict(text):
#     text = np.array(token2idx(text))
#
#     model.eval()
#     with torch.no_grad():
#         feature = torch.from_numpy(text).view(1, -1).long()
#         feature = feature
#         batch_size = feature.size(0)
#         h = model.init_hidden(batch_size)
#
#         output, h = model(feature, h)
#         pred = torch.round(output.squeeze())
#
#     return "Фу, токсик" if pred.item() else "Ты ж мой хороший"
