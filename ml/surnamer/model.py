import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Vocabulary(object):
    def __init__(self, token_to_idx=None):

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

    def to_serializable(self):
        return {'token_to_idx': self._token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        return self._token_to_idx[token]

    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         'begin_seq_token': self._begin_seq_token,
                         'end_seq_token': self._end_seq_token})
        return contents

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]


class SurnameVectorizer(object):
    def __init__(self, char_vocab, nationality_vocab):
        self.char_vocab = char_vocab
        self.nationality_vocab = nationality_vocab

    def vectorize(self, surname, vector_length=-1):
        indices = [self.char_vocab.begin_seq_index]
        indices.extend(self.char_vocab.lookup_token(token)
                       for token in surname)
        indices.append(self.char_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.char_vocab.mask_index

        return out_vector, len(indices)

    @classmethod
    def from_dataframe(cls, surname_df):
        char_vocab = SequenceVocabulary()
        nationality_vocab = Vocabulary()

        for index, row in surname_df.iterrows():
            for char in row.surname:
                char_vocab.add_token(char)
            nationality_vocab.add_token(row.nationality)

        return cls(char_vocab, nationality_vocab)

    @classmethod
    def from_serializable(cls, contents):
        char_vocab = SequenceVocabulary.from_serializable(contents['char_vocab'])
        nat_vocab = Vocabulary.from_serializable(contents['nationality_vocab'])

        return cls(char_vocab=char_vocab, nationality_vocab=nat_vocab)

    def to_serializable(self):
        return {'char_vocab': self.char_vocab.to_serializable(),
                'nationality_vocab': self.nationality_vocab.to_serializable()}


class SurnameClassifier(nn.Module):
    def __init__(
            self,
            embedding_size=300,
            num_embeddings=100,
            rnn_hidden_size=1024,
            num_layers=2,
            num_classes=18,
            padding_idx=0
    ):
        super(SurnameClassifier, self).__init__()
        self.hidden_size = rnn_hidden_size
        self.layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_size,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            embedding_size,
            rnn_hidden_size,
            bidirectional=True,
            num_layers=num_layers
        )

        self.out = nn.Linear(self.hidden_size * 2 * num_layers, num_classes)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_size * 2, self.layers)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)

        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).squeeze(2)
        return context, soft_attn_weights.cpu().data.numpy()

    def forward(self, X):
        input = self.embedding(X)
        input = input.permute(1, 0, 2)

        hidden_state = Variable(torch.zeros(self.layers * 2, len(X), self.hidden_size))
        cell_state = Variable(torch.zeros(self.layers * 2, len(X), self.hidden_size))

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)

        attn_output, attention = self.attention_net(output, final_hidden_state)
        output = attn_output.view(-1, self.hidden_size * 2 * self.layers)
        output = self.out(output)
        return output, attention
