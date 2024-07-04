import torch.nn as nn


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(LSTMFeatureExtractor, self).__init__()
        self.n_features = args.n_features
        self.LSTM = nn.LSTM(input_size=self.n_features, hidden_size=args.n_rnn_hidden, num_layers=args.n_rnn_layers,
                            batch_first=True, bidirectional=args.bidirectional)  # (seq_len, batch, input_size)

    def forward(self, s):
        output, (hid, cell) = self.LSTM(s)
        lstm_output = hid.view(hid.size(1), -1)  # => batch , layers * hid
        return lstm_output
