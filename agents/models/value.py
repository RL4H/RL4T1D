import torch.nn as nn
import torch.nn.functional as F

from agents.models.normed_linear import NormedLinear
from agents.models.feature_extracter import LSTMFeatureExtractor


class ValueModule(nn.Module):
    def __init__(self, args):
        super(ValueModule, self).__init__()

        self.feature_extractor = args.n_rnn_hidden * args.n_rnn_layers * args.rnn_directions

        self.last_hidden = self.feature_extractor * 2

        self.fc_layer1 = nn.Linear(self.feature_extractor, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)
        self.value = NormedLinear(self.last_hidden, 1, scale=0.1)

    def forward(self, extract_states):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        value = (self.value(fc_output))
        return value
