import torch
import torch.nn as nn
import torch.nn.functional as F


from agents.models.feature_extracter import LSTMFeatureExtractor



class QvalModel(nn.Module):
    def __init__(self, args):
        super(QvalModel, self).__init__()

        self.output = args.n_action

        self.feature_extractor = args.n_rnn_hidden * args.n_rnn_layers * args.rnn_directions
        self.last_hidden = self.feature_extractor * 2

        self.fc_layer1 = nn.Linear(self.feature_extractor + self.output, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)

        self.q = nn.Linear(self.last_hidden, self.output)

    def forward(self, extract_state, action, mode):
        concat_dim = 1 if (mode == 'batch') else 0
        concat_state_action = torch.cat((extract_state, action), dim=concat_dim)
        fc_output1 = F.relu(self.fc_layer1(concat_state_action))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        qval = self.q(fc_output)
        return qval
