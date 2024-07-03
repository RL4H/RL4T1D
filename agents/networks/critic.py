import torch.nn as nn

from agents.networks.feature_extracter import LSTMFeatureExtractor
from agents.networks.value import ValueModule


class CriticNetwork(nn.Module):
    def __init__(self, args):
        super(CriticNetwork, self).__init__()
        self.FeatureExtractor = LSTMFeatureExtractor(args)
        self.ValueModule = ValueModule(args)

    def forward(self, s):
        lstmOut = self.FeatureExtractor.forward(s)
        value = self.ValueModule.forward(lstmOut)
        return value
