import torch
import torch.nn as nn
import torch.nn.functional as F


from agents.models.normed_linear import NormedLinear


class PolicyModule(nn.Module):
    def __init__(self, args):
        super(PolicyModule, self).__init__()
        self.device = args.device

        self.output = args.n_action
        self.feature_extractor = (args.n_rnn_hidden * args.n_rnn_layers * args.rnn_directions)

        self.last_hidden = self.feature_extractor * 2

        self.fc_layer1 = nn.Linear(self.feature_extractor, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)

        self.mu = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.sigma = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.normalDistribution = torch.distributions.Normal

    def forward(self, extract_states):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        # removed normalization of expected vaalue. Although the output is sending through the normalization, see inside normalization block. it is not done.
        # print("fc_output\n")
        # print(fc_output)
        mu = F.tanh(self.mu(fc_output))
        sigma = F.sigmoid(self.sigma(fc_output) + 1e-5)
        z = self.normalDistribution(0, 1).sample()
        action = mu + sigma * z
        action = torch.clamp(action, -1, 1)
        try:
            dst = self.normalDistribution(mu, sigma)
            log_prob = dst.log_prob(action[0])
        except ValueError:
            print('\nCurrent mu: {}, sigma: {}'.format(mu, sigma))
            print('shape: {}. {}'.format(mu.shape, sigma.shape))
            print(extract_states.shape)
            print(extract_states)
            log_prob = torch.ones(2, 1, device=self.device, dtype=torch.float32) * self.glucose_target
        return mu, sigma, action, log_prob
    
