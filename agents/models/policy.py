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
    
    def get_fim(self, x):
        mu, sigma, _, _ = self.forward(x)

        cov_inv = sigma.pow(-2).squeeze(0).repeat(x.size(0))

        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "sigma.weight":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1

        return cov_inv.detach(), mu, {'std_id': std_id, 'std_index': std_index}
    
    def get_kl(self, x):
        mu1, sigma1, _, _ = self.forward(x)

        mu0 = mu1.detach()
        sigma0 = sigma1.detach()

        kl = sigma1.log() - sigma0.log() + (sigma0.pow(2) + (mu0 - mu1).pow(2)) / (2.0 * sigma1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    
