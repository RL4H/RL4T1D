import torch
import torch.nn as nn

from agents.algorithm.ppo import PPO
from agents.models.actor_critic_g2p2c import ActorCritic
from utils.buffers.auxilialry_buffers import AuxiliaryBuffer
from utils.buffers.onpolicy_buffers import RolloutBuffer
from utils.core import f_kl, r_kl

# from utils.logger import LogExperiment


class G2P2C(PPO):
    def __init__(self, args, env_args, logger, load_model, actor_path, critic_path):
        super(G2P2C, self).__init__(args, env_args, logger, load_model, actor_path, critic_path)
        self.device = args.device
        self.completed_interactions = 0

        self.gamma = 1 if args.return_type else args.gamma
        self.distribution = torch.distributions.Normal

        # training params
        self.train_v_iters = args.n_vf_epochs
        self.train_pi_iters = args.n_pi_epochs
        self.batch_size = args.batch_size
        self.pi_lr = args.pi_lr
        self.vf_lr = args.vf_lr

        # load models and setup optimiser.
        # TODO: printing the netork params of PPO instead; FIX.
        self.policy = ActorCritic(self.args, load_model, actor_path, critic_path).to(self.device)
        self.optimizer_Actor = torch.optim.Adam(self.policy.Actor.parameters(), lr=self.pi_lr)
        self.optimizer_Critic = torch.optim.Adam(self.policy.Critic.parameters(), lr=self.vf_lr)
        self.value_criterion = nn.MSELoss()

        # Auxiliary model learning phase
        self.AuxiliaryBuffer = AuxiliaryBuffer(self.args)
        self.aux_mode = args.aux_mode
        self.aux_iterations = args.n_aux_epochs
        self.aux_batch_size = args.aux_batch_size
        self.aux_vf_coef = args.aux_vf_coef
        self.aux_pi_coef = args.aux_pi_coef
        self.aux_lr = args.aux_lr
        self.optimizer_aux_pi = torch.optim.Adam(self.policy.Actor.parameters(), lr=self.aux_lr)
        self.optimizer_aux_vf = torch.optim.Adam(self.policy.Critic.parameters(), lr=self.aux_lr)

        # Planning phase
        self.use_planning = True if args.use_planning == 'yes' else False
        self.n_planning_simulations = args.n_planning_simulations
        self.plan_batch_size = args.plan_batch_size
        self.n_plan_epochs = args.n_plan_epochs

        # ppo params
        self.grad_clip = args.grad_clip
        self.entropy_coef = args.entropy_coef
        self.eps_clip = args.eps_clip
        self.target_kl = args.target_kl

    def train_MCTS_planning(self):
        print('Running Planning Update...')
        planning_loss_log = torch.zeros(1, device=self.device)
        planning_grad, count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        continue_training, buffer_len = True, self.rollout_buffer['len']
        for i in range(self.n_plan_epochs):
            start_idx, n_batch = 0, 0
            while start_idx < buffer_len:
                n_batch += 1
                end_idx = min(start_idx + self.plan_batch_size, buffer_len)
                old_states_batch = self.rollout_buffer['states'][start_idx:end_idx, :, :]
                self.optimizer_Actor.zero_grad()
                rew_norm_var = (self.buffer.reward_normaliser.ret_rms.var).cpu().numpy()
                expert_loss = torch.zeros(1, device=self.device)
                for exp_iter in range(0, old_states_batch.shape[0]):
                    batched_states = old_states_batch[exp_iter].repeat(self.n_planning_simulations, 1, 1)
                    expert_pi, mu, sigma, terminal_s, Gt = self.policy.Actor.expert_search(batched_states, rew_norm_var, mode='batch')
                    V_terminal = self.policy.evaluate_critic(terminal_s, action=None, cgm_pred=False)
                    returns_batch = (Gt + V_terminal.unsqueeze(1) * (self.gamma ** self.args.planning_n_step))
                    _, index = torch.max(returns_batch, 0)
                    index = index[0]
                    dst = self.distribution(mu[index], sigma[index])
                    expert_loss += -dst.log_prob(expert_pi[index].detach())
                expert_loss = expert_loss / (old_states_batch.shape[0])
                expert_loss.backward()
                planning_grad += torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), self.grad_clip)
                self.optimizer_Actor.step()
                count += 1
                start_idx += self.plan_batch_size
                planning_loss_log += expert_loss.detach()
            if not continue_training:
                break
        mean_pi_grad = planning_grad / count if count != 0 else 0
        print('Successful Planning Update')
        return mean_pi_grad, planning_loss_log

    def train_aux(self):
        print('Running aux update...')
        self.AuxiliaryBuffer.update_targets(self.policy)
        aux_val_grad, aux_pi_grad = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        aux_val_loss_log, aux_val_count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        aux_pi_loss_log, aux_pi_count = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        buffer_len = self.AuxiliaryBuffer.old_states.shape[0]
        rand_perm = torch.randperm(buffer_len)
        state = self.AuxiliaryBuffer.old_states[rand_perm, :, :]  # torch.Size([batch, n_steps, features])
        cgm_target = self.AuxiliaryBuffer.cgm_target[rand_perm]
        actions_old = self.AuxiliaryBuffer.actions[rand_perm]
        # new target old_logprob and value are calc based on networks trained after pi and vf
        logprob_old = self.AuxiliaryBuffer.logprob[rand_perm]
        value_target = self.AuxiliaryBuffer.value_target[rand_perm]

        start_idx = 0
        for i in range(self.aux_iterations):
            while start_idx < buffer_len:
                end_idx = min(start_idx + self.aux_batch_size, buffer_len)
                state_batch = state[start_idx:end_idx, :, :]
                cgm_target_batch = cgm_target[start_idx:end_idx]
                value_target_batch = value_target[start_idx:end_idx]
                logprob_old_batch = logprob_old[start_idx:end_idx]
                actions_old_batch = actions_old[start_idx:end_idx]

                if self.aux_mode == 'dual' or self.aux_mode == 'vf_only':
                    self.optimizer_aux_vf.zero_grad()
                    value_predict, cgm_mu, cgm_sigma, _ = self.policy.evaluate_critic(state_batch, actions_old_batch, cgm_pred=True)
                    # Maximum Log Likelihood
                    dst = self.distribution(cgm_mu, cgm_sigma)
                    aux_vf_loss = -dst.log_prob(cgm_target_batch).mean() + self.aux_vf_coef * self.value_criterion(value_predict, value_target_batch)
                    aux_vf_loss.backward()
                    aux_val_grad += torch.nn.utils.clip_grad_norm_(self.policy.Critic.parameters(), self.grad_clip)
                    self.optimizer_aux_vf.step()
                    aux_val_loss_log += aux_vf_loss.detach()
                    aux_val_count += 1

                if self.aux_mode == 'dual' or self.aux_mode == 'pi_only':
                    self.optimizer_aux_pi.zero_grad()
                    logprobs, dist_entropy, cgm_mu, cgm_sigma, _ = self.policy.evaluate_actor(state_batch, actions_old_batch, mode="aux")
                    # debugging
                    if logprobs.shape[0] == 2:
                        print('debugging the error')
                        print(state_batch)
                        print(actions_old_batch)
                        # print(handcraft_feat_batch)
                        print(state_batch.shape)
                        print(actions_old_batch.shape)
                    # experimenting with KL divegrence implementations, kl = 1 is used!
                    if self.args.kl == 0:
                        kl_div = f_kl(logprob_old_batch, logprobs)
                    elif self.args.kl == 1:
                        kl_div = f_kl(logprobs, logprob_old_batch)
                    elif self.args.kl == 2:
                        kl_div = r_kl(logprob_old_batch, logprobs)
                    else:
                        kl_div = r_kl(logprobs, logprob_old_batch)

                    # maximum liklihood est
                    dst = self.distribution(cgm_mu, cgm_sigma)
                    aux_pi_loss = -dst.log_prob(cgm_target_batch).mean() + self.aux_pi_coef * kl_div
                    aux_pi_loss.backward()
                    aux_pi_grad += torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), self.grad_clip)
                    self.optimizer_aux_pi.step()
                    aux_pi_loss_log += aux_pi_loss.detach()
                    aux_pi_count += 1

                start_idx += self.aux_batch_size
        if self.args.verbose:
            print('Successful Auxilliary Update.')
        return aux_val_grad / aux_val_count, aux_val_loss_log, aux_pi_grad / aux_pi_count, aux_pi_loss_log

    def update(self):
        self.rollout_buffer = self.buffer.get(AuxiliaryBuffer=self.AuxiliaryBuffer)
        pi_grad, pi_loss = self.train_pi()
        vf_grad, vf_loss, explained_var, true_var = self.train_vf()

        # Note: aux mode trains the glucose models
        if self.AuxiliaryBuffer.buffer_filled:
            aux_val_grad, aux_val_loss, aux_pi_grad, aux_pi_loss = self.train_aux()  # if (rollout + 1) % self.aux_frequency == 0:

        # TODO: incorrect; have to fix: in the original paper only when RMSE of CGM < 15mg/dL; is when start planning is true
        # This is so that until the glucose models are sufficiently trained we dont run planning.
        # Maybe look at the
        self.start_planning = True

        if self.start_planning:
            plan_pi_grad, plan_loss = self.train_MCTS_planning()

        # TODO: not logging everything; add the G2P2 specific stuff
        data = dict(policy_grad=pi_grad, policy_loss=pi_loss, value_grad=vf_grad, value_loss=vf_loss,
                    explained_var=explained_var, true_var=true_var)
        return {k: v.detach().cpu().flatten().numpy()[0] for k, v in data.items()}


