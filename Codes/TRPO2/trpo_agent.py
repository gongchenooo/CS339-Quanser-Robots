import torch
import numpy as np
import os
from models import network
from rl_utils.running_filter.running_filter import ZFilter
from utils import select_actions, eval_actions, conjugated_gradient, line_search, set_flat_params_to
from datetime import datetime
import csv
import quanser_robots
class trpo_agent:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.net = network(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.old_net = network(self.env.observation_space.shape[0], self.env.action_space.shape[0])

        self.old_net.load_state_dict(self.net.state_dict()) # old net means the old policy

        self.optimizer = torch.optim.Adam(self.net.Value_net.parameters(), lr=self.args.lr) #it is used to optimize value net
        # define the running mean filter
        self.running_state = ZFilter((self.env.observation_space.shape[0],), clip=5)

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = self.args.save_dir + self.args.env_name + '/'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    def learn(self):
        file = open('logs/%s/lr=%.5f_batchsize=%d_gamma=%.3f_tau=%.3f.csv' % (self.args.env_name, self.args.lr, self.args.batch_size, self.args.gamma, self.args.tau), 'w', encoding='utf-8', newline='')
        csv_writer = csv.writer(file)
        csv_writer.writerow(['episode', 'rewards', 'final reward', 'VL', 'PL*1e7'])
        print('name of environment: ', self.args.env_name)
        print('gamma=%.3f\tlr=%.5f\tbatchsize=%d\ttau=%.4f' % (self.args.gamma, self.args.lr, self.args.batch_size, self.args.tau))

        num_updates = self.args.total_timesteps // self.args.nsteps # how many samples we need to collect in a step
        obs = self.running_state(self.env.reset())
        final_reward = 0
        episode_reward = 0
        self.dones = False

        for update in range(num_updates):
            obs = self.running_state(self.env.reset())
            mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    obs_tensor = self._get_tensors(obs)
                    value, pi = self.net(obs_tensor) # state_value, (action_mean, action_std)
                # choose action according to sampling the normalization (action_mean, action_std)
                actions = select_actions(pi)
                mb_obs.append(np.copy(obs))
                mb_actions.append(actions)
                mb_dones.append(self.dones)
                mb_values.append(value.detach().numpy().squeeze())

                # execute action and get new obs, reward, done
                obs_, reward, done, _ = self.env.step(actions)
                self.dones = done
                mb_rewards.append(reward)
                if done: # after executing the actions the state is done
                    obs_ = self.env.reset()
                obs = self.running_state(obs_)
                episode_reward += reward #
                mask = 0.0 if done else 1.0
                final_reward *= mask # if done then final_reward = episode_reward else final_rewards unchanged
                final_reward += (1 - mask) * episode_reward
                episode_reward *= mask # if done then episode_reward=0 else episode_reward unchanged

            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            mb_values = np.asarray(mb_values, dtype=np.float32)

            with torch.no_grad():
                obs_tensor = self._get_tensors(obs)
                last_value, _ = self.net(obs_tensor)
                last_value = last_value.detach().numpy().squeeze()

            mb_returns = np.zeros_like(mb_rewards) # [0,0,...,0]
            mb_advs = np.zeros_like(mb_rewards) # [0,0,...,0]
            lastgaelam = 0
            for t in reversed(range(self.args.nsteps)):
                if t == self.args.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - mb_dones[t + 1]
                    nextvalues = mb_values[t + 1]
                delta = mb_rewards[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values[t] # A(s)=[R(s->s') + r*V(s') - V(s)]
                mb_advs[t] = lastgaelam = delta + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam # advs[t] = A(t) + r*tau*advs[t+1]
            mb_returns = mb_advs + mb_values # A(t)+V(t) = new_V(t)
            mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-5) # normalize advantages

            self.old_net.load_state_dict(self.net.state_dict()) # store the old policy
            # start to update the network
            policy_loss, value_loss = self._update_network(mb_obs, mb_actions, mb_returns, mb_advs)
            torch.save([self.net.state_dict(), self.running_state], self.model_path + 'lr=%.5f_batchsize=%d.pt' %
                       (self.args.lr, self.args.batch_size))
            csv_writer.writerow([update, sum(mb_rewards), final_reward,value_loss, policy_loss*10000000])
            print('[{}] Update: {} / {}, Frames: {}, Reward sum: {:.8f}, Final reward: {:.8f}, VL: {:.8f}, PL: {:.3f}*1e7'.format(datetime.now(), update, \
                    num_updates, (update + 1)*self.args.nsteps, sum(mb_rewards), final_reward, value_loss, policy_loss*10000000))
        file.close()

    # start to update network
    def _update_network(self, mb_obs, mb_actions, mb_returns, mb_advs):
        mb_obs_tensor = torch.tensor(mb_obs, dtype=torch.float32)
        mb_actions_tensor = torch.tensor(mb_actions, dtype=torch.float32)
        mb_returns_tensor = torch.tensor(mb_returns, dtype=torch.float32).unsqueeze(1)
        mb_advs_tensor = torch.tensor(mb_advs, dtype=torch.float32).unsqueeze(1)

        values, _ = self.net(mb_obs_tensor) # values calculated by the old policy
        with torch.no_grad():
            _, pi_old = self.old_net(mb_obs_tensor) # actions(mean, std) calculated by the old policy

        # define surr_loss
        surr_loss = self._get_surrogate_loss(mb_obs_tensor, mb_advs_tensor, mb_actions_tensor, pi_old)
        # calculate the gradients of surr_loss to Policy_net.parameters
        surr_grad = torch.autograd.grad(surr_loss, self.net.Policy_net.parameters())
        # surr_grad -> flat surr_grad
        flat_surr_grad = torch.cat([grad.view(-1) for grad in surr_grad]).data

        # use the conjugated gradient to calculate the scaled direction vector (natural gradient)
        nature_grad = conjugated_gradient(self._fisher_vector_product, -flat_surr_grad, 10, mb_obs_tensor, pi_old)
        # calculate the scaleing ratio
        non_scale_kl = 0.5 * (nature_grad * self._fisher_vector_product(nature_grad, mb_obs_tensor, pi_old)).sum(0, keepdim=True)
        scale_ratio = torch.sqrt(non_scale_kl / self.args.max_kl)
        final_nature_grad = nature_grad / scale_ratio[0]
        # calculate the expected improvement rate...
        expected_improve = (-flat_surr_grad * nature_grad).sum(0, keepdim=True) / scale_ratio[0]
        # get the flat param ...
        prev_params = torch.cat([param.data.view(-1) for param in self.net.Policy_net.parameters()])
        # start to do the line search
        success, new_params = line_search(self.net.Policy_net, self._get_surrogate_loss, prev_params, final_nature_grad, \
                                expected_improve, mb_obs_tensor, mb_advs_tensor, mb_actions_tensor, pi_old)
        set_flat_params_to(self.net.Policy_net, new_params)
        # then trying to update the Value_net network
        inds = np.arange(mb_obs.shape[0])

        for _ in range(self.args.vf_itrs):
            value_loss = []
            np.random.shuffle(inds)
            for start in range(0, mb_obs.shape[0], self.args.batch_size):
                end = start + self.args.batch_size
                mbinds = inds[start:end]
                mini_obs = mb_obs[mbinds]
                mini_returns = mb_returns[mbinds]
                # put things in the tensor
                mini_obs = torch.tensor(mini_obs, dtype=torch.float32)
                mini_returns = torch.tensor(mini_returns, dtype=torch.float32).unsqueeze(1)
                values, _ = self.net(mini_obs)
                v_loss = (mini_returns - values).pow(2).mean()
                value_loss.append(v_loss.item())
                self.optimizer.zero_grad()
                v_loss.backward()
                self.optimizer.step()
        value_loss = np.asarray(value_loss)
        return surr_loss.item(), value_loss.mean()

    # get the surrogate loss
    def _get_surrogate_loss(self, obs, adv, actions, pi_old):
        _, pi = self.net(obs)
        log_prob = eval_actions(pi, actions) # the log probability of actions sampling in pi
        old_log_prob = eval_actions(pi_old, actions).detach() # the log probability of actions sampling in pi_old
        surr_loss = -torch.exp(log_prob - old_log_prob) * adv
        return surr_loss.mean()

    # the product of the fisher informaiton matrix and the nature gradient -> Ax
    def _fisher_vector_product(self, v, obs, pi_old): # (natural grad, obs, pi_old)
        kl = self._get_kl(obs, pi_old)
        kl = kl.mean()
        # start to calculate the second order gradient of the KL
        kl_grads = torch.autograd.grad(kl, self.net.Policy_net.parameters(), create_graph=True)
        flat_kl_grads = torch.cat([grad.view(-1) for grad in kl_grads])
        kl_v = (flat_kl_grads * torch.autograd.Variable(v)).sum()
        kl_second_grads = torch.autograd.grad(kl_v, self.net.Policy_net.parameters())
        flat_kl_second_grads = torch.cat([grad.contiguous().view(-1) for grad in kl_second_grads]).data
        flat_kl_second_grads = flat_kl_second_grads + self.args.damping * v
        return flat_kl_second_grads

    # get the kl divergence between two distributions
    def _get_kl(self, obs, pi_old):
        mean_old, std_old = pi_old
        _, pi = self.net(obs)
        mean, std = pi
        kl = -torch.log(std / std_old) + (std.pow(2) + (mean - mean_old).pow(2)) / (2 * std_old.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
 
    # get the tensors
    def _get_tensors(self, obs):
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
