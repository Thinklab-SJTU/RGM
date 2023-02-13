import random

import numpy as np
import gym
import copy
import datetime
import torch
from utils.gm_solver import get_norm_affinity, get_aff_score_norm

maxRound = 300


# am: affinity metric
# ag: affinity graph
class Environment:
    def __init__(self, am, beta=1):
        self.N = int(np.sqrt(len(am)))
        action_dims = self.N * self.N
        ob_dims = self.N * self.N
        self.beta = beta
        self.action_dims = action_dims
        self.action_space = gym.spaces.MultiDiscrete(action_dims)
        self.observation_space = gym.spaces.Box(0, 10, (1, ob_dims // self.N, ob_dims // self.N))
        self.current_sol = np.zeros((self.N, self.N), dtype=np.float32)
        self.am = am
        self.old_sol = copy.deepcopy(self.current_sol)
        self.best_sol = copy.deepcopy(self.current_sol)
        self.best_sol_position = 0
        self.best_ans = self.calc_score(self.current_sol) - 100
        self.ag = am
        self.ag_weight = np.array(self.ag, dtype=np.float32)
        self.ag_adjacent = np.zeros((self.N * self.N, self.N * self.N), dtype=np.float32)
        self.max_rounds = 30
        self.avail_actions = np.ones((self.N * self.N,)) * 1
        for i in range(self.N * self.N):
            for j in range(self.N * self.N):
                if self.ag_weight[i][j] > 0 or i == j:
                    self.ag_adjacent[i][j] = 1
        self.rounds = 0

    def calc_score(self, sol, normalize=False):
        am = torch.FloatTensor(self.am).cuda()
        if normalize:
            sol = torch.FloatTensor(sol.T).cuda()
            return self.beta * get_aff_score_norm(am, sol, self.N).cpu().detach().numpy()[0][0]
        else:
            sol = torch.FloatTensor(sol).cuda()
        return self.beta * torch.matmul(torch.matmul(torch.reshape(sol, (1, -1)), am),
                                        torch.reshape(sol, (-1, 1))).cpu().detach().numpy()[0][0]

    def image_state(self):
        avail_actions = np.ones((self.N * self.N,)) * 0
        for i in range(self.N * self.N):
            if self.check(i):
                avail_actions[i] = 1
        weights = np.reshape(self.ag_weight, (self.N * self.N, self.N * self.N))
        state = np.concatenate(([self.current_sol.flatten()], [avail_actions]))
        state = np.concatenate((state, weights))
        state = np.array(state, dtype=np.float32)
        return state

    def reset(self):
        self.current_sol = np.zeros((self.N, self.N))
        self.old_sol = copy.deepcopy(self.current_sol)
        self.best_sol = copy.deepcopy(self.current_sol)
        self.best_sol_position = 0
        self.best_ans = self.calc_score(self.current_sol) - 100
        self.ag = self.am
        self.rounds = 0
        self.avail_actions = np.ones((self.N * self.N,)) * 1
        for i in range(self.N * self.N):
            if self.check(i):
                self.avail_actions[i] = 1
        # return [np.concatenate((self.get_state(), avail_actions, np.reshape(self.ag_adjacent, self.N * self.N * self.N * self.N),
        #                         np.reshape(self.ag_weight, self.N * self.N * self.N * self.N)))]
        return self.image_state()

    def get_state(self):
        return np.reshape(self.current_sol, (-1,))

    def get_best_ans(self):
        return self.best_sol_position, self.best_sol, self.best_ans

    def check(self, action):
        return self.avail_actions[action]

    def check_sol(self, action):
        x = action // self.N
        y = action % self.N
        if np.sum(self.current_sol[:, y]) >= 1:
            return False
        if np.sum(self.current_sol[x, :]) >= 1:
            return False
        # for i in range(self.N):
        #     if self.current_sol[i][y] == 1:
        #         return False
        # for j in range(self.N):
        #     if self.current_sol[x][j] == 1:
        #         return False
        return True

    def step(self, action, normalize=False):
        # action = action[0]
        # action = np.argmax(action)
        # self.current_sol[action // self.N][action % self.N] = 1
        if self.check(action):
            self.current_sol[action // self.N][action % self.N] = 1
        else:
            self.current_sol[action // self.N, :] = 0
            self.current_sol[:, action % self.N] = 0
            self.current_sol[action // self.N][action % self.N] = 1

        old_ans = self.calc_score(self.old_sol, normalize)
        current_ans = self.calc_score(self.current_sol, normalize)
        self.rounds += 1
        if current_ans > self.best_ans:
            self.best_ans = current_ans
            self.best_sol = copy.deepcopy(self.current_sol)
            self.best_sol_position = self.rounds
        self.old_sol = copy.deepcopy(self.current_sol)
        self.avail_actions = np.ones((self.N * self.N,)) * 0
        for i in range(self.N * self.N):
            if self.check_sol(i):
                self.avail_actions[i] = 1
        # return [np.concatenate((self.get_state(), avail_actions, np.reshape(self.ag_adjacent, self.N * self.N * self.N * self.N),
        #                         np.reshape(self.ag_weight, self.N * self.N * self.N * self.N)))], (current_ans - old_ans), np.sum(
        #     self.get_state()) == self.N, [self.rounds]
        return self.image_state(), (current_ans - old_ans), self.rounds == self.max_rounds, [self.best_sol_position]


if __name__ == '__main__':
    am = np.load('data/qap_ngm_train.npy')
    starttime = datetime.datetime.now()
    env = Environment(am[0])
    random_pos = 0
    random_sol = 0
    random_ans = 0
    random_reward_list = []
    for i in range(10000):
        env.reset()
        done = False
        r_list = []
        while not done:
            n = np.random.randint(0, env.N * env.N)
            if env.check(n):
                s, r, done, rounds = env.step([n])
                r_list.append(r[0])
        pos, sol, ans = env.get_best_ans()
        if ans > random_ans:
            random_pos = pos
            random_sol = sol
            random_ans = ans
            random_reward_list = r_list
    print(random_ans, random_pos)
    print(random_sol)
    print(random_reward_list)
    print('Hello world!')
    endtime = datetime.datetime.now()
    print('Running time: %s Seconds' % (endtime - starttime))
