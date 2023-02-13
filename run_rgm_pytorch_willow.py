#!/usr/bin/env python3
from datetime import datetime
import random
import copy
import logging

from prioritized_memory import Memory
from multiprocessing import Pool, Queue, Process
import dqn_model_r
from environment_qap_ag import Environment
import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings("ignore")
import os
from utils.rrwm import RRWM
from utils.rrwm_mask import MaskRRWM
from utils.hungarian import hungarian

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DYNAMIC = 0

GAMMA = 0.9
BATCH_SIZE = 20
REPLAY_SIZE = 50000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 20
REPLAY_START_SIZE = 1000
MULTI_STEP = 1

EPSILON_DECAY_LAST_FRAME = (10 ** 4) * 2.0
EPSILON_BONUS = 1.1
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

output_unary = []

ADDITION_RATE_DECAY = 0.1
ADDITION_RATE_FINAL = 0.01
ADDITION_RATE_BOOST = 5

UNITS = 64
HIDDEN_SIZE = 128
T = 3

MEAN_REWARD_BOUND = 1.05

rrwm_solver = RRWM()
rrwm_mask = MaskRRWM()


# Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones,
                                                                                                  dtype=np.uint8), np.array(
            next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.mistake = 1
        self.env = env
        self.exp_buffer = exp_buffer
        self.length = 0
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.state_list = [self.state]
        self.action_list = []
        self.reward_list = []
        self.random_list = []
        self.acc_list = []
        self.length = 0
        self.total_reward = 0.0

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    def set_env(self, env, reset=True):
        self.env = env
        if reset:
            self.state = env.reset()
        else:
            self.state = env.image_state()
        self.state_list = [self.state]

    def append_sample(self, net, tgt_net, state, action, reward, next_state, done):
        target = net(torch.autograd.Variable(torch.FloatTensor([state])).to(device), hard_mask=hard_mask,
                     normalize=normalize).data
        old_val = target[0][action].cpu().detach().numpy()
        target_val = tgt_net(torch.autograd.Variable(torch.FloatTensor([next_state])).to(device), hard_mask=hard_mask,
                             normalize=normalize).data
        if done:
            target[0][action] = torch.FloatTensor([reward])[0].cuda()
        else:
            target[0][action] = torch.FloatTensor([reward])[0].cuda() + GAMMA * torch.max(target_val)
        error = abs(old_val - target[0][action].cpu().detach().numpy())
        self.exp_buffer.add(error, (state, action, reward, next_state, done))

    def play_step(self, net, tgt_net, addition, order, epsilon=0.0, use_addition=False, device="cpu", k=-1):
        self.length += 1
        done_reward = None
        rand = np.random.random()
        if DYNAMIC:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v, use_dynamic_embedding=True, hard_mask=hard_mask, normalize=normalize)
            # q_vals_v = net(state_v, use_dynamic_embedding=True, hard_mask=(np.sum(self.env.best_sol) < self.env.N))
        if rand < epsilon:
            t = np.random.randint(0, self.env.action_dims)
            action = t
            self.random_list.append('!')
        else:
            if not DYNAMIC:
                state_a = np.array([self.state], copy=False)
                state_v = torch.tensor(state_a).to(device)
                q_vals_v = net(state_v, use_dynamic_embedding=False, hard_mask=hard_mask, normalize=normalize)
                # q_vals_v = net(state_v, use_dynamic_embedding=True, hard_mask=(np.sum(self.env.best_sol) < self.env.N), normalize=True)
            self.random_list.append('*')
            q_vals_v = q_vals_v.cpu().detach().numpy()
            q_vals_v = q_vals_v[0]
            # action = np.argmax(q_vals_v)
            action_ = np.argmax(q_vals_v)
            value = np.reshape(q_vals_v, (self.env.N, self.env.N))
            value_n = (value - np.min(value)) / (np.max(value) - np.min(value))
            if np.sum(self.env.current_sol) == 0:
                output_unary.append(value_n)
            while k > 0:
                q_vals_v[np.argmax(q_vals_v)] = np.min(q_vals_v) - 1
                k -= 1
            if not self.env.check(np.argmax(q_vals_v)):
                action = action_
            else:
                action = np.argmax(q_vals_v)
        new_state, reward, is_done, pos = self.env.step(action, normalize=normalize)
        self.action_list.append(action)

        _, sol, _ = self.env.get_best_ans()
        self.reward_list.append(reward)
        self.total_reward += reward

        if len(self.state_list) >= MULTI_STEP and isTrain:
            self.append_sample(net, tgt_net, self.state_list[-MULTI_STEP], self.action_list[-MULTI_STEP],
                               np.sum(self.reward_list[-MULTI_STEP:]), new_state, is_done)
            if use_addition:
                for _ in range(ADDITION_RATE_BOOST):
                    self.append_sample(net, tgt_net, self.state_list[-MULTI_STEP], self.action_list[-MULTI_STEP],
                                       np.sum(self.reward_list[-MULTI_STEP:]), new_state, is_done)
        self.state = new_state
        self.state_list.append(new_state)
        best_sol = self.env.best_sol
        random_list = []
        if is_done or (hard_mask and np.sum(self.env.current_sol) == self.env.N) or (
                ic and np.sum(self.env.best_sol) == inlier):
            # done_reward = self.total_reward
            done_reward = self.env.calc_score(self.env.best_sol, normalize=normalize)
            random_list = copy.deepcopy(self.random_list)
            # print(self.acc_list)
            self._reset()
        return done_reward, best_sol, random_list, pos


# double dqn 参数更新
def calc_loss_double(net, tgt_net, device="cpu"):
    mini_batch, idxs, is_weights = agent.exp_buffer.sample(BATCH_SIZE)
    mini_batch = np.array(mini_batch).transpose()
    # states, actions, rewards, dones, next_states = batch
    states = []
    next_states = []
    for i in range(len(mini_batch[0])):
        states.append(np.vstack(mini_batch[0][i]))
        next_states.append(np.vstack(mini_batch[3][i]))
    states = np.array(states)
    next_states = np.array(next_states)
    actions = list(mini_batch[1])
    rewards = list(mini_batch[2])
    dones = np.array(mini_batch[4], dtype=int)

    maxN = 0
    for i in range(len(states)):
        if int(np.sqrt(len(states[i]) - 2)) > maxN:
            maxN = int(np.sqrt(len(states[i]) - 2))

    new_states = []
    for i in range(len(states)):
        n = int(np.sqrt(len(states[i])))
        old_sol = np.reshape(states[i][0], (n, n))
        new_sol = np.pad(old_sol, ((0, maxN - n), (0, maxN - n)), 'constant', constant_values=0).reshape(1, maxN * maxN)
        old_mask = np.reshape(states[i][1], (n, n))
        new_mask = np.pad(old_mask, ((0, maxN - n), (0, maxN - n)), 'constant', constant_values=0).reshape(1,
                                                                                                           maxN * maxN)
        state = np.reshape(states[i][2:][:], (n, n, n, n))

        new_K = np.pad(state, ((0, maxN - n), (0, maxN - n), (0, maxN - n), (0, maxN - n)), 'constant',
                       constant_values=0)
        new_K = np.reshape(new_K, (maxN * maxN, maxN * maxN))
        new_state = np.concatenate((np.concatenate((new_sol, new_mask)), new_K))
        new_states.append(new_state)
        x = actions[i] // n
        y = actions[i] % n
        actions[i] = x * maxN + y

    new_next_states = []
    for i in range(len(next_states)):
        n = int(np.sqrt(len(next_states[i])))
        old_next_sol = np.reshape(next_states[i][0], (n, n))
        new_next_sol = np.pad(old_next_sol, ((0, maxN - n), (0, maxN - n)), 'constant', constant_values=0).reshape(1,
                                                                                                                   maxN * maxN)
        old_next_mask = np.reshape(next_states[i][1], (n, n))
        new_next_mask = np.pad(old_next_mask, ((0, maxN - n), (0, maxN - n)), 'constant', constant_values=0).reshape(1,
                                                                                                                     maxN * maxN)

        next_state = np.reshape(next_states[i][2:][:], (n, n, n, n))
        new_next_K = np.pad(next_state, ((0, maxN - n), (0, maxN - n), (0, maxN - n), (0, maxN - n)), 'constant',
                            constant_values=0)
        new_next_K = np.reshape(new_next_K, (maxN * maxN, maxN * maxN))
        new_next_state = np.concatenate((np.concatenate((new_next_sol, new_next_mask)), new_next_K))
        new_next_states.append(new_next_state)

    states_v = torch.FloatTensor(new_states).to(device)
    next_states_v = torch.FloatTensor(new_next_states).to(device)
    actions_v = torch.LongTensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v, hard_mask=hard_mask, normalize=normalize).gather(1, actions_v.unsqueeze(
        -1)).squeeze(-1)
    next_state_selected = net(next_states_v, hard_mask=hard_mask, normalize=normalize).detach().argmax(1)
    next_state_values = tgt_net(next_states_v, hard_mask=hard_mask, normalize=normalize).gather(1,
                                                                                                next_state_selected.unsqueeze(
                                                                                                    -1)).squeeze(-1)
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v

    errors = torch.abs(expected_state_action_values - state_action_values).detach().cpu().numpy()
    for i in range(BATCH_SIZE):
        idx = idxs[i]
        agent.exp_buffer.update(idx, errors[i])

    return (torch.FloatTensor(is_weights).to(device) * torch.nn.functional.mse_loss(
        expected_state_action_values.to(device),
        state_action_values.to(device))).mean()


if __name__ == "__main__":
    isTrain = True
    cls = 'Car'
    time0 = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=1, type=int, help="Enable cuda")
    parser.add_argument("--train", default=isTrain, type=int, help="Training or just testing")
    parser.add_argument("--support", default=0.0, type=float, help="Ratio of additional information")
    parser.add_argument("--cls", default=cls, type=str)
    parser.add_argument("--load_from_cls", default=cls, type=str)
    parser.add_argument("--gamma", default=GAMMA, type=float)
    parser.add_argument("--bs", default=BATCH_SIZE, type=int)
    parser.add_argument("--rs", default=REPLAY_SIZE, type=int)
    parser.add_argument("--lr", default=LEARNING_RATE, type=float)
    parser.add_argument("--sync", default=SYNC_TARGET_FRAMES, type=int)
    parser.add_argument("--ls", default=REPLAY_START_SIZE, type=int)
    parser.add_argument("--ms", default=MULTI_STEP, type=int)
    parser.add_argument("--ed", default=EPSILON_DECAY_LAST_FRAME, type=int)
    parser.add_argument("--eb", default=EPSILON_BONUS, type=float)
    parser.add_argument("--es", default=EPSILON_START, type=float)
    parser.add_argument("--ef", default=EPSILON_FINAL, type=float)
    parser.add_argument("--units", default=UNITS, type=int)
    parser.add_argument("--hs", default=HIDDEN_SIZE, type=int)
    parser.add_argument("--t", default=T, type=int)
    parser.add_argument("--ad", default=ADDITION_RATE_DECAY, type=float)
    parser.add_argument("--af", default=ADDITION_RATE_FINAL, type=float)
    parser.add_argument("--ab", default=ADDITION_RATE_BOOST, type=int)
    parser.add_argument("--b", default=6, type=int)  # beam search size
    parser.add_argument("--d", default=0, type=int)  # whether to use the dynamic embedding
    parser.add_argument("--outlier", default=2, type=int)  # the number of outliers
    parser.add_argument("--normalize", default=1, type=int)  # whether to use the affinity regularization
    parser.add_argument("--hard_mask", default=1, type=int)  # whether to enable the revocable framework
    parser.add_argument("--inlier_count", default=0, type=int)  # whether to use the inlier count information
    parser.add_argument("--inlier", default=10, type=int)  # the inlier count information if used
    args = parser.parse_args()
    inlier = args.inlier
    normalize = args.normalize
    hard_mask = args.hard_mask
    ic = args.inlier_count
    isTrain = args.train
    outlier = args.outlier
    cls = args.cls
    DYNAMIC = args.d
    GAMMA = args.gamma
    BATCH_SIZE = args.bs
    REPLAY_SIZE = args.rs
    LEARNING_RATE = args.lr
    SYNC_TARGET_FRAMES = args.sync
    REPLAY_START_SIZE = args.ls
    MULTI_STEP = args.ms
    EPSILON_DECAY_LAST_FRAME = args.ed
    EPSILON_BONUS = args.eb
    EPSILON_START = args.es
    EPSILON_FINAL = args.ef
    UNITS = args.units
    HIDDEN_SIZE = args.hs
    T = args.t
    ADDITION_RATE_DECAY = args.ad
    ADDITION_RATE_FINAL = args.af
    ADDITION_RATE_BOOST = args.ab
    b = args.b

    args.save_dir = "model/Willow-{}-{}-DQN-pytorch-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}/".format(
        cls, DYNAMIC, GAMMA, BATCH_SIZE, REPLAY_SIZE, LEARNING_RATE, SYNC_TARGET_FRAMES, REPLAY_START_SIZE,
        MULTI_STEP, EPSILON_DECAY_LAST_FRAME, EPSILON_BONUS, EPSILON_START, EPSILON_FINAL, UNITS,
        HIDDEN_SIZE, T, args.support, ADDITION_RATE_DECAY, ADDITION_RATE_FINAL, ADDITION_RATE_BOOST)
    args.log_dir = "log/Willow-{}-{}/Willow-{}-DQN-pytorch-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
        cls, DYNAMIC, cls, GAMMA, BATCH_SIZE, REPLAY_SIZE, LEARNING_RATE, SYNC_TARGET_FRAMES, REPLAY_START_SIZE,
        MULTI_STEP,
        EPSILON_DECAY_LAST_FRAME, EPSILON_BONUS, EPSILON_START, EPSILON_FINAL, UNITS, HIDDEN_SIZE, T, args.support,
        ADDITION_RATE_DECAY, ADDITION_RATE_FINAL, ADDITION_RATE_BOOST)
    args.load_dir = "model/Willow-{}-{}-DQN-pytorch-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}/".format(
        args.load_from_cls, DYNAMIC, GAMMA, BATCH_SIZE, REPLAY_SIZE, LEARNING_RATE, SYNC_TARGET_FRAMES,
        REPLAY_START_SIZE, MULTI_STEP,
        EPSILON_DECAY_LAST_FRAME, EPSILON_BONUS, EPSILON_START, EPSILON_FINAL, UNITS, HIDDEN_SIZE, T, args.support,
        ADDITION_RATE_DECAY, ADDITION_RATE_FINAL, ADDITION_RATE_BOOST)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.log_dir, datetime.now().strftime('%Y%m%d-%H%M%S') + ".log"))
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    am_ = []
    am = []
    gt_ = []

    device = torch.device("cuda" if args.cuda else "cpu")
    if isTrain:
        am_ = np.load('cache/{}_k_test_{}.npy'.format(cls, outlier))
        am = []
        gt_ = np.load('cache/{}_x_test_{}.npy'.format(cls, outlier))

    am_test_ = np.load('cache/{}_k_test_{}.npy'.format(cls, outlier))
    gt_test = np.load('cache/{}_x_test_{}.npy'.format(cls, outlier))
    gt = []
    opt = []
    opt_test = []
    envs = []
    sizes = []
    am_test = []
    training_threshold = 20
    if cls == 'bur':
        training_threshold = 27
    envs_test = []
    for i in range(len(am_test_)):
        a = - am_test_[i]
        a_min = np.min(a)
        for x in range(len(a)):
            for y in range(len(a[x])):
                if a[x][y] != 0 and x == y:
                    a[x][y] -= a_min
        am_test.append(a)
        gt_test[i] = gt_test[i].T
        envs_test.append(Environment(-am_test[i], -1))
        opt_test.append(envs_test[i].calc_score(gt_test[i], normalize))
        sizes.append(int(np.sqrt(len(am_test[i]))))
    if isTrain:
        for i in range(len(am_)):
            if int(np.sqrt(len(am_[i]))) < training_threshold:
                a = - am_[i]
                a_min = np.min(a)
                for x in range(len(a)):
                    for y in range(len(a[x])):
                        if a[x][y] != 0 and x == y:
                            a[x][y] -= a_min
                am.append(a)
                gt_[i] = gt_[i].T
                gt.append(gt_[i])
                sizes.append(int(np.sqrt(len(am[i]))))
                envs.append(Environment(am[i], 1))
                opt.append(envs[i].calc_score(gt[i], normalize))

    maxN = np.max(sizes)
    env = envs[0] if isTrain else Environment(am_test[0], 1)

    rrwm_sol = []
    for i in range(len(am)):
        if isTrain and args.support > 0:
            rrwm_raw_sol = rrwm_solver(torch.FloatTensor([am[i]]), sizes[i], rrwm_mask, [sizes[i]], [sizes[i]]).reshape(
                1, sizes[i], sizes[i])
            rrwm_sol.append(hungarian(rrwm_raw_sol).cpu().detach().numpy()[0])
        else:
            rrwm_sol.append(np.zeros((sizes[i], sizes[i])))
    time1 = datetime.now()

    net = dqn_model_r.DQN(env.observation_space.shape, env.action_dims, UNITS, HIDDEN_SIZE, T, device).to(device)
    tgt_net = dqn_model_r.DQN(env.observation_space.shape, env.action_dims, UNITS, HIDDEN_SIZE, T, device).to(device)
    writer = SummaryWriter(comment="-gm")
    # logger.info(net)
    # logger.info("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))

    buffer = Memory(REPLAY_SIZE)
    agent = Agent(env, buffer)
    # net.apply(agent.weights_init)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    mean_reward_e = []
    total_accs = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    best_mean_acc = None

    env_index = 0
    use_addition = False
    order = []
    while True:
        if not isTrain:
            break
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        addition_rate = max(epsilon * ADDITION_RATE_DECAY, ADDITION_RATE_FINAL)
        length = agent.length
        reward, best_sol, random_list, pos = agent.play_step(net, tgt_net, addition=rrwm_sol[env_index], order=order,
                                                             epsilon=epsilon, use_addition=use_addition, device=device)
        if reward is not None:
            recall = np.sum(best_sol * gt[env_index]) / (np.sum(gt[env_index]) + 1e-5)
            precision = np.sum(best_sol * gt[env_index]) / (np.sum(best_sol) + 1e-5)
            f1 = 2 * recall * precision / (recall + precision + 1e-5)
            total_accs.append(f1)
            total_rewards.append(reward / (opt[env_index] + 1e-5))
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            mean_acc = np.mean(total_accs[-100:])
            mean_reward_e.append(mean_reward)
            logger.info(
                "%d: done %d games, mean reward %.3f, mean f1 %.3f, eps %.2f, speed %.2f f/s, env %.2f, pos %d/%d" % (
                    frame_idx, len(total_rewards), mean_reward, mean_acc, epsilon,
                    speed, env_index, pos[0], length))
            logger.info("Current Solution: {}; Matched Pairs:{}, Ans: {:.4}, F1: {:.4}".format(
                [(str(np.argmax(best_sol[i])) + random_list[i]) if i < len(random_list) else "^" for i in
                 range(len(best_sol))], np.sum(best_sol), reward / opt[env_index], f1))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.save_dir + "best.dat")
                if best_mean_reward is not None:
                    logger.info("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward

            if best_mean_acc is None or best_mean_acc < mean_acc:
                torch.save(net.state_dict(), args.save_dir + "best_new.dat")
                if best_mean_acc is not None:
                    logger.info("Best mean f1 updated %.3f -> %.3f, model saved" % (best_mean_acc, mean_acc))
                best_mean_acc = mean_acc

            env_index = np.random.randint(0, len(am))
            agent.set_env(envs[env_index])

            # use_addition = False
            # if np.random.random() < addition_rate:
            #     use_addition = True
            #     order = np.random.permutation(maxN)

            if frame_idx > EPSILON_DECAY_LAST_FRAME * 2:
                logger.info("Solved in %d frames!" % frame_idx)
                time2 = datetime.now()
                logger.info('Training time: %s Seconds' % (time2 - time1))
                break

        if agent.exp_buffer.tree.n_entries < REPLAY_START_SIZE:
            continue

        if frame_idx % 5 != 0:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        # logger.info("-----learning start----------")
        optimizer.zero_grad()
        loss_t = calc_loss_double(net, tgt_net, device=device).to(device)
        loss_t.backward()
        optimizer.step()

    sol = []
    test_reward = []
    test_acc = []
    net.load_state_dict(torch.load(args.load_dir + "best_new.dat"))


    def takeSecond(elem):
        return -elem[1]


    def beam_search(env_b, b, gt):
        beams = [[copy.deepcopy(env_b), 0, 0, 0] for _ in range(b)]

        e = np.zeros((agent.env.N, agent.env.N))
        for j1 in range(agent.env.N):
            e[j1][j1] = 1
        o = [j for j in range(agent.env.N)]
        check = 0
        c = 0
        while True:
            c += 1
            beam_list = []
            for env, r, _, _ in beams:
                for j in range(b):
                    env_ = copy.deepcopy(env)
                    env_.max_rounds = 20
                    agent.set_env(env_, reset=False)
                    epsilon = 0.0
                    reward, best_sol, random_list, pos = agent.play_step(net, tgt_net, addition=e, order=o, epsilon=0,
                                                                         use_addition=use_addition, device=device, k=j)
                    if reward is not None:
                        check = 1
                    beam_list.append(
                        [env_, env_.calc_score(env_.best_sol, normalize=normalize) if reward is None else reward,
                         best_sol, pos[0]])
                    reward_0 = env_.calc_score(env_.current_sol, 0) / env_.calc_score(gt, 0)
                    reward_1 = env_.calc_score(env_.current_sol, normalize) / env_.calc_score(gt, normalize)
                    recall = np.sum(env_.current_sol * gt) / np.sum(gt)
                    precision = np.sum(env_.current_sol * gt) / np.sum(env_.current_sol)
                    f1 = 2 * recall * precision / (recall + precision + 1e-5)
                    # print(reward_1)

            beam_list.sort(key=takeSecond)
            beams = beam_list[0:b]
            reward, best_sol, best_pos = beams[0][1], beams[0][2], beams[0][3]
            if check:
                return best_sol, best_pos


    # beam search
    use_addition = False

    time3 = datetime.now()
    test_size = []

    for i in range(len(am_test)):
        # if i != 6:
        #     continue
        if cls == 'tai' and i >= 8:
            break
        best_reward = 0
        best_f1 = 0
        best_size = 0
        best_pos = 0
        for k in range(b):
            best_sol, pos = beam_search(envs_test[i], k + 1, gt_test[i])
            reward = envs_test[i].calc_score(best_sol, normalize)
            recall = np.sum(best_sol * gt_test[i]) / np.sum(gt_test[i])
            precision = np.sum(best_sol * gt_test[i]) / np.sum(best_sol)
            f1 = 2 * recall * precision / (recall + precision + 1e-5)
            if reward > best_reward:
                best_reward = reward
                best_f1 = f1
                best_size = np.sum(best_sol)
                best_pos = pos
        test_reward.append(best_reward / (opt_test[i] + 1e-5))
        test_acc.append(best_f1)
        test_size.append(best_size)
        print(
            "Class:{}, env_{}/{}, rate_{:.4f}, ans_{:.4f}, size_{}/{}, gt_{:.4f}, f1_{:.4f}, pos_{}".format(cls, i,
                                                                                                            len(am_test),
                                                                                                            test_reward[
                                                                                                                -1],
                                                                                                            best_reward,
                                                                                                            best_size,
                                                                                                            np.sum(
                                                                                                                gt_test[
                                                                                                                    i]),
                                                                                                            opt_test[i],
                                                                                                            best_f1,
                                                                                                            best_pos))
        logger.info(
            "Class:{}, env_{}/{}, rate_{:.4f}, ans_{:.4f}, size_{}/{}, gt_{:.4f}, f1_{:.4f}, pos_{}".format(cls, i,
                                                                                                            len(am_test),
                                                                                                            test_reward[
                                                                                                                -1],
                                                                                                            best_reward,
                                                                                                            best_size,
                                                                                                            np.sum(
                                                                                                                gt_test[
                                                                                                                    i]),
                                                                                                            opt_test[i],
                                                                                                            best_f1,
                                                                                                            best_pos))

    time4 = datetime.now()
    logger.info('Consuming time: %s Seconds' % (time4 - time3))
    logger.info(
        'Class:{}, ans:{}, f1:{}, size:{}'.format(cls, np.mean(test_reward), np.mean(test_acc), np.mean(test_size)))
    print('Consuming time: %s Seconds' % (time4 - time3))
    print('Class:{}, ans:{}, f1:{}, size:{}'.format(cls, np.mean(test_reward), np.mean(test_acc), np.mean(test_size)))
    writer.close()
    writer.close()
    output_unary = np.array(output_unary)
    np.save("cache_w_u/unary_{}_{}.npy".format(cls, outlier), output_unary)
