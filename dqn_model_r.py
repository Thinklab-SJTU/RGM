import torch
import torch.nn as nn
from utils.gm_solver import get_norm_affinity


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, units=64, hidden_size=64, T=3, device="cpu"):
        super(DQN, self).__init__()
        assert input_shape[1] * input_shape[2] == n_actions
        self.N = input_shape[1]
        self.units = units
        self.hidden_size = hidden_size
        self.T = T
        self.n_actions = n_actions
        self.device = device
        shape = [(1, self.units), (self.units, self.units), (self.units, self.units), (1, self.units), (1, self.units)]
        self.W = []
        for i in range(len(shape)):
            if self.device == "cpu":
                self.W.append(torch.autograd.Variable(torch.randn(shape[i][0], shape[i][1]), requires_grad=True))
            else:
                self.W.append(torch.autograd.Variable(torch.randn(shape[i][0], shape[i][1]), requires_grad=True).cuda())
            self.register_parameter("W_{}".format(i), torch.nn.Parameter(self.W[i]))

        self.fc_a = nn.Sequential(
            nn.Linear(self.units, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        )

        self.fc_v = nn.Sequential(
            nn.Linear(self.units, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        self.embedding = None

        # self.fc = nn.Sequential(
        #     nn.Linear(100, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, n_actions)
        # )

    def gnn(self, embedding, cur_sol, weights, adjacent, mask):
        for t in range(self.T):
            e1 = torch.matmul(cur_sol, self.W[0])
            e2_ = torch.bmm(adjacent, embedding) / self.n_actions
            e2 = torch.matmul(e2_, self.W[1])
            e3_ = torch.sum(torch.nn.functional.relu(torch.matmul(weights, self.W[3])), dim=2) / self.n_actions
            e3 = torch.matmul(e3_, self.W[2])
            # e4 = torch.matmul(mask, self.W[4])
            embedding = torch.nn.functional.relu(e1 + e2 + e3)
        return embedding

    def forward(self, x, use_dynamic_embedding=False, hard_mask=False, normalize=False):
        self.n_actions = x.shape[2]
        cur_sol = torch.reshape(x[:, 0, :], (-1, self.n_actions, 1))
        mask = x[:, 1, :]
        weights = x[:, 2: self.n_actions + 2, :]
        if normalize:
            for i in range(len(x)):
                weights[i] = get_norm_affinity(weights[i], self.N, self.N, torch.reshape(cur_sol[i], (self.N, self.N)))

        adjacent = (weights != 0).float()
        weights = torch.reshape(weights, (-1, self.n_actions, self.n_actions, 1))

        if self.device == "cpu":
            embedding = torch.zeros(len(x), self.n_actions, self.units)
        else:
            embedding = torch.zeros(len(x), self.n_actions, self.units).cuda()

        if not use_dynamic_embedding:
            self.embedding = None

        if self.embedding is None:
            output_ = self.gnn(embedding, cur_sol, weights, adjacent, torch.reshape(mask, (-1, self.n_actions, 1)))
            if use_dynamic_embedding:
                self.embedding = output_
        else:
            output_ = self.embedding
            mask_ = mask.reshape(1, x.shape[2], 1)
            mask_ = mask_.expand(1, x.shape[2], self.units)
            output_ = output_.mul(mask_)

        output_a = torch.reshape(self.fc_a(output_), (-1, self.n_actions))
        output_v = torch.reshape(self.fc_v(output_), (-1, self.n_actions))
        output_v = torch.sum(output_v, dim=1, keepdim=True)
        output = output_v + (output_a - torch.sum(output_a, dim=1, keepdim=True))
        if hard_mask:
            inf_mask = torch.log(mask)
        else:
            inf_mask = mask * 10
        output = output + inf_mask
        return output
