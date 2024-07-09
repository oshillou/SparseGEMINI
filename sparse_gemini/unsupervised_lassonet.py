import torch
from torch import nn
from torch.nn import functional as F


class LambdaPath:
    def __init__(self, lambda_start, lambda_multiplier):
        self.lambda_ = lambda_start
        self.lambda_multiplier = lambda_multiplier

    def get_lambda(self):
        return self.lambda_

    def next(self):
        self.lambda_ *= self.lambda_multiplier


class Model(nn.Module):
    def __init__(self, *dims, dropout=None, M=10):
        super().__init__()
        print(f"I received the dims {dims}")
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.skip = nn.Linear(dims[0], dims[-1], bias=False)

        self.M = M

    def forward(self, inp):
        current_layer = inp
        result = self.skip(inp)
        for theta in self.layers:
            current_layer = theta(current_layer)
            if theta is not self.layers[-1]:
                if self.dropout is not None:
                    current_layer = self.dropout(current_layer)
                current_layer = F.relu(current_layer)
        return result + current_layer

    def l1_regularisation_skip(self):
        return torch.norm(self.skip.weight.data, p=2, dim=0).sum()

    def input_mask(self):
        with torch.no_grad():
            return torch.norm(self.skip.weight.data, p=2, dim=0) != 0

    def selected_count(self):
        return self.input_mask().sum().item()

    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}

    def soft_threshold(self, l, x):
        return torch.sign(x) * torch.relu(torch.abs(x) - l)

    def sign_binary(self, x):
        ones = torch.ones_like(x)
        return torch.where(x >= 0, ones, -ones)

    def prox(self, *, lambda_, lambda_bar=0):
        v = self.skip.weight.data
        u = self.layers[0].weight.data

        onedim = len(v.shape) == 1
        if onedim:
            v = v.unsqueeze(-1)
            u = u.unsqueeze(-1)

        u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values

        k, batch = u.shape

        s = torch.arange(k + 1.0).view(-1, 1).to(v)
        zeros = torch.zeros(1, batch).to(u)

        a_s = lambda_ - self.M * torch.cat(
            [zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)]
        )

        norm_v = torch.norm(v, p=2, dim=0)

        x = F.relu(1 - a_s / norm_v) / (1 + s * self.M ** 2)

        w = self.M * x * norm_v
        intervals = self.soft_threshold(lambda_bar, u_abs_sorted)
        lower = torch.cat([intervals, zeros])

        idx = torch.sum(lower > w, dim=0).unsqueeze(0)

        x_star = torch.gather(x, 0, idx).view(1, batch)
        w_star = torch.gather(w, 0, idx).view(1, batch)

        beta_star = x_star * v
        theta_star = self.sign_binary(u) * torch.min(self.soft_threshold(lambda_bar, u.abs()), w_star)

        if onedim:
            beta_star.squeeze_(-1)
            theta_star.squeeze_(-1)

        self.skip.weight.data = beta_star
        self.layers[0].weight.data = theta_star
