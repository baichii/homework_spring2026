"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
import torch.nn.functional as F


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        hidden_dims = (state_dim, ) + hidden_dims
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
        self.out_layer = nn.Linear(hidden_dims[-1], chunk_size * action_dim)

    def forward(self, state: torch.Tensor):
        x = state
        for layer in self.layers:
            x = layer(x)
        x = self.out_layer(x)
        x = x.reshape(-1, self.chunk_size, self.action_dim)
        return x

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.forward(state)
        loss = F.mse_loss(pred, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.forward(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        hidden_dims = (state_dim + chunk_size * action_dim + 1, ) + hidden_dims

        self.layers = nn.ModuleList()
        for in_channel, out_channel in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.append(nn.Linear(in_channel, out_channel))
            self.layers.append(nn.ReLU())

        self.output = nn.Linear(hidden_dims[-1], chunk_size * action_dim)

    def forward(self, state, x_t, t):
        batch_size = state.shape[0]
        x_t_flat = x_t.reshape(batch_size, -1)
        t = t.view(batch_size, 1)
        x = torch.cat([state, x_t_flat, t], dim=-1)
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x.reshape(batch_size, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]

        x_0 = torch.rand_like(action_chunk)
        t = torch.rand(batch_size, device=state.device, dtype=state.dtype)
        t_broadcast = t.view(batch_size, 1, 1)

        x_t = (1 - t_broadcast) * x_0 + t_broadcast * action_chunk

        target = action_chunk - x_0

        pred = self.forward(state, x_t, t)
        return F.mse_loss(pred, target)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        x = torch.randn(
            batch_size, self.chunk_size, self.action_dim, device=state.device
        )

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full(
                (batch_size,),
                step / num_steps,
                device=state.device,
                dtype=state.dtype,
            )
            v = self.forward(state, x, t)
            x = x + dt * v

        return x


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
