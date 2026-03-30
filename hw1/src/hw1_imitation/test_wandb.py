"""Train a simple CartPole agent to validate local wandb logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import copy
import shutil
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
import wandb

LOGDIR_PREFIX = "exp"


@dataclass
class Config:
    project: str = "cartpole_local-poc"
    entity: str | None = None
    run_name: str = "cartpole-ac"
    seed: int = 42

    env_id: str = "CartPole-v1"
    total_env_steps: int = 50_000
    gamma: float = 0.99
    lr: float = 3e-4
    hidden_dim: tuple[int, ...] = (128, 128)
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_episode_steps: int = 500

    log_interval: int = 200
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: tuple[int, ...],
    ) -> None:
        super().__init__()
        hidden_dims = (state_dim,) + hidden_dim
        layers: list[nn.Module] = []
        for in_channel, out_channel in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(in_channel, out_channel))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dims[-1], action_dim)
        self.value_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def sample(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logp, entropy, value


class Logger:
    """Minimal logger that mirrors the train.py logging flow without Push-T deps."""

    CSV_DISALLOWED_TYPES = (wandb.Image, wandb.Video, wandb.Histogram)

    def __init__(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f"Log directory {path} already exists.")
        path.mkdir(parents=True)
        self.path = path
        self.csv_path = path / "log.csv"
        self.header: list[str] | None = None
        self.rows: list[dict[str, Any]] = []

    def log(self, row: dict[str, Any], step: int) -> None:
        row["step"] = step
        if self.header is None:
            self.header = [
                key
                for key, value in row.items()
                if not isinstance(value, self.CSV_DISALLOWED_TYPES)
            ]
            with self.csv_path.open("w") as f:
                f.write(",".join(self.header) + "\n")
        filtered_row = {
            key: value
            for key, value in row.items()
            if not isinstance(value, self.CSV_DISALLOWED_TYPES)
        }
        with self.csv_path.open("a") as f:
            f.write(",".join([str(filtered_row.get(key, "")) for key in self.header]) + "\n")
        wandb.log(row, step=step)
        self.rows.append(copy.deepcopy(row))

    def dump_for_grading(self) -> None:
        wandb_dir = Path(wandb.run.dir).parent
        wandb.finish()
        shutil.copytree(wandb_dir, self.path / "wandb")


def parse_config(args: list[str] | None = None) -> Config:
    return tyro.cli(Config, args=args, description=__doc__)


def config_to_dict(config: Config) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_returns(rewards: list[float], gamma: float, device: torch.device) -> torch.Tensor:
    returns = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=device)


def run_training(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    env = gym.make(cfg.env_id, max_episode_steps=cfg.max_episode_steps)
    env.action_space.seed(cfg.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(obs_dim, action_dim, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    exp_name = f"{cfg.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=exp_name,
        config=config_to_dict(cfg),
    )
    logger = Logger(log_dir)

    env_steps = 0
    episode_idx = 0
    next_log_step = cfg.log_interval

    while env_steps < cfg.total_env_steps:
        obs, _ = env.reset(seed=cfg.seed + episode_idx)
        done = False

        episode_rewards: list[float] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        values: list[torch.Tensor] = []

        while not done and env_steps < cfg.total_env_steps:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, logp, entropy, value = model.sample(obs_tensor)

            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            episode_rewards.append(float(reward))
            log_probs.append(logp.squeeze(0))
            entropies.append(entropy.squeeze(0))
            values.append(value.squeeze(0))
            env_steps += 1

        returns = compute_returns(episode_rewards, cfg.gamma, device)
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)
        values_tensor = torch.stack(values)
        advantages = returns - values_tensor.detach()

        policy_loss = -(log_probs_tensor * advantages).mean()
        value_loss = F.mse_loss(values_tensor, returns)
        entropy = entropies_tensor.mean()
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        episode_idx += 1
        episode_return = float(sum(episode_rewards))
        episode_length = len(episode_rewards)

        if env_steps >= next_log_step or env_steps >= cfg.total_env_steps:
            logger.log(
                {
                    "train/loss": float(loss.item()),
                    "train/policy_loss": float(policy_loss.item()),
                    "train/value_loss": float(value_loss.item()),
                    "train/entropy": float(entropy.item()),
                    "train/episode_return": episode_return,
                    "train/episode_length": episode_length,
                    "train/episode": episode_idx,
                    "train/env_steps": env_steps,
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=env_steps,
            )
            next_log_step += cfg.log_interval

    env.close()
    logger.dump_for_grading()


def main() -> None:
    cfg = parse_config()
    run_training(cfg)


if __name__ == "__main__":
    main()
