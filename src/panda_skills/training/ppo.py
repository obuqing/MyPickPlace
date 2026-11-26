"""GPU-parallel PPO for any vectorized ManiSkill skill (state observations).

Everything — physics, rollout buffers, advantage estimation and the network — stays on
the GPU, so thousands of parallel envs train at full simulator speed. The algorithm
follows CleanRL/ManiSkill's PPO baseline: tanh MLP actor/critic, state-independent
log-std, GAE, clipped surrogate, early stop on approximate KL.

Each run writes to ``<output_dir>/<skill>/<run_name>/``::

    config.json      resolved hyper-parameters and env settings
    metrics.jsonl    one line per training iteration / evaluation
    latest.pt        most recent checkpoint (for --checkpoint resume)
    best.pt          checkpoint with the highest evaluation success rate
    ckpt_<step>.pt   periodic checkpoints
    videos/          evaluation videos when --capture-video is set
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import json
import math
from pathlib import Path
import random
import time
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

from panda_skills.paths import CHECKPOINTS_DIR
from panda_skills.skills import get_skill


@dataclass
class PPOConfig:
    skill: str = "peg_insertion"
    run_name: Optional[str] = None
    seed: int = 1
    output_dir: Path = CHECKPOINTS_DIR
    control_mode: Optional[str] = None
    """Overrides the skill's default controller (e.g. pd_ee_delta_pose)."""

    # Environment / rollout
    total_timesteps: int = 50_000_000
    num_envs: int = 1024
    num_steps: int = 50
    """Rollout length per env per iteration; batch size = num_envs * num_steps."""
    partial_reset: bool = True
    """Reset each env as soon as it terminates (on success) instead of at the horizon."""
    reconfiguration_freq: Optional[int] = None
    """How often training envs re-sample object geometry (None = never, after the first reset)."""

    # Evaluation / checkpointing
    num_eval_envs: int = 16
    num_eval_steps: int = 100
    eval_reconfiguration_freq: int = 1
    eval_freq: int = 25
    """Evaluate (and checkpoint) every N iterations."""
    save_freq: int = 100
    """Keep a numbered ckpt_<step>.pt every N iterations; latest.pt/best.pt are always updated."""
    capture_video: bool = False

    # PPO
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.1
    reward_scale: float = 1.0
    hidden_size: int = 256

    # Resume / evaluation-only
    checkpoint: Optional[Path] = None
    """Resume training from (or, with --evaluate, evaluate) this checkpoint."""
    evaluate: bool = False
    cuda: bool = True
    gpu_memory_fraction: Optional[float] = None
    """Cap PyTorch's share of GPU memory (0-1). PhysX sim buffers are allocated outside this cap."""


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim, hidden, out_dim, out_std):
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden)), nn.Tanh(),
        layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        layer_init(nn.Linear(hidden, out_dim), std=out_std),
    )


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.critic = mlp(obs_dim, hidden, 1, out_std=1.0)
        self.actor_mean = mlp(obs_dim, hidden, act_dim, out_std=0.01 * math.sqrt(2))
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), -0.5))

    def get_value(self, obs):
        return self.critic(obs)

    def get_action(self, obs, deterministic=False):
        mean = self.actor_mean(obs)
        if deterministic:
            return mean
        return Normal(mean, self.actor_logstd.expand_as(mean).exp()).sample()

    def get_action_and_value(self, obs, action=None):
        mean = self.actor_mean(obs)
        dist = Normal(mean, self.actor_logstd.expand_as(mean).exp())
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(obs)


def make_envs(config, num_envs, reconfiguration_freq, ignore_terminations, video_dir=None):
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    import gymnasium as gym

    skill = get_skill(config.skill)
    if not skill.vectorized:
        raise ValueError(f"Skill {skill.name!r} is not a GPU-vectorizable ManiSkill env; PPO here needs one")
    overrides = dict(num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, sim_backend="physx_cuda",
                     render_mode="rgb_array" if video_dir else None)
    if config.control_mode:
        overrides["control_mode"] = config.control_mode
    env = skill.make(**overrides)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    if video_dir:
        env = RecordEpisode(env, output_dir=str(video_dir), save_trajectory=False,
                            max_steps_per_video=config.num_eval_steps, video_fps=30)
    return ManiSkillVectorEnv(env, num_envs, ignore_terminations=ignore_terminations, record_metrics=True)


@torch.no_grad()
def evaluate(agent, eval_envs, num_eval_steps):
    """Deterministic rollouts; returns per-episode metric means (success_once, return, ...)."""
    obs, _ = eval_envs.reset()
    metrics = defaultdict(list)
    for _ in range(num_eval_steps):
        obs, _, _, _, infos = eval_envs.step(agent.get_action(obs, deterministic=True))
        if "final_info" in infos:
            mask = infos["_final_info"]
            for key, value in infos["final_info"]["episode"].items():
                metrics[key].append(value[mask].float())
    summary = {key: torch.cat(values).mean().item() for key, values in metrics.items()}
    summary["episodes"] = int(torch.cat(metrics["return"]).numel()) if metrics else 0
    return summary


def save_checkpoint(path, agent, optimizer, config, iteration, global_step, best_success, obs_dim, act_dim):
    torch.save(dict(agent=agent.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration,
                    global_step=global_step, best_success=best_success, obs_dim=obs_dim, act_dim=act_dim,
                    config=json.loads(json.dumps(asdict(config), default=str))), path)


def train(config: PPOConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("GPU-parallel ManiSkill PPO requires CUDA (sim_backend=physx_cuda)")
    if config.gpu_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction, torch.cuda.current_device())

    run_name = config.run_name or datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{config.seed}"
    run_dir = Path(config.output_dir) / config.skill / run_name
    if config.checkpoint is not None and config.run_name is None:
        run_dir = Path(config.checkpoint).resolve().parent  # resume in place
    if config.evaluate:
        if config.checkpoint is None:
            raise SystemExit("--evaluate requires --checkpoint")
        run_dir = Path(config.checkpoint).resolve().parent / ("eval_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=config.checkpoint is not None and not config.evaluate)
    video_dir = run_dir / "videos" if config.capture_video else None

    eval_envs = make_envs(config, config.num_eval_envs, config.eval_reconfiguration_freq,
                          ignore_terminations=True, video_dir=video_dir)
    obs_dim = int(np.prod(eval_envs.single_observation_space.shape))
    act_dim = int(np.prod(eval_envs.single_action_space.shape))
    agent = Agent(obs_dim, act_dim, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    start_iteration, global_step, best_success = 1, 0, -1.0
    if config.checkpoint is not None:
        state = torch.load(config.checkpoint, map_location=device, weights_only=False)
        agent.load_state_dict(state["agent"])
        if not config.evaluate:
            optimizer.load_state_dict(state["optimizer"])
            start_iteration = state["iteration"] + 1
            global_step = state["global_step"]
            best_success = state.get("best_success", -1.0)
        print(f"Loaded {config.checkpoint} (step {state['global_step']})", flush=True)

    eval_envs.reset(seed=config.seed + 10_000)
    if config.evaluate:
        agent.eval()
        summary = evaluate(agent, eval_envs, config.num_eval_steps)
        (run_dir / "evaluation.json").write_text(json.dumps(dict(checkpoint=str(config.checkpoint), **summary), indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        eval_envs.close()
        return summary

    envs = make_envs(config, config.num_envs, config.reconfiguration_freq, ignore_terminations=not config.partial_reset)
    (run_dir / "config.json").write_text(json.dumps(dict(
        asdict(config), run_dir=str(run_dir), env_id=get_skill(config.skill).env_id,
        obs_dim=obs_dim, act_dim=act_dim, torch=torch.__version__,
    ), indent=2, default=str))
    metrics_file = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")

    def log(record):
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    batch_size = config.num_envs * config.num_steps
    minibatch_size = batch_size // config.num_minibatches
    num_iterations = config.total_timesteps // batch_size
    action_low = torch.as_tensor(envs.single_action_space.low, device=device)
    action_high = torch.as_tensor(envs.single_action_space.high, device=device)

    obs = torch.zeros((config.num_steps, config.num_envs, obs_dim), device=device)
    actions = torch.zeros((config.num_steps, config.num_envs, act_dim), device=device)
    logprobs = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards = torch.zeros_like(logprobs)
    dones = torch.zeros_like(logprobs)
    values = torch.zeros_like(logprobs)

    next_obs, _ = envs.reset(seed=config.seed)
    next_done = torch.zeros(config.num_envs, device=device)
    print(f"Training {config.skill} -> {run_dir}\n  {num_iterations} iterations x {batch_size} steps "
          f"({config.num_envs} envs x {config.num_steps})", flush=True)

    def checkpoint(iteration, eval_summary=None):
        nonlocal best_success
        args = (agent, optimizer, config, iteration, global_step)
        if eval_summary is not None and eval_summary.get("success_once", -1) > best_success:
            best_success = eval_summary["success_once"]
            save_checkpoint(run_dir / "best.pt", *args, best_success, obs_dim, act_dim)
        save_checkpoint(run_dir / "latest.pt", *args, best_success, obs_dim, act_dim)

    last_episode = {}  # most recent training-episode means; rollouts shorter than the horizon may finish none

    iteration = start_iteration - 1
    try:
        for iteration in range(start_iteration, num_iterations + 1):
            if (iteration - 1) % config.eval_freq == 0:
                agent.eval()
                summary = evaluate(agent, eval_envs, config.num_eval_steps)
                log(dict(kind="eval", iteration=iteration, global_step=global_step, **summary))
                print(f"[eval] step={global_step} success_once={summary.get('success_once', float('nan')):.3f} "
                      f"return={summary.get('return', float('nan')):.2f}", flush=True)
                checkpoint(iteration - 1 if iteration > 1 else 0, summary)

            if config.anneal_lr:
                optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1) / num_iterations) * config.learning_rate

            agent.eval()
            started = time.perf_counter()
            # Values of the true next state for envs that ended (auto-reset replaces next_obs).
            # Like ManiSkill's baseline we bootstrap on success terminations too: with dense
            # positive rewards, cutting value at success would teach the agent to avoid succeeding.
            final_values = torch.zeros((config.num_steps, config.num_envs), device=device)
            episode_stats = defaultdict(list)
            for step in range(config.num_steps):
                global_step += config.num_envs
                obs[step], dones[step] = next_obs, next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step], actions[step], logprobs[step] = value.flatten(), action, logprob
                next_obs, reward, terminations, truncations, infos = envs.step(torch.clamp(action, action_low, action_high))
                next_done = torch.logical_or(terminations, truncations).float()
                rewards[step] = reward.view(-1) * config.reward_scale
                if "final_info" in infos:
                    mask = infos["_final_info"]
                    for key, value in infos["final_info"]["episode"].items():
                        episode_stats[key].append(value[mask].float())
                    with torch.no_grad():
                        final_values[step, mask] = agent.get_value(infos["final_observation"][mask]).view(-1)
            rollout_time = time.perf_counter() - started

            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                last_gae = 0
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        next_not_done, next_values = 1.0 - next_done, next_value
                    else:
                        next_not_done, next_values = 1.0 - dones[t + 1], values[t + 1]
                    real_next_values = next_not_done * next_values + final_values[t]
                    delta = rewards[t] + config.gamma * real_next_values - values[t]
                    advantages[t] = last_gae = delta + config.gamma * config.gae_lambda * next_not_done * last_gae
                returns = advantages + values

            b_obs, b_actions = obs.reshape(-1, obs_dim), actions.reshape(-1, act_dim)
            b_logprobs, b_advantages = logprobs.reshape(-1), advantages.reshape(-1)
            b_returns, b_values = returns.reshape(-1), values.reshape(-1)

            agent.train()
            started = time.perf_counter()
            clipfracs, approx_kl = [], torch.tensor(0.0)
            for _ in range(config.update_epochs):
                for idx in torch.randperm(batch_size, device=device).split(minibatch_size):
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[idx], b_actions[idx])
                    logratio = newlogprob - b_logprobs[idx]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())
                    if config.target_kl is not None and approx_kl > config.target_kl:
                        break
                    mb_adv = b_advantages[idx]
                    if config.norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(-mb_adv * ratio,
                                        -mb_adv * ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef)).mean()
                    newvalue = newvalue.view(-1)
                    if config.clip_vloss:
                        clipped = b_values[idx] + (newvalue - b_values[idx]).clamp(-config.clip_coef, config.clip_coef)
                        v_loss = 0.5 * torch.max((newvalue - b_returns[idx]) ** 2, (clipped - b_returns[idx]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[idx]) ** 2).mean()
                    loss = pg_loss - config.ent_coef * entropy.mean() + config.vf_coef * v_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    optimizer.step()
                if config.target_kl is not None and approx_kl > config.target_kl:
                    break
            update_time = time.perf_counter() - started

            var_y = b_returns.var()
            record = dict(kind="train", iteration=iteration, global_step=global_step,
                          lr=optimizer.param_groups[0]["lr"], value_loss=v_loss.item(), policy_loss=pg_loss.item(),
                          entropy=entropy.mean().item(), approx_kl=approx_kl.item(),
                          clipfrac=float(np.mean(clipfracs)) if clipfracs else 0.0,
                          explained_variance=(1 - (b_values - b_returns).var() / var_y).item() if var_y > 0 else float("nan"),
                          sps=int(batch_size / (rollout_time + update_time)), rollout_time=rollout_time, update_time=update_time)
            record.update({f"episode_{k}": torch.cat(v).mean().item() for k, v in episode_stats.items()})
            last_episode.update({k: v for k, v in record.items() if k.startswith("episode_")})
            log(record)
            if iteration % config.save_freq == 0:
                save_checkpoint(run_dir / f"ckpt_{global_step}.pt", agent, optimizer, config, iteration, global_step,
                                best_success, obs_dim, act_dim)
                checkpoint(iteration)
            if iteration % 5 == 0 or iteration == start_iteration:
                print(f"[train] it={iteration}/{num_iterations} step={global_step} sps={record['sps']} "
                      f"success_once={last_episode.get('episode_success_once', float('nan')):.3f} "
                      f"return={last_episode.get('episode_return', float('nan')):.2f} kl={record['approx_kl']:.4f}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted; saving latest checkpoint", flush=True)
    finally:
        if iteration >= start_iteration:
            agent.eval()
            summary = evaluate(agent, eval_envs, config.num_eval_steps)
            log(dict(kind="eval", iteration=iteration, global_step=global_step, final=True, **summary))
            checkpoint(iteration, summary)
            print(f"[final eval] step={global_step} success_once={summary.get('success_once', float('nan')):.3f} "
                  f"best={best_success:.3f}\nCheckpoints: {run_dir}", flush=True)
        metrics_file.close()
        envs.close()
        eval_envs.close()
    return run_dir


def build_parser(defaults: PPOConfig) -> argparse.ArgumentParser:
    """Expose every PPOConfig field as --kebab-case flag, defaulting to `defaults`."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for f in fields(PPOConfig):
        flag, default = "--" + f.name.replace("_", "-"), getattr(defaults, f.name)
        kind = f.type if isinstance(f.type, type) else None
        if f.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif kind in (int, float, str, Path):
            parser.add_argument(flag, type=kind, default=default)
        else:  # Optional[...]
            inner = {Optional[int]: int, Optional[float]: float, Optional[str]: str, Optional[Path]: Path}[f.type]
            parser.add_argument(flag, type=lambda v, t=inner: None if v.lower() == "none" else t(v), default=default)
    return parser


def main(argv=None, defaults: Optional[PPOConfig] = None):
    defaults = defaults or PPOConfig()
    config = PPOConfig(**vars(build_parser(defaults).parse_args(argv)))
    return train(config)


if __name__ == "__main__":
    main()
