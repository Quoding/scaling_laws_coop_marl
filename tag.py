import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from pettingzoo.mpe import simple_tag_v2
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

from utils import *
N_AGENTS = 4
NAMES = ["pred_1", "pred_2", "pred_3", "prey_1"]

def train_agent(
    args: argparse.Namespace = get_args(),
    agents: Optional[Tuple[BasePolicy]] = None,
    optims: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:
    env = simple_tag_v2
    # ======== environment setup =========
    train_envs = DummyVectorEnv(
        [lambda: get_env(env) for _ in range(args.training_num)]
    )
    test_envs = DummyVectorEnv([lambda: get_env(env) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== agent setup =========
    policy, optim, agents, n_params = get_agents(env, args, agents, optims=optims)

    # print(policy, optim, agents)

    # ======== collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=False,
    )
    test_collector = Collector(policy, test_envs)
    # policy.set_eps(1)
    # train_collector.collect(n_step=args.batch_size * args.training_num)

    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, "tag", "ppo", str("_".join(str(x) for x in args.hidden_sizes)), str(args.seed))
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # ======== callback functions used during training =========
    def save_best_fn(policy):
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir,
                "tag",
                "ppo",
                "best",
                "n=" + str(n_params),
                "seed=" + str(args.seed),
            )
        os.makedirs(model_save_path, exist_ok=True)

        for i in range(N_AGENTS):
            torch.save(
                policy.policies[agents[i]].state_dict(),
                model_save_path + f"/{NAMES[i]}.pth",
            )

    def reward_metric(rews):
        return sum(rews[:, :3])  # Maximize hits on prey

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # ckpt_path = os.path.join(log_path, "checkpoint_.pth")
        # Example: saving by epoch num
        # if hasattr(args, "model_save_path"):
        #     model_save_path = args.model_save_path
        # else:
        if epoch % 100 != 0:
            return
        
        model_save_path = os.path.join(
            args.logdir,
            "tag",
            "ppo",
            "cp",
            "n=" + str(n_params),
            "seed=" + str(args.seed),
            f"epoch={epoch}",
        )
        os.makedirs(model_save_path, exist_ok=True)
        for i in range(N_AGENTS):
            torch.save(
                policy.policies[agents[i]].state_dict(),
                model_save_path + f"/{NAMES[i]}.pth",
            )
        return model_save_path

    # trainer
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,  # Number training rounds on the same collected data
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        episode_per_collect=args.episode_per_collect,  # Number of steps between updates of networks
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    # policies = [policy.policies[agents[i]] for i in range(len(agents))]

    return result, policy


# ======== a test function that tests a pre-trained agent ======
def watch(
    args: argparse.Namespace = get_args(),
    agents: Optional[Tuple[BasePolicy]] = None,
) -> None:
    env_name = simple_tag_v2
    env = get_env(env=simple_tag_v2, render_mode="human")
    env = DummyVectorEnv([lambda: env])
    policy, optim, agents = get_agents(env_name, args, agents=agents)
    policy.eval()
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, :3].mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = get_args()

    result, agent = train_agent(args)
    if args.watch:
        watch(args)
