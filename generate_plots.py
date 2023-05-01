# %%
import pickle
from argparse import Namespace
from copy import deepcopy
from os.path import exists
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tianshou
import torch
from causal_ccm.causal_ccm import ccm
from pettingzoo.mpe import simple_tag_v2
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.data.experimental import ExperimentFromDev
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    PPOPolicy,
    RandomPolicy,
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from tqdm import tqdm
from matplotlib.colors import hsv_to_rgb
from utils import *
from viz_config import *

plt.hsv()

TAU = 1  # Sensible default
ENV = simple_tag_v2
ENV_INSTANCE = get_env(simple_tag_v2)
NAMES = ENV_INSTANCE.agents
# MIN_PARAMS = 0
# MAX_PARAMS = 2112512
# START_HUE = 250 / 360
# END_HUE = 50 / 360
# SAT = 1
# VAL = 0.90
# scale_param_hue = lambda n: abs(
#     (n - MIN_PARAMS) / (MAX_PARAMS - MIN_PARAMS) * (END_HUE - START_HUE) + START_HUE
# )
# param_to_rgb = lambda n: hsv_to_rgb([scale_param_hue(n), SAT, VAL])
COLORS = [
    "tab:orange",
    "tab:purple",
    "tab:green",
    "tab:red",
    "tab:cyan",
    "tab:brown",
    "tab:pink",
]
STEPS_PER_EPOCH = 1000
# EPISODE_PER_UPDATE = 10  # translates to 1000 steps per update
N_EPOCHS = 1000
N_EPOCHS_BETWEEN_CPS = 100
SLICE_SIZE = STEPS_PER_EPOCH * N_EPOCHS_BETWEEN_CPS
NUM_SLICES = (N_EPOCHS // N_EPOCHS_BETWEEN_CPS) + 1
N_EP_EVAL = 10


def get_rewards(archs: list, seeds: list, save_loc=None):
    """Get rewards stored inside TFEvents in Tensorboard

    Args:
        archs (list): List of architectures to retrieve
        seeds (list): List of seeds to retrieve
        save_loc (str, optional): save location of the resulting dict. Defaults to None.

    Returns:
        dict: dictionary holding rewards for differents architectures and seeds
    """
    rewards = {}

    for arch in tqdm(archs):
        rewards[arch] = {}
        for seed in seeds:
            # Get TFEvents store with Tensorboard
            rewards[arch][seed] = []
            path = f"log/tag/ppo/{arch}/{seed}"
            event_acc = EventAccumulator(path)
            event_acc.Reload()

            # Iterate through them, select ones that match with SLICE_SIZE
            # to be coherent with number of CCM values.
            for event in event_acc.Scalars("test/reward"):
                # Event steps increment by the number of agents each time
                # So we need need to take the first slice every time to get
                # a step that lands on an evaluation step in the training pipeline.
                # e.g. 4 agents means every single loop of the agents is 4 steps and
                # not 1 step, although only one frame of the game is played
                if (event.step / len(NAMES)) % SLICE_SIZE == 0:
                    rewards[arch][seed].append(event.value)

    # Save rewards if needed
    if save_loc is not None:
        with open(save_loc, "wb") as f:
            pickle.dump(rewards, f)
    return rewards


def save_ccms(archs: list, seeds: list, save_loc=None):
    """Compute Convergent Cross-Mapping (CCM) for given architectures and seeds. Requires pretrained models.

    Args:
        archs (list): List of architectures to retrieve
        seeds (list): List of seeds to retrieve
        save_loc (str, optional): save location of the resulting dict. Defaults to None.

    Returns:
        dict: Dictionary containing CCMs for the given architectures and seeds
    """
    possible_random_agents = [0, 1, 2]
    for arch in tqdm(archs):
        if exists(save_loc + f"{arch=}.pkl"):
            print(f"Skipped {arch=}")
            continue
        ccms = {}
        ccms[arch] = {}
        for seed in seeds:
            # Retrieve arguments used for the targeted run
            path = f"log/tag/ppo/{arch}/{seed}"
            event_acc = EventAccumulator(path)
            event_acc.Reload()
            args = eval(
                event_acc.Tensors("args/text_summary")[0].tensor_proto.string_val[0]
            )
            # To load on device without GPU
            args.device = "cpu"

            for random_agent in possible_random_agents:
                assert (
                    random_agent < 4
                )  # Constraint for simple_tag_v2 environment. 4th agent is prey.

                # Get remaning agents that we can evaluate on CCM
                remaining_adv = deepcopy(possible_random_agents)
                remaining_adv.remove(random_agent)

                # Retrieve available checkpoints in directory
                checkpoints_path = f"log/tag/ppo/cp/{arch}/{seed}/"
                avail_checkpoints = sorted(
                    os.listdir(checkpoints_path), key=lambda x: int(x.split("=")[1])
                )
                avail_checkpoints_paths = [
                    checkpoints_path + checkpoint_name
                    for checkpoint_name in avail_checkpoints
                ]

                # Iterate through checkpoints, loading agents and setting one as random policy
                # then, collect episodes to compute CCM
                for current_checkpoint_path in avail_checkpoints_paths:
                    args.resume_path = current_checkpoint_path

                    # Load agent, set one as random
                    policy, optim, agents, n_params, flops_per_loop = get_agents(
                        ENV, args, override_agent=[random_agent]
                    )

                    # Setup data collection with Tianshou
                    test_env = DummyVectorEnv([lambda: get_env(ENV) for i in range(10)])
                    replay_buffer = VectorReplayBuffer(20_000, len(test_env))
                    collector = Collector(policy, test_env, replay_buffer)

                    ep = collector.collect(n_episode=N_EP_EVAL)  #

                    data = replay_buffer.sample(0)[0]
                    actions = {}

                    # Attribute actions to proper agent so CCM can construct proper manifolds
                    for name in NAMES:
                        agent_indices = data["obs"]["agent_id"] == name
                        actions[name] = data["act"][agent_indices]

                    L = len(actions[NAMES[0]])

                    # Iterate through remaining adversaries that aren't random
                    # compute CCM for each of them for the current random agent.
                    for adv in remaining_adv:
                        adv_name = f"adversary_{adv}"
                        save_name = f"adversary_random_{random_agent}_on_{adv_name}"

                        # Give basic structure to dict
                        if save_name not in ccms[arch].keys():
                            ccms[arch][save_name] = {}
                            for s in seeds:
                                ccms[arch]["n_params"] = n_params
                                ccms[arch]["n_flops"] = flops_per_loop
                                ccms[arch][save_name][s] = {}
                                ccms[arch][save_name][s]["correl"] = []
                                ccms[arch][save_name][s]["p-value"] = []

                        # np.seterr(all="raise")
                        # scipy.special.seterr(all="raise")           # Compute and store ccm
                        ccm1 = ccm(
                            actions[f"adversary_{random_agent}"],
                            actions[adv_name],
                            TAU,
                            E,
                            L,
                        )
                        correl, p = ccm1.causality()
                        ccms[arch][save_name][seed]["correl"].append(correl)
                        ccms[arch][save_name][seed]["p-value"].append(p)

        # Save ccms if needed
        if save_loc is not None:
            with open(save_loc + f"{arch=}.pkl", "wb") as f:
                pickle.dump(ccms, f)

    return


def plot_ccms(save_loc, archs, seeds, fig_save_loc_tweak=""):
    fig, ax = plt.subplots(1, 1)
    x_axis = [f"{SLICE_SIZE * i:.2e}" for i in range(1, NUM_SLICES)]
    sort_fn = key = lambda x: (int(x.split("_")[0]), len(x.split("_")))
    for i, arch in enumerate(sorted(archs, key=sort_fn)):
        with open(save_loc + f"{arch=}.pkl", "rb") as f:
            ccms = pickle.load(f)

        correls = []
        n_params = ccms[arch]["n_params"]
        for save_name in ccms[arch].keys():
            if save_name == "n_params" or save_name == "n_flops":
                continue
            n_obs = len(ccms[arch][save_name].keys())
            for seed in seeds:
                correls.append(ccms[arch][save_name][seed]["correl"])

        arch_label = arch.replace("_", "x")

        # Compute values to show
        correl_mean = np.nanmean(correls, axis=0)
        correl_ci = np.nanstd(correls, axis=0) / np.sqrt(n_obs)

        # Plot
        ax.fill_between(
            x_axis,
            correl_mean - correl_ci,
            correl_mean + correl_ci,
            alpha=0.3,
            color=COLORS[i],
        )
        ax.plot(x_axis, correl_mean, label=arch_label, color=COLORS[i])

    ax.set_xlabel("Number of environment interactions")
    ax.set_ylabel("Convergent X-mapping")
    ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    fig.legend()
    fig.savefig(f"{fig_save_loc_tweak}_{save_loc}_ccms.png")


def plot_rewards(save_loc, rewards_dict, seeds, fig_save_loc_tweak=""):
    fig, ax = plt.subplots(1, 1)
    x_axis = [f"{SLICE_SIZE * i:.2e}" for i in range(1, NUM_SLICES + 1)]

    sort_fn = key = lambda x: (int(x.split("_")[0]), len(x.split("_")))

    for i, arch in enumerate(sorted(rewards_dict.keys(), key=sort_fn)):
        rewards_list = []

        for seed in seeds:
            rewards_list.append(rewards_dict[arch][seed])

        arch_label = arch.replace("_", "x")

        # Compute values to show
        reward_mean = np.nanmean(rewards_list, axis=0)
        reward_ci = np.nanstd(rewards_list, axis=0) / np.sqrt(
            len(rewards_dict[arch].keys())
        )

        # Plot
        ax.fill_between(
            x_axis,
            reward_mean - reward_ci,
            reward_mean + reward_ci,
            alpha=0.3,
            color=COLORS[i],
        )
        ax.plot(x_axis, reward_mean, label=arch_label, color=COLORS[i])

    ax.set_xlabel("Number of environment interactions")
    ax.set_ylabel("Obtained reward")
    ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    fig.legend()
    fig.savefig(f"{fig_save_loc_tweak}_{save_loc}.png")


def plot_ccm_flops(save_loc, archs, seeds, fig_save_loc_tweak=""):
    """Plot ccms w.r.t. flops

    Args:
        save_loc (str): save location
        archs (list): architectures to plot
        seeds (list): seeds to plot
    """
    fig, ax = plt.subplots(1, 1)
    regression_params = []
    regression_correls = []
    sort_fn = key = lambda x: (int(x.split("_")[0]), len(x.split("_")))

    for i, arch in enumerate(sorted(archs, key=sort_fn)):
        with open(save_loc + f"{arch=}.pkl", "rb") as f:
            ccms = pickle.load(f)

        correls = []
        n_flops = ccms[arch]["n_flops"]
        x_axis = [SLICE_SIZE * i * n_flops for i in range(1, NUM_SLICES)]
        # print(ccms[arch].keys())
        for save_name in ccms[arch].keys():
            if save_name == "n_params" or save_name == "n_flops":
                continue
            n_obs = len(ccms[arch][save_name].keys())

            for seed in seeds:
                correls.append(ccms[arch][save_name][seed]["correl"])

        arch_label = arch.replace("_", "x")

        # Compute values to show
        correls = np.array(correls)
        correl_mean = np.nanmean(correls, axis=0)
        correl_ci = np.nanstd(correls, axis=0) / np.sqrt(n_obs)

        # Plot
        ax.fill_between(
            x_axis,
            correl_mean - correl_ci,
            correl_mean + correl_ci,
            alpha=0.3,
            color=COLORS[i],
        )
        ax.plot(x_axis, correl_mean, label=arch_label, color=COLORS[i])

    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Convergent X-mapping")
    # ax.tick_params(axis="x", labelrotation=90)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.legend()
    fig.savefig(f"{fig_save_loc_tweak}_{save_loc}_ccms_flops.png")


def plot_regr_ccm_parameters(save_loc, archs, seeds, fig_save_loc_tweak=""):
    """Plot best ccm (so, 1 checkpoint only) across all seeds for all archs

    Args:
        save_loc (str): save location
        archs (list): architectures to plot
        seeds (list): seeds to plot
    """
    fig, ax = plt.subplots(1, 1)
    # x_axis = [f"{SLICE_SIZE * i:.2e}" for i in range(1, NUM_SLICES)]
    regression_params = []
    regression_correls = []
    sort_fn = key = lambda x: (int(x.split("_")[0]), len(x.split("_")))

    for i, arch in enumerate(sorted(archs, key=sort_fn)):
        with open(save_loc + f"{arch=}.pkl", "rb") as f:
            ccms = pickle.load(f)

        correls = []
        n_params = ccms[arch]["n_params"]
        regression_params.append(n_params)
        # print(ccms[arch].keys())
        for save_name in ccms[arch].keys():
            if save_name == "n_params" or save_name == "n_flops":
                continue
            for seed in seeds:
                correls.append(ccms[arch][save_name][seed]["correl"])
        # Compute values to show
        correls = np.array(correls)
        correl_mean = np.nanmean(correls, axis=0)
        checkpoint_index = np.argmax(correl_mean)  # Get checkpoint with best ccm
        regression_correls.append(correl_mean[checkpoint_index])
        correl_mean_perseed = [
            correls[i :: len(seeds), checkpoint_index].mean() for i in range(len(seeds))
        ]

        ax.scatter(
            [n_params],
            [correl_mean[checkpoint_index]],
            color="black",
            zorder=2,
            label="Mean",
        )
        ax.scatter(
            [n_params] * len(seeds),
            correl_mean_perseed,
            color=COLORS[i],
            zorder=1,
        )

    z = np.polyfit(np.log(regression_params), regression_correls, 1)
    a, b = np.poly1d(z)
    xseq = np.linspace(min(regression_params), max(regression_params), num=100)
    f = lambda x: a * np.log(x) + b
    ax.plot(xseq, f(xseq), color="k", label=f"{a:.2} * log(x) + {b:.2}")

    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Convergent X-mapping")
    # ax.tick_params(axis="x", labelrotation=45)

    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    fig.tight_layout()
    fig.savefig(f"{fig_save_loc_tweak}_{save_loc}_ccms_regr_parameters.png")


def plot_regr_ccm_flops(save_loc, archs, seeds, fig_save_loc_tweak=""):
    """Plot best ccm (so, 1 checkpoint only) across all seeds for all archs

    Args:
        save_loc (str): save location
        archs (list): architectures to plot
        seeds (list): seeds to plot
    """
    fig, ax = plt.subplots(1, 1)
    # x_axis = [f"{SLICE_SIZE * i:.2e}" for i in range(1, NUM_SLICES)]
    regression_params = []
    regression_correls = []
    sort_fn = key = lambda x: (int(x.split("_")[0]), len(x.split("_")))

    for i, arch in enumerate(sorted(archs, key=sort_fn)):
        with open(save_loc + f"{arch=}.pkl", "rb") as f:
            ccms = pickle.load(f)

        correls = []
        n_flops = ccms[arch]["n_flops"]
        regression_params.append(n_flops)
        # print(ccms[arch].keys())
        for save_name in ccms[arch].keys():
            if save_name == "n_params" or save_name == "n_flops":
                continue
            for seed in seeds:
                correls.append(ccms[arch][save_name][seed]["correl"])
        # Compute values to show
        correls = np.array(correls)
        correl_mean = np.nanmean(correls, axis=0)
        checkpoint_index = np.argmax(correl_mean)  # Get checkpoint with best ccm
        regression_correls.append(correl_mean[checkpoint_index])
        correl_mean_perseed = [
            correls[i :: len(seeds), checkpoint_index].mean() for i in range(len(seeds))
        ]

        ax.scatter(
            [n_flops],
            [correl_mean[checkpoint_index]],
            color="black",
            zorder=2,
            label="Mean",
        )
        ax.scatter(
            [n_flops] * len(seeds),
            correl_mean_perseed,
            color=COLORS[i],
            zorder=1,
        )

    z = np.polyfit(np.log(regression_params), regression_correls, 1)
    a, b = np.poly1d(z)
    xseq = np.linspace(min(regression_params), max(regression_params), num=100)
    f = lambda x: a * np.log(x) + b
    ax.plot(xseq, f(xseq), color="k", label=f"{a:.2} * log(x) + {b:.2}")

    ax.set_xscale("log")
    ax.set_xlabel("Number of FLOPs per RL loop")
    ax.set_ylabel("Convergent X-mapping")
    # ax.tick_params(axis="x", labelrotation=45)

    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    fig.tight_layout()
    fig.savefig(f"{fig_save_loc_tweak}_{save_loc}_ccms_regr_flops.png")


def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def get_archs_seeds(_dir, seeds=None):
    archs = os.listdir(_dir)
    if "best" in archs:
        archs.remove("best")
    if "cp" in archs:
        archs.remove("cp")
    valid = []

    # Filter seeds which crashed, retrieve only good runs across all archs
    for arch in archs:
        path = _dir + "/" + arch + "/"

        if seeds is None:
            seeds = sorted(os.listdir(path), key=lambda x: int(x))

        dir_sizes = []
        for seed in seeds:
            path_seed = path + str(seed)
            dir_sizes.append(int(get_dir_size(path_seed)) // 1024)

        dir_sizes = np.array(dir_sizes)
        max_size = max(dir_sizes)
        # print(arch, dir_sizes)
        # input()
        valid_seeds = set(np.where(dir_sizes == max_size)[0])
        valid.append(valid_seeds)

    # for i, v in enumerate(valid):
    #     print(archs[i], v)
    # print(valid)
    final_set = valid[0]
    for seed_set in valid[1:]:
        final_set = seed_set & final_set

    final_set = list(final_set)

    return archs, np.array(seeds)[final_set]


def get_n_flops(archs, seeds, save_loc):
    """Post hoc gathering of number of flops per RL loop

    Args:
        archs (list): architectures for which to retrieve number of flops
        seeds (list): list of valid seeds, only first one will be taken anyway
        save_loc (str): save/load location
    """
    for arch in tqdm(archs):
        # if exists(save_loc + f"{arch=}.pkl"):
        # print(f"Skipped {arch=}")
        # continue
        with open(save_loc + f"{arch=}.pkl", "rb") as f:
            ccms = pickle.load(f)

        seed = seeds[0]
        # Retrieve arguments used for the targeted run
        path = f"log/tag/ppo/{arch}/{seed}"
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        args = eval(
            event_acc.Tensors("args/text_summary")[0].tensor_proto.string_val[0]
        )
        # To load on device without GPU
        args.device = "cpu"

        # Take first path since we need ANY model loaded, they're all the same in flops.
        checkpoints_path = f"log/tag/ppo/cp/{arch}/{seed}/"
        avail_checkpoint = sorted(
            os.listdir(checkpoints_path), key=lambda x: int(x.split("=")[1])
        )[0]
        avail_checkpoints_path = checkpoints_path + avail_checkpoint

        # Iterate through checkpoints, loading agents and setting one as random policy
        # then, collect episodes to compute CCM
        args.resume_path = avail_checkpoints_path

        # Load agent, set one as random
        policy, optim, agents, n_params, flops_per_loop = get_agents(ENV, args)

        ccms[arch]["n_flops"] = flops_per_loop

        with open(save_loc + f"{arch=}.pkl", "wb") as f:
            pickle.dump(ccms, f)


def get_n_params(archs, seeds, save_loc):
    """Post hoc gathering of number of params for a given arch

    Args:
        archs (list): architectures for which to retrieve number of flops
        seeds (list): list of valid seeds, only first one will be taken anyway
        save_loc (str): save/load location
    """
    for arch in tqdm(archs):
        # if exists(save_loc + f"{arch=}.pkl"):
        # print(f"Skipped {arch=}")
        # continue
        with open(save_loc + f"{arch=}.pkl", "rb") as f:
            ccms = pickle.load(f)

        seed = seeds[0]
        # Retrieve arguments used for the targeted run
        path = f"log/tag/ppo/{arch}/{seed}"
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        args = eval(
            event_acc.Tensors("args/text_summary")[0].tensor_proto.string_val[0]
        )
        # To load on device without GPU
        args.device = "cpu"

        # Take first path since we need ANY model loaded, they're all the same in flops.
        checkpoints_path = f"log/tag/ppo/cp/{arch}/{seed}/"
        avail_checkpoint = sorted(
            os.listdir(checkpoints_path), key=lambda x: int(x.split("=")[1])
        )[0]
        avail_checkpoints_path = checkpoints_path + avail_checkpoint

        # Iterate through checkpoints, loading agents and setting one as random policy
        # then, collect episodes to compute CCM
        args.resume_path = avail_checkpoints_path

        # Load agent, set one as random
        policy, optim, agents, n_params, flops_per_loop = get_agents(ENV, args)

        ccms[arch]["n_params"] = n_params

        with open(save_loc + f"{arch=}.pkl", "wb") as f:
            pickle.dump(ccms, f)


E = 5  # To vary


if __name__ == "__main__":
    seeds = list(range(0, 15))
    archs, seeds = get_archs_seeds("log/tag/ppo", seeds=seeds)
    save_loc = f"ccms_{E=}"
    # print(seeds, archs)
    # exit()
    save_ccms(archs, seeds, save_loc=save_loc)
    # get_n_flops(archs, seeds, save_loc)
    # get_n_params(archs, seeds, save_loc)
    # decide whether to do depth-wise of width-wise analysis
    # Counter-intuitive:
    # If depth is selected, we decide a fixed depth and vary the width
    # If width is selected, we decided a fixed with and vary the depth
    depth = 3
    # depth = False
    width = False
    # width = 64
    assert width != depth
    if depth != False:
        archs = [arch for arch in archs if len(arch.split("_")) == depth]
        fig_save_loc_tweak = "fixed_depth_var_width"
    elif width != False:
        archs = [arch for arch in archs if list(set(arch.split("_")))[0] == str(width)]
        fig_save_loc_tweak = "fixed_width_var_depth"

    plot_ccms(save_loc, archs, seeds, fig_save_loc_tweak)
    plot_ccm_flops(save_loc, archs, seeds, fig_save_loc_tweak)
    plot_regr_ccm_parameters(save_loc, archs, seeds, fig_save_loc_tweak)
    plot_regr_ccm_flops(save_loc, archs, seeds, fig_save_loc_tweak)

    rewards = get_rewards(archs, seeds)
    plot_rewards("reward", rewards, seeds, fig_save_loc_tweak)
    # retrieve_key_data_from_dict(ccms, "correl")
