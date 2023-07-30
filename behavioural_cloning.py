# Basic behavioural cloning
# Note: this uses gradient accumulation in batches of ones
#       to perform training.
#       This will fit inside even smaller GPUs (tested on 8GB one),
#       but is slow.
import datetime
from argparse import ArgumentParser
import pickle
import time

import os
import logging
import datetime

import gym
import minerl
import torch as th
import numpy as np

from openai_vpt.agent import PI_HEAD_KWARGS, MineRLAgent
from data_loader import DataLoader
from openai_vpt.lib.tree_util import tree_map
from random_word import RandomWords

EPOCHS = 1
# Needs to be <= number of videos
BATCH_SIZE = 28
# Ideally more than batch size to create
# variation in datasets (otherwise, you will
# get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 32
DEVICE = "cuda:3"
#DEVICE_2 = "cuda:2"

# Has to be a decimal [0; 1]
MIN_REQUIRED_ACTIVE_QUEUES_PERCENTAGE = 0.1

LOSS_REPORT_RATE = 100

# Tuned with bit of trial and error
LEARNING_RATE = 0.000181
# OpenAI VPT BC weight decay
# WEIGHT_DECAY = 0.039428
WEIGHT_DECAY = 0.0
# KL loss to the original model was not used in OpenAI VPT
KL_LOSS_WEIGHT = 1.0
MAX_GRAD_NORM = 5.0

# 10 000 was enough for 69 videos,  1699.66 with 8 workers
MAX_BATCHES = int(1e11)

HEAD_START_TIME_FOR_PROCESSES = 10

variables = [
    ("EPOCHS", EPOCHS),
    ("N_WORKERS", N_WORKERS),
    ("MAX_BATCHES", MAX_BATCHES),
    ("BATCH_SIZE", BATCH_SIZE),
    ("MIN_ACTIVE_QUEUES %", MIN_REQUIRED_ACTIVE_QUEUES_PERCENTAGE),
    ("LEARNING_RATE", LEARNING_RATE),
    ("WEIGHT_DECAY", WEIGHT_DECAY),
    ("KL_LOSS_WEIGHT", KL_LOSS_WEIGHT),
    ("MAX_GRAD_NORM", MAX_GRAD_NORM),
    ("DEVICES", DEVICE),
]


def log_parameters(training_name, time_spent, data_dir, sample_size, remaining_rec, start_timestamp, batches_done,
                   exception=None):
    with open("train/_model_parameters.txt", "a+") as file:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"┌───────────────────────────────────────────────┐\n")
        file.write(f'│{training_name : <46} │\n')
        file.write(f"│                                               │\n")
        file.write(f'│    TRAINING TIME = {time_spent: <26} │\n')
        # Write each variable name and its value to the file
        for variable in variables:
            name, value = variable
            to_print = f"{name} = {value}"
            file.write(f'│    {to_print : <42} │\n')
        file.write(f"│                                               │\n")
        file.write(f'│    TIMESTAMP = {timestamp: <30} │\n')
        file.write(f'│    BATCHES  = {batches_done: <31} │\n')
        file.write(f'│    START TIME = {start_timestamp: <29} │\n')
        file.write(f'│    DATA DIR = {data_dir: <31} │\n')
        file.write(f'│    SAMPLE SIZE = {sample_size: <29}│\n')
        file.write(f'│    REMAINING REC = {remaining_rec: <27}│\n')
        if exception:
            file.write(f"│ EXCEPTION:                                    │\n")
            file.write(f"| {type(exception).__name__} was raised. {exception} \n")
        file.write("└───────────────────────────────────────────────┘\n\n")


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def behavioural_cloning_train(data_dir, in_model, in_weights):
    time_now = datetime.datetime.now()
    start_timestamp = time_now.strftime('%Y%m%d_%H%M%S')
    r = RandomWords()
    training_name = start_timestamp + "_" + r.get_random_word() + "_" + r.get_random_word()
    out_weights = "train/" + training_name + ".weights"
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    # Logging
    logs_folder = "logs/train"  # Specify the folder for logs
    os.makedirs(logs_folder, exist_ok=True)  # Create the logs folder if it doesn't exist
    setup_logging(logs_folder, training_name)

    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    env = gym.make("MineRLBasaltFindCave-v0")
    agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)

    # Create a copy which will have the original parameters
    original_agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs,
                                 pi_head_kwargs=agent_pi_head_kwargs)
    original_agent.load_weights(in_weights)
    env.close()

    policy = agent.policy
    original_policy = original_agent.policy

    # Freeze most params if using small dataset
    for param in policy.parameters():
        param.requires_grad = False
    # Unfreeze final layers
    trainable_parameters = []
    for param in policy.net.lastlayer.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)
    for param in policy.pi_head.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)

    # Parameters taken from the OpenAI VPT paper
    optimizer = th.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,
        min_required_queues=int(N_WORKERS * MIN_REQUIRED_ACTIVE_QUEUES_PERCENTAGE)
    )

    # Print logging details
    logging.info(f"Training name: {training_name}")
    logging.info(f"Start time: {time_now.strftime('%Y-%m-%d %H:%M:%S')}")

    time.sleep(HEAD_START_TIME_FOR_PROCESSES)
    start_time = time.time()

    # Keep track of the hidden state per episode/trajectory.
    # DataLoader provides unique id for each episode, which will
    # be different even for the same trajectory when it is loaded
    # up again
    episode_hidden_states = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)

    loss_sum = 0
    exception = None
    batches_done = 0
    try:
        for batch_i, (batch_images, batch_actions, batch_episode_id) in enumerate(data_loader):
            batch_loss = 0
            batches_done = batch_i
            for image, action, episode_id in zip(batch_images, batch_actions, batch_episode_id):
                if image is None and action is None:
                    # A work-item was done. Remove hidden state
                    if episode_id in episode_hidden_states:
                        removed_hidden_state = episode_hidden_states.pop(episode_id)
                        del removed_hidden_state
                    continue

                agent_action = agent._env_action_to_agent(action, to_torch=True, check_if_null=True)
                if agent_action is None:
                    # Action was null
                    continue

                agent_obs = agent._env_obs_to_agent({"pov": image})
                if episode_id not in episode_hidden_states:
                    episode_hidden_states[episode_id] = policy.initial_state(1)
                agent_state = episode_hidden_states[episode_id]

                pi_distribution, _, new_agent_state = policy.get_output_for_observation(
                    agent_obs,
                    agent_state,
                    dummy_first
                )

                with th.no_grad():
                    original_pi_distribution, _, _ = original_policy.get_output_for_observation(
                        agent_obs,
                        agent_state,
                        dummy_first
                    )

                log_prob = policy.get_logprob_of_action(pi_distribution, agent_action)
                kl_div = policy.get_kl_of_action_dists(pi_distribution, original_pi_distribution)

                # Make sure we do not try to backprop through sequence
                # (fails with current accumulation)
                new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
                episode_hidden_states[episode_id] = new_agent_state

                # Finally, update the agent to increase the probability of the
                # taken action.
                # Remember to take mean over batch losses
                loss = (-log_prob + KL_LOSS_WEIGHT * kl_div) / BATCH_SIZE
                batch_loss += loss.item()
                loss.backward()

            th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()

            loss_sum += batch_loss
            if batch_i % LOSS_REPORT_RATE == 0:
                time_since_start = time.time() - start_time
                logging.info(
                    f"Time: {time_since_start:.2f}, Batches: {batch_i}, Avrg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
                loss_sum = 0

            if batch_i > MAX_BATCHES:
                logging.info("[MAX] Max batches reached")
                break

    except Exception as e:
        logging.exception(f"EXCEPTION WAS RAISED {type(e).__name__}")
        exception = e

    finally:
        data_loader.__del__()
        state_dict = policy.state_dict()
        remaining_rec = data_loader.task_queue.qsize()
        th.save(state_dict, out_weights)
        log_parameters(training_name, time.time() - start_time, data_dir, len(data_loader.demonstration_tuples),
                       remaining_rec, start_timestamp, batches_done, exception)
        logging.info(">>>>DONE<<<<")


def setup_logging(logs_folder, training_name):
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        handlers=[logging.FileHandler(f"{logs_folder}/log_{training_name}.txt"),
                  logging.StreamHandler()],
        level=logging.INFO,
        format=log_format)
    logging.info(f"Behavioural Cloning: {variables}")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the directory containing recordings to be trained on")
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be finetuned")

    args = parser.parse_args()
    behavioural_cloning_train(args.data_dir, args.in_model, args.in_weights)
