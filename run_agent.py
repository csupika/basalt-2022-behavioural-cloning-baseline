import datetime
from argparse import ArgumentParser
import pickle

import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt
from ultralytics import YOLO

from utils.recorder.record import Recorder
import aicrowd_gym
import minerl
import time
from openai_vpt.agent import MineRLAgent

SEED = 42
CONF = 0.95
inside_cave_probabilities = []
counter = 1


def save_inside_cave_probabilities(record_name):
    global counter, inside_cave_probabilities

    # Create a time series array for the x-axis (steps) from 1 to the length of inside_cave_probabilities
    x = np.arange(1, len(inside_cave_probabilities) + 1)

    # Plot the inside_cave_probabilities against the steps
    plt.plot(x, inside_cave_probabilities, label=f'{record_name}_{counter:02}')

    # Add labels and title to the plot
    plt.xlabel('Steps')
    plt.ylabel('Inside Cave Probability')
    plt.title('Inside Cave Probability Over Time')

    # Add a legend to the plot
    plt.legend()

    # Get the value of the last probability
    last_probability = inside_cave_probabilities[-1]

    # Add text label to the last point on the graph
    plt.text(len(inside_cave_probabilities), last_probability, f'{last_probability:.2f}', ha='left', va='center')

    # Save the plot as a PNG image
    plt.savefig(f'video/{record_name}_{counter:02}.png')
    counter += 1

    # Clear the current figure to avoid overlapping plots in subsequent episodes
    plt.clf()

    inside_cave_probabilities = []


def log_parameters(record_name, weights_name, vpt_model_name, yolo_model, max_steps, no_steps, n_episodes, completed_runs, env_name, start_time, device):
    time_now = datetime.datetime.now()
    timestamp = time_now.strftime("%Y-%m-%d %H:%M:%S")
    formatted_start_time = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    total_time = int(time.time()-start_time)

    # Convert the completed_runs list to a formatted string
    completed_runs_str = " ".join([f"[{episode}, {steps}, {done}]" for episode, steps, done in completed_runs])
    with open("video/record_parameters.txt", "a+") as file:
        file.write(f"┌───────────────────────────────────────────────┐\n")
        file.write(f'│{record_name : <46} │\n')
        file.write(f"│                                               │\n")
        file.write(f'│    Weights Name = {weights_name: <27} │\n')
        file.write(f'│    VPT Model    = {vpt_model_name: <27} │\n')
        file.write(f'│    yolo Model   = {yolo_model: <27} │\n')
        file.write(f'│    Max Steps    = {max_steps: <27} │\n')
        file.write(f'│    No. Steps    = {no_steps: <27} │\n')
        file.write(f'│    No. Episodes = {n_episodes: <27} │\n')
        file.write(f'│    Confidence   = {CONF: <27} │\n')
        file.write(f'│    Seed         = {SEED: <27} │\n')
        file.write(f'│    Completed Runs ep. : steps = {completed_runs_str: <10} │\n')
        file.write(f'│    Device       = {device: <27} │\n')
        file.write(f'│    Env. Name    = {env_name: <27} │\n')
        file.write(f'│    Runtime      = {total_time: <27} │\n')
        file.write(f'│    Start time   = {formatted_start_time: <27} │\n')
        file.write(f'│    End time     = {timestamp: <27} │\n')
        file.write(f"│                                               │\n")
        file.write("└───────────────────────────────────────────────┘\n\n")


def predict_on_observation(yolo_model, img):
    # Run the model prediction on the temporary file
    results = yolo_model(img, verbose=False)

    # Get probabilities [0 - Unlabeled, 1 - inside cave]
    probs = results[0].probs.data.tolist()
    inside_cave_probabilities.append(probs[1])

    return probs[1] >= CONF


def main(vpt_model, yolo_model, weights, env, n_episodes=3, max_steps=int(1e9), show=False, record=True, device="cuda"):
    global counter
    counter = 1
    # Using aicrowd_gym is important! Your submission will not work otherwise
    vpt_model_name = vpt_model.split("/")[-1]
    yolo_model_name = yolo_model.split("/")[-3] + "/" + yolo_model.split("/")[-1]

    weights_name = weights.split("/")[-1]
    env_name = env
    env = aicrowd_gym.make(env)
    start_time = int(time.time())
    record_name = f"{start_time}_{weights_name.split('.')[0]}"
    if record:
        env = Recorder(env, './video', fps=60, name=record_name)
    agent_parameters = pickle.load(open(vpt_model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, device=device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    # load a custom YOLO model
    yolo_model = YOLO(yolo_model)

    env.reset()
    no_steps = 0
    completed_runs = []
    try:
        for i in range(n_episodes):
            env.seed(SEED)
            obs = env.reset()
            z = 0
            done = None
            for z in range(max_steps):
                no_steps += 1
                action = agent.get_action(obs)
                # ESC is not part of the predictions model.
                # For baselines, we just set it to zero.
                # We leave proper execution as an exercise for the participants :)
                action["ESC"] = 0
                obs, _, done, _ = env.step(action)
                if show:
                    env.render()
                if done or predict_on_observation(yolo_model, env.render(mode='rgb_array')):
                    break
            completed_runs.append([i, z, done])
            save_inside_cave_probabilities(record_name)

    finally:
        env.close()
        if record:
            log_parameters(record_name, weights_name, vpt_model_name, yolo_model_name, max_steps, no_steps, n_episodes, completed_runs, env_name, start_time, device)
        print("Done")


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--vpt-model", type=str, required=True, help="Path to the VPT '.model' file to be loaded.")
    parser.add_argument("--yolo-model", type=str, required=True, help="Path to the YOLOv8 '.pt' file to be loaded.")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.vpt_model, args.yolo_model, args.weights, args.env, show=args.show)
