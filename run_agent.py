import datetime
from argparse import ArgumentParser
import pickle
from utils.recorder.record import Recorder
import aicrowd_gym
import minerl
import time
from openai_vpt.agent import MineRLAgent


def log_parameters(recording_name, weight_name, model_name, n_episodes, max_steps, done, env, total_time, timestamp):

    with open("video/record_parameters.txt", "a+") as file:
        file.write(f"┌───────────────────────────────────────────────┐\n")
        file.write(f'│{recording_name : <46} │\n')
        file.write(f"│                                               │\n")
        file.write(f'│    Weights Name = {weight_name: <27} │\n')
        file.write(f'│    Model Name   = {model_name: <27} │\n')
        file.write(f'│    Max Steps    = {max_steps: <27} │\n')
        file.write(f'│    Done         = {done: <27} │\n')
        file.write(f'│    Env. Name    = {env: <27} │\n')
        file.write(f'│    Runtime      = {total_time: <27} │\n')
        file.write(f'│    Timestamp    = {timestamp: <27} │\n')
        file.write(f"│                                               │\n")
        file.write("└───────────────────────────────────────────────┘\n\n")


def main(model, weights, env, max_steps=int(1e9), show=False, record=True):
    # Using aicrowd_gym is important! Your submission will not work otherwise
    model_name = model.split("/")[-1]
    weights_name = weights.split("/")[-1]
    env_name = env
    env = aicrowd_gym.make(env)
    start_time = int(time.time())
    record_name = weights_name.split(".")[0] + f"_{start_time}"
    if record:
        env = Recorder(env, './video', fps=60, name=record_name)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    for _ in range(max_steps):
        action = agent.get_action(obs)
        # ESC is not part of the predictions model.
        # For baselines, we just set it to zero.
        # We leave proper execution as an exercise for the participants :)
        action["ESC"] = 0
        obs, _, done, _ = env.step(action)
        if show:
            env.render()
        if done:
            break

    env.close()
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    if record:
        log_parameters(record_name, weights_name, model_name,  max_steps, done, env_name, int(time.time()-start_time), timestamp)
    print("Done")


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, show=args.show)
