from run_agent import main as run_agent_main
from config import EVAL_EPISODES, EVAL_MAX_STEPS

def main():
    run_agent_main(
        model="data/VPT-models/foundation-model-1x.model",
        weights="train/reduced_copers.weights",
        env="MineRLBasaltFindCave-v0",
        max_steps=2500
    )

if __name__ == '__main__':
    main()
