from run_agent import main as run_agent_main
from config import EVAL_EPISODES, EVAL_MAX_STEPS

def main():
    run_agent_main(
        model="data/VPT-models/foundation-model-1x.model",
        # weights="train/20230729_024840_statfarad_saruk.weights",
        # weights="train/20230728_050222_irreplaceability_saturnineness.weights",
        weights="data/VPT-models/foundation-model-1x.weights",
        env="MineRLBasaltFindCave-v0",
        n_episodes=15,
        max_steps=4000,
        device="cuda:0"
    )
