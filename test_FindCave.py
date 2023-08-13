from run_agent import main as run_agent_main

def main():
    yolo_training_no = 21
    run_agent_main(
        vpt_model="data/VPT-models/foundation-model-1x.model",
        yolo_model=f"yolov8/runs/classify/train{yolo_training_no}/weights/best.pt",
        weights="train/20230729_024840_statfarad_saruk.weights",
        env="MineRLBasaltFindCave-v0",
        n_episodes=5,
        max_steps=3600,
        device="cuda:3"
    )
