# MineRL Agent
## Acknowledgments
- Forked from [minerllabs/basalt-2022-behavioural-cloning-baseline](https://github.com/csupika/basalt-2022-behavioural-cloning-baseline)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [YOLOv8 prediction code](https://docs.ultralytics.com/tasks/classify/#predict) `yolov8/predict.py`
- [Minecraft run recorder](https://github.com/ryanrudes/colabgymrender/blob/main/colabgymrender/recorder.py) `utils/recorder/record.py`


Difference between the original MineRL Labs and my code can be found [here](https://github.com/minerllabs/basalt-2022-behavioural-cloning-baseline/compare/main...csupika:basalt-2022-behavioural-cloning-baseline:main)

Most of the changes was applied in:
- [data_loader.py](https://github.com/minerllabs/basalt-2022-behavioural-cloning-baseline/compare/main...csupika:basalt-2022-behavioural-cloning-baseline:main#diff-e42ca25bb5510426ff2e770dc5dec52a2e8bce0c9a6ac8a51ec50277e98b7ddf)
- [behavoral_cloning.py](https://github.com/minerllabs/basalt-2022-behavioural-cloning-baseline/compare/main...csupika:basalt-2022-behavioural-cloning-baseline:main#diff-45a5f8c20489ae9d62a9df8d836d0f96979c582de6e40488690cbaaa0a0d81bd)

## Run MineRL Agent
### I. Train VPT Model
1) Download dataset with `utils/download_dataset.py`
2) Setup parameters in `behavioural_cloning.py` & `train.py`
3) Run `train.py`

### II. Train YOLO model
1) Get dataset by running `utils/sample_videos.py` or download from Roboflow
    - [MineRL FindCave Inside Cave v1](https://universe.roboflow.com/minerl-findcave-u5lrz/minerl-findcave-inside-cave-v1)
    - [MineRL FindCave Inside Cave v2: No Water Inside Cave](https://universe.roboflow.com/minerl-findcave-u5lrz/minerl-findcave-inside-cave-v2-no-water-inside-cave)
2) Select pre-trained model based on your preference. You can find the specifications of the models [on their website.](https://docs.ultralytics.com/tasks/classify/#models)
3) Train pre-trained YOLO model on dataset by running `yolov8/train_yolov8.py`
- To evaluate the performance of your YOLO model, run `yolov8/eval_yolov8.py`

### III. Run MineRL Agent
1) Setup parameters in `test_FindCave.py`
   - Optional: Setup Minecraft word seed and YOLO confidence level from `run_agent.py`
1) Select the VPT model and weight to run the agent
    - The VPT models were in this repo were all trained on `foundation-model-1x.model` and `foundation-model-1x.weights`.
    - If you want to run your fine-tuned VPT model then use `train/<NAME OF YOUR MODEL>.weights`
    - If you want to run the baseline model then run `foundation-model-1x.weights`
    - Please note, it's essential to use the appropriate model and weights. If you used the 1x width model, then you must select the matching 1x width weight for either running or training. When using your freshly trained VPT model weights, opt for the 1x width model if that's the version you used for training.
    - More information on the VPT models can be found on [OpenAI's repo](https://github.com/openai/Video-Pre-Training)
2) Select the YOLO model to run the agent from `yolov8/runs/classify/train{yolo_training_no}/`
3) Run the agent
- Headless: `xvfb-run run.py` from the console
- Headed: Set the show to `True` on line 96 in `run_agent.py` and then run.
- The recordings of the run(s) will be saved to `video/`

# Get Fine-Tuned VPT Models From Git LFS 
The VPT models trained by me are on Git LFS
To download please follow these steps:
1) `git lfs install`
2) `git lfs fetch`
3) Checkout or Pull: Depending on how you cloned the repository, you might need to perform a checkout or pull operation to get the VPT files trained by me.
   - `git checkout main`
   - `git pull origin main`
