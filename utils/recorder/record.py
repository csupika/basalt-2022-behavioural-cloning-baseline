# Source: https://github.com/ryanrudes/colabgymrender/blob/main/colabgymrender/recorder.py

from IPython.core.display import Video, display
from moviepy.editor import *
import time
import cv2
import gym
import os

class Recorder(gym.Wrapper):
    def __init__(self, env, directory, auto_release=True, size=None, fps=None, name="Default"):
        super(Recorder, self).__init__(env)
        self.directory = directory
        self.auto_release = auto_release
        self.active = True
        self.name = name
        self.video_counter = 0

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        if size is None:
            self.env.reset()
            self.size = self.env.render(mode = 'rgb_array').shape[:2][::-1]
        else:
            self.size = size

        if fps is None:
            if 'video.frames_per_second' in self.env.metadata:
                self.fps = self.env.metadata['video.frames_per_second']
            else:
                self.fps = 30
        else:
            self.fps = fps

    def pause(self):
        self.active = False

    def resume(self):
        self.active = True

    def _start(self):
        self.path = f'{self.directory}/{self.name}_{self.video_counter:02}.mp4'
        self.video_counter += 1
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)

    def _write(self):
        if self.active:
            frame = self.env.render(mode = 'rgb_array')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._writer.write(frame)

    def release(self):
        self._writer.release()

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        self._start()
        self._write()
        return observation

    def step(self, *args, **kwargs):
        data = self.env.step(*args, **kwargs)
        self._write()

        if self.auto_release and data[2]:
            self.release()

        return data

    def play(self):
        start = time.time()
        filename = 'temp-{start}.mp4'
        clip = VideoFileClip(self.path)
        clip.write_videofile(filename, progress_bar = False, verbose = False)
        display(Video(filename, embed = True))
        os.remove(filename)
