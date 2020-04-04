import sys
from typing import NoReturn

import cv2 as cv
import numpy as np

from videoplayer import VideoPlayer
from videowriter import VideoWriter


class MyVideoPlayer(VideoPlayer):
    def __init__(self, filename: str, window_title: str, worker_type='thread'):
        super().__init__(filename, window_title, reverse=True, worker_type=worker_type)
        self._video_writer = VideoWriter('rendered.ts', cv.VideoWriter_fourcc('m', 'p', '2', 'v'),
                                         self._video_reader.fps, self._video_reader.frame_size, worker_type)

    def _pre_render_hook(self, frame: np.ndarray) -> NoReturn:
        frame[:, :frame.shape[1] // 2] = np.clip(1.5 * frame[:, :frame.shape[1] // 2], 0, 255)
        self._video_writer.write(frame)

    def _dispose(self) -> NoReturn:
        self._video_writer.close()
        super()._dispose()


if __name__ == '__main__':
    player = MyVideoPlayer(sys.argv[1], 'playback', worker_type='process')
    player.play()
