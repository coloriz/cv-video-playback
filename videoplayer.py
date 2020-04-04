import asyncio
import time
from typing import NoReturn

import cv2 as cv
import numpy as np

from videoreader import VideoReader


class VideoPlayer:
    def __init__(self, filename: str, window_title: str, reverse=False, qsize=16, worker_type='thread'):
        video_reader = VideoReader(filename, reverse, qsize, worker_type)
        print(f'{filename} / {video_reader.frame_size} / {video_reader.fps} fps')

        cv.namedWindow(window_title)

        self._window_title = window_title
        self._video_reader = video_reader
        self._rendered_frames = 0
        self._interval = 1 / video_reader.fps

        self._next_execution = 0
        self._loop = asyncio.get_event_loop()

        # for performance measuring
        self._start_time = self._report_time = 0

    def _pre_render_hook(self, frame: np.ndarray) -> NoReturn:
        pass

    def _render_frame(self) -> NoReturn:
        # start fps measuring on first frame
        if self._rendered_frames == 0:
            self._start_time = self._report_time = time.perf_counter_ns()

        self._next_execution += self._interval
        self._loop.call_at(self._next_execution, self._render_frame)

        frame = self._video_reader.read()
        if frame is None:
            self._dispose()
            return

        self._pre_render_hook(frame)
        cv.imshow(self._window_title, frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            self._dispose()
            return

        self._rendered_frames += 1

        # performance measurment
        current_time = time.perf_counter_ns()
        if current_time > (self._report_time + 2 * 1_000_000_000):
            elasped_time = current_time - self._start_time
            secs = elasped_time / 1_000_000_000
            try:
                qsize = self._video_reader.qsize
            except NotImplementedError:
                qsize = 'Unknown'
            print(f'Rendered {self._rendered_frames} frames in {secs:.3f} sec / '
                  f'{self._rendered_frames / secs:.2f} fps / '
                  f'qsize: {qsize}')
            self._report_time = current_time

    def play(self) -> NoReturn:
        self._next_execution = self._loop.time() + 1  # 1s delay for warming up
        self._loop.call_at(self._next_execution, self._render_frame)
        self._loop.run_forever()
        self._loop.close()

    def _dispose(self) -> NoReturn:
        self._video_reader.close()
        self._loop.stop()
