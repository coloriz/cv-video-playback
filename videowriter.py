from typing import NoReturn, Tuple

import cv2 as cv
import numpy as np


class VideoWriter:
    def __init__(self, filename: str, fourcc: int, fps: float, frame_size: Tuple[int, int], worker_type='thread'):
        self._filename = filename
        self._fourcc = fourcc
        self._fps = fps
        self._frame_size = frame_size

        if worker_type == 'thread':
            from threading import Thread
            from queue import Queue
            self._q = Queue()
            self._worker = Thread(target=self._thread_func)
        elif worker_type == 'process':
            from multiprocessing import Process, Queue
            self._q = Queue()
            self._worker = Process(target=self._thread_func)
        else:
            raise ValueError(f'No such worker type: {worker_type} (available: [thread, process])')
        self._worker.daemon = True
        self._worker.start()

    def _thread_func(self) -> NoReturn:
        writer = cv.VideoWriter(self._filename, self._fourcc, self._fps, self._frame_size)
        assert writer.isOpened()
        print(f'Using {writer.getBackendName()} VideoWriter backend.')

        while True:
            frame = self._q.get()
            if frame is None:
                break
            writer.write(frame)

        writer.release()

    def write(self, frame: np.ndarray) -> NoReturn:
        self._q.put(frame)

    def close(self) -> NoReturn:
        self._q.put(None)
        self._worker.join()
