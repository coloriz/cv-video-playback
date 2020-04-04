from typing import Optional, Tuple, NoReturn

import cv2 as cv
import numpy as np


class VideoReader:
    def __init__(self, filename: str, reverse=False, q_size=16, worker_type='thread'):
        self._frame_width, self._frame_height, self._fps = self._get_metadata_from_file(filename)
        self._filename = filename
        self._reverse = reverse
        self._eof = False

        if worker_type == 'thread':
            from threading import Thread, Event
            from queue import Queue
            self._terminated = Event()
            self._q = Queue(maxsize=q_size)
            self._worker = Thread(target=self._thread_func)
        elif worker_type == 'process':
            from multiprocessing import Process, Queue, Event
            self._terminated = Event()
            self._q = Queue(maxsize=q_size)
            self._worker = Process(target=self._thread_func)
        else:
            raise ValueError(f'No such worker type: {worker_type} (available: [thread, process])')
        self._worker.daemon = True
        self._worker.start()

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._frame_width, self._frame_height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def qsize(self) -> int:
        return self._q.qsize()

    @staticmethod
    def _get_metadata_from_file(filename: str) -> Tuple[int, int, float]:
        cap = cv.VideoCapture(filename)
        assert cap.isOpened()
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return frame_width, frame_height, fps

    def _thread_func(self) -> NoReturn:
        cap = cv.VideoCapture(self._filename)
        assert cap.isOpened()
        print(f'Using {cap.getBackendName()} for VideoCapture backend.')

        frame_orders = range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
        if self._reverse:
            frame_orders = reversed(frame_orders)

        for pos in frame_orders:
            if self._terminated.is_set():
                break
            cap.set(cv.CAP_PROP_POS_FRAMES, pos)
            _, frame = cap.read()
            self._q.put(frame)

        # EOF sign
        self._q.put(None)
        cap.release()

    def read(self) -> Optional[np.ndarray]:
        # to prevent hanging from re-enterance after EOF
        if self._eof:
            return None

        frame = self._q.get()
        if frame is None:
            self._eof = True

        return frame

    def close(self) -> NoReturn:
        self._terminated.set()
        while True:
            if self.read() is None:
                break
        self._worker.join()
