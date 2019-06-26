import numpy as np
from skimage.transform import resize


class Utils(object):
    def __init__(self, frame):
        self.frame = frame

    def _grey_scale(self):
        return np.mean(self.frame, axis=2)

    def _resize(self, frame, frame_height=84, frame_width=84):
        return resize(frame, (frame_height, frame_width), preserve_range=True).astype(np.float32)

    def _preprocess(self):
        return self._resize(self._grey_scale()/142)

    # TODO  stacked frames fun may be modified
    def _stack_frames(self):
        i = 0
        stacked_frames = []
        while i != 4:
            processed_frame = self._preprocess()
            stacked_frames.append(processed_frame)
            i += 1
        return np.moveaxis(np.array(stacked_frames), 1, -1)


