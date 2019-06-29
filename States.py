import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque


class States(object):

    @staticmethod
    def stacking_frames(frame, is_new_eps, stack_size=4):

        # Greyscale frame
        grey = rgb2gray(frame)

        # Crop the screen (remove the part where the score is)
        cropped_frame = grey[28:-12,]

        # Resize
        preprocessed_frame = resize(cropped_frame, [84, 84])

        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        if is_new_eps:
            stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

            for i in range(stack_size):
                stacked_frames.append(preprocessed_frame)
            stacked_states = np.stack(stacked_frames, axis=2)
        else:
            stacked_frames.append(preprocessed_frame)
            stacked_states = np.stack(stacked_frames, axis=2)

        return stacked_states, stacked_frames

