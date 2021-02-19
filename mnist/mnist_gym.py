# @title MNIST class definition
# Define the gym environment.
import gym
from gym import spaces
import numpy as np
import cv2
import matplotlib.pyplot as plt


class USPSGym(gym.Env):
    def __init__(self, dataset, width=128, height=128, channels=1):
        # Training dataset (Handwritten digits on a 16x16px canvas).
        self.X, self.y = dataset

        # Reset the state index, used to step through dataset.
        self.idx = 0

        # Digits 0-9 are valid actions.
        self.action_space = spaces.Discrete(10)

        # A 1-channel canvas is used for observations.
        self.observation_space = spaces.Box(low=0, high=255, shape=(width, height, channels), dtype=np.uint8)

    def _obs(self):
        # Return a frame at the target dimensions from self.X at the current state index for the CnnPolicy.
        width, height, channels = (self.observation_space.shape[0],
                                   self.observation_space.shape[1],
                                   self.observation_space.shape[2])
        obs = self.X[self.idx]

        # Enlarge the observation if the dataset is smaller than the target canvas.
        if obs.shape[0] < width or obs.shape[1] < height:
            obs = cv2.resize(np.array(obs).astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
            obs = obs.reshape(width, height, channels)
        obs = obs.reshape(width, height, channels)
        return obs

    def step(self, action):
        # The agent earns 1 point for a correct label.
        reward = 1 if action == self.y[self.idx] else 0

        # The state index increments at each step then wraps around at the end of the training dataset.
        self.idx = self.idx + 1 if self.idx < len(self.X) - 1 else 0

        # Return the observation, earned reward, terminal state, and info dict.
        d = self._obs()
        return d, reward, self.idx == 0, {'obs': d}

    def reset(self):
        # Reset the index to the beginning of the training dataset and return the initial observation.
        self.idx = 0
        return self._obs()

    def render(self, action='', mode='human', close=False):
        # Display the labeled observation.
        width, height = self.observation_space.shape[0], self.observation_space.shape[1]
        fig, ax = plt.subplots(1)
        ax.imshow(self._obs().reshape(width, height), cmap='Greys')

        # Label with the correct value and action if supplied.
        title = '{}-{}'.format(action, self.y[self.idx]) if action != '' else self.y[self.idx]
        ax.set_title(title)
        plt.show()