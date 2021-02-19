import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim

class DQN(nn.Module):
    """
    Deep Q-Network: Actor (Policy) Model.
    (function approximator for the Q-table)
    """

    def __init__(self, action_size, seed):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of neurons in first hidden layer
            fc2_unit (int): Number of neurons in second hidden layer
        """
        super(DQN, self).__init__()  # calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(
            1 * 10 * 4 * 4,
            256)  # flattening 3 frames of 128 kernels, of imgs of size 19x8
        self.fc2 = nn.Linear(256, action_size)  # 6 actions in space invaders

    def forward(self, state):
        """
        mapping a state to action-values.
        ---
        args:
            state: state tensor (grayscale img)
        returns:
            q_values: array of length 6. It corresponds to the action-values for each action given the input state
                q_values=[Q(state, a_1), Q(state, a_2), ..., Q(state, a_6)]
        """
        # gym gives frames as height, width, channel
        # whereas the network expects the channels to come first
        # print('afafasf', state, state.shape)
        x = state.clone()
        x = x.view(-1, 1, 28, 28)

        # forward pass through conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # flatten the tensor for the fc layers
        x = x.view(-1, 10 * 4 * 4)

        # forward pass through fc layers
        x = F.relu(self.fc1(x))

        return self.fc2(x)
