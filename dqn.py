import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network: Actor (Policy) Model.
    (function approximator for the Q-table)
    """

    def __init__(self, state_size, action_size, seed, fc1_unit=64,
                 fc2_unit=64):
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
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(
            1 * 128 * 6 * 6,
            512)  # flattening 3 frames of 128 kernels, of imgs of size 19x8
        self.fc2 = nn.Linear(512, action_size)  # 6 actions in space invaders

        # dropout
        # self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        # self.drop1 = nn.Dropout(p=0.2)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.drop2 = nn.Dropout(p=0.3)
        # self.conv3 = nn.Conv2d(64, 128, 3)
        # self.drop3 = nn.Dropout(p=0.4)
        # self.fc1 = nn.Linear(
        #     1 * 128 * 6 * 6,
        #     512)  # flattening 3 frames of 128 kernels, of imgs of size 19x8
        # self.fc2 = nn.Linear(512, action_size)  # 6 actions in space invaders

        # maxpooling
        # self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        # self.pool = nn.MaxPool2d(2,1)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.pool1 = nn.MaxPool2d(4,2)
        # self.conv3 = nn.Conv2d(64, 128, 3)
        # self.fc1 = nn.Linear(
        #     1 * 128,
        #     512)  # flattening 3 frames of 128 kernels, of imgs of size 19x8
        # self.fc2 = nn.Linear(512, action_size)  # 6 actions in space invaders

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
        x = state.clone()
        x = x.view(-1, 1, 80, 80)

        # forward pass through conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # flatten the tensor for the fc layers
        x = x.view(-1, 128 * 6 * 6)

        # # forward pass through conv layers
        # x = F.relu(self.conv1(x))
        # x = self.drop1(x)
        # x = F.relu(self.conv2(x))
        # x = self.drop2(x)
        # x = F.relu(self.conv3(x))
        # x = self.drop3(x)

        # # flatten the tensor for the fc layers
        # x = x.view(-1, 128*6*6)

        # forward pass through conv layers
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # x = F.relu(self.conv2(x))
        # x = self.pool1(x)
        # x = F.relu(self.conv3(x))

        # # flatten the tensor for the fc layers
        # x = x.view(-1, 128)

        # forward pass through fc layers
        x = F.relu(self.fc1(x))

        return self.fc2(x)
