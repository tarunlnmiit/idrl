import numpy as np
import random
from collections import namedtuple, deque
# Importing the model
from dqn import DQN

import torch
import torch.optim as optim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from skimage.transform import resize as imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.cat([e.state for e in experiences if e is not None])
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(DEVICE)
        next_states = torch.cat(
            [e.next_state for e in experiences if e is not None])
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences
                       if e is not None]).astype(np.uint8)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DQNAgent():
    """Interacts with and learns form the environment."""

    def __init__(self,
                 action_size,
                 seed,
                 lr=1e-3,
                 gamma=0.99,
                 tau=1e-3,
                 buffer_size=int(1e5),
                 batch_size=64,
                 update_every=100):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau

        # Q- Network
        self.qnetwork_local = DQN(action_size, seed).to(DEVICE)
        self.qnetwork_target = DQN(action_size, seed).to(DEVICE)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

    def act(self, state, eps=0):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # if state.shape == (1, 80, 80):
        #     # print('here')
        #     state = torch.from_numpy(state).float().to(DEVICE)
        #     # print(state.size())
        #     # print(state[0, 0, 0], type(state[0, 0, 0]))
        # else:
        #     state = torch.from_numpy(state).to(DEVICE)
        #     # print(state.size())
        #     # print(state[0, 0, 0], type(state[0, 0, 0]))
        # print(type(state), state.size())

        state = state.float()

        # freeze the q network to make predictions
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # unfreeze the q network to continue the training
        self.qnetwork_local.train()

        # Epsilon -greedy action selction
        if random.random() > eps:
            return action_values.cpu()
            # return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state_, action, reward, next_state_, done):
        state = state_.to(DEVICE)
        next_state = next_state_.to(DEVICE)

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experience = self.memory.sample()
                self.learn(experience)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            self.gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        self.qnetwork_local.train()
        self.qnetwork_target.eval()

        # shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.qnetwork_target(next_states).detach().max(
                1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (self.gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(DEVICE)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.hard_update()

        # def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        # for target_param, local_param in zip(target_model.parameters(),
        #                                     local_model.parameters()):
        #    target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def hard_update(self):
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())