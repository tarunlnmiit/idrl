import pickle
from collections import deque

import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from agent import DQNAgent
from mnist_gym import MnistGym
import matplotlib.pyplot as plt


def train(n_episodes=100,
          max_t=10000,
          eps_start=1.0,
          eps_end=0.01,
          eps_decay=0.996, datalen=60000):
    scores = []

    # list containing the timestep per episode at which the game is over
    done_timesteps = []

    # scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start
    for i_episode in range(1, n_episodes + 1): #epochs
        state = env.reset()
        score = 0
        for timestep in range(max_t):
            logits = agent.act(state, eps)
            action = np.argmax(logits)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            # above step decides whether we will train(learn) the network
            # actor (local_qnetwork) or we will fill the replay buffer
            # if len replay buffer is equal to the batch size then we will
            # train the network or otherwise we will add experience tuple in our
            # replay buffer.
            state = next_state
            score += reward
            if done:
                print('\tEpisode {} done in {} timesteps.'.format(
                    i_episode, timestep))
                done_timesteps.append(timestep)
                break
            # scores_window.append(score)  # save the most recent score
            scores.append(score)  # save the most recent score
            eps = max(eps * eps_decay, eps_end)  # decrease the epsilon

            if timestep % SAVE_EVERY == 0:
                print('\rEpisode {}\tTimestep {}\tAccuracy {:.2f}'.format(
                    i_episode, timestep, (score / datalen) * 100), end="")

                # save the final network
                torch.save(agent.qnetwork_local.state_dict(),
                           SAVE_DIR + '22_mnist_model_10000.pth')

                # save the final scores
                with open(SAVE_DIR + 'scores', 'wb') as fp:
                    pickle.dump(scores, fp)

                # save the done timesteps
                with open(SAVE_DIR + 'dones', 'wb') as fp:
                    pickle.dump(done_timesteps, fp)

        print('\rEpisode {}\t Accuracy {:.2f}'.format(
            i_episode, (score / datalen) * 100), end="\n")

    # save the final network
    torch.save(agent.qnetwork_local.state_dict(), SAVE_DIR + '22_mnist_model_10000.pth')

    # save the final scores
    with open(SAVE_DIR + 'scores', 'wb') as fp:
        pickle.dump(scores, fp)

    # save the done timesteps
    with open(SAVE_DIR + 'dones', 'wb') as fp:
        pickle.dump(done_timesteps, fp)

    return scores


def test(trained_agent):
    total_rewards = 0

    for idx in range(len(test_dataset.data)):
        # Generate an evaluation observation frame.
        # obs = cv2.resize(np.array(test_dataset.data[idx]).astype(np.float32), (width, height),
        #                  interpolation=cv2.INTER_CUBIC)
        # obs = obs.reshape(width, height, 1)

        obs = test_dataset.data[idx]

        # Predict an action based on the observation.
        logits = trained_agent.act(obs)
        action = np.argmax(logits)

        # Score the prediction.
        reward = 1 if action == test_dataset.targets[idx] else 0
        total_rewards += reward

    print('Accuracy: {:.2f}%'.format(total_rewards / len(test_dataset.data) * 100.0))


if __name__ == '__main__':
    TRAIN = True  # train or test
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64  # minibatch size
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR = 5e-4  # learning rate
    UPDATE_EVERY = 10000  # how often to update the target network
    SAVE_EVERY = 10000  # how often to save the network to disk
    MAX_TIMESTEPS = 60000
    N_EPISODES = 10
    SAVE_DIR = "train/"

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.MNIST('files/', train=True, download=True,
                                               transform=transform)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    shuffle=True,
                                                    num_workers=1)

    test_dataset = torchvision.datasets.MNIST('files/', train=False, download=True,
                                              transform=transform)

    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                   num_workers=1)

    # train_dataset = train_dataset.map(normalize)
    # test_dataset = test_dataset.map(normalize)

    env = MnistGym(width=28, height=28, channels=1, dataset=(train_dataset.data, train_dataset.targets))
    # env = MnistGym(width=28, height=28, channels=1, dataset=train_data_loader)

    if TRAIN:
        # init agent
        agent = DQNAgent(action_size=env.action_space.n,
                         seed=0,
                         lr=LR,
                         gamma=GAMMA,
                         tau=TAU,
                         buffer_size=BUFFER_SIZE,
                         batch_size=BATCH_SIZE,
                         update_every=UPDATE_EVERY)

        # train and get the scores
        scores = train(n_episodes=N_EPISODES, max_t=MAX_TIMESTEPS)
        # plot the running mean of scores
        # N = 100  # running mean window
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.plot(
        #     np.convolve(np.array(scores), np.ones((N,)) / N, mode='valid'))
        # plt.ylabel('Score')
        # plt.xlabel('Timestep #')
        # plt.show()
    else:
        # init a new agent
        trained_agent = DQNAgent(action_size=env.action_space.n,
                                 seed=0)

        # replace the weights with the trained weights
        trained_agent.qnetwork_local.load_state_dict(
            torch.load(SAVE_DIR + '22_mnist_model_10000.pth'))

        # enable inference mode
        trained_agent.qnetwork_local.eval()

        # test and save results to disk
        test(trained_agent)
