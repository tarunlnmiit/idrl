import argparse
import time
import numpy as np
import gym
import torch
from agent import DQNAgent
from sklearn.datasets import load_svmlight_file
from gym import wrappers
import torch.nn.functional as F
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from get_saliency import *
from torchvision import transforms
from mnist_gym import USPSGym
import torchvision
import cv2
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import CnnPolicy
from skimage.color import gray2rgb


occlude = lambda I, mask: I*(1-mask) + gaussian_filter(I, sigma=3)*mask # choose an area to blur
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(env, trained_agent, n_steps_per_game=10000):
    history = {'ins': [], 'logits': [], 'values': [], 'outs': [], 'hx': [], 'cx': []}
    # print(0)

    observation = env.reset()
    score = 0
    for step in range(n_steps_per_game):
        # print(2)
        # print(observation.shape)
        logits, _states = trained_agent.predict(observation)
        # print(logits)
        # action = np.argmax(logits.cpu().data.numpy())
        # action = np.argmax(logits)
        action = logits
        # print(action, type(action))
        observation, reward, done, info = env.step(action)
        score += reward[0]
        # print(reward, type(reward), score, type(score))

        # prob = F.softmax(action)
        # print(observation==info[0]['obs'])
        # print('obs', observation[0][np.nonzero(observation[0])])
        # print('info', info[0]['obs'][np.nonzero(info[0]['obs'])])
        # print(observation.shape)
        # print(info[0]['obs'].shape)
        # print(observation[0].shape)
        # # plt.imshow(observation[0].reshape(64, 64))
        # plt.imshow(info[0]['obs'])
        # plt.show()
        # exit(0)
        history['ins'].append(info[0]['obs'])
        # history['hx'].append(hx.squeeze(0).data.numpy())
        # history['cx'].append(cx.squeeze(0).data.numpy())
        history['logits'].append(logits[0])
        # history['values'].append(value.data.numpy()[0])
        # history['outs'].append(prob.data.numpy()[0])
        print('\tstep # {}, reward {:.0f}'.format(step, score), end='\r')

        if done:
            print('GAME OVER! score={}'.format(score))
            break
    env.close()
    return history


def main(env_name, checkpoint, num_frames, first_frame, resolution, save_dir, density, radius, prefix, overfit):
    # train_dataset = load_svmlight_file('data/mnist_data')
    # x, y = train_dataset
    # x = x.todense()

    test_dataset = load_svmlight_file('data/mmd_mnist_torch.t')
    testx, testy = test_dataset
    testx = testx.todense()

    sortind = np.argsort(testy)
    testx = testx[sortind, :]
    testy = testy[sortind]

    indices = []
    idx = num_frames // 10
    print(idx)
    for ii in range(10):
        for item in list(np.where(testy == ii)[0][:idx]):
            indices.append(item)
    indices = np.array(indices)
    # digitsdat.X[selected[0:testm], :], digitsdat.y[selected[0:testm]]
    test_data = []
    for i in range(len(testx[indices, :])):
        img = testx[indices, :][i].reshape(28, 28)
        dim = (64, 64)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        # test_data.append(np.array([img.flatten()]))
        test_data.append(img)

    print('datalen', len(test_data))

    env = DummyVecEnv([lambda: USPSGym(width=64, height=64, channels=1, dataset=(test_data, testy[indices]))])

    # Grab the observation shape for generating evaluation frames.
    width, height = env.observation_space.shape[0], env.observation_space.shape[1]

    meta = get_env_meta(env_name)
    N_STEPS_PER_GAME = num_frames

    # init a new agent
    # trained_agent = DQNAgent(action_size=env.action_space.n,
    #                          seed=0)
    #
    # # replace the weights with the trained weights
    # trained_agent.qnetwork_local.load_state_dict(
    #     torch.load('train/21_mnist_model.pth', map_location=torch.device('cpu')))
    #
    # # enable inference mode
    # trained_agent.qnetwork_local.eval()
    import glob
    models = sorted(glob.glob('*custom*zip'))
    for model_name in models:
        print(model_name)
        trained_agent = DQN.load(model_name, device=DEVICE)

        history = test(env, trained_agent, N_STEPS_PER_GAME)

        movie_title = "{}-{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower(), model_name)
        print('\tmaking movie "{}" using checkpoint at {}'.format(movie_title, model_name))

        start = time.time()
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=movie_title, artist='tgupta', comment='mnist-saliency-video')
        writer = FFMpegWriter(fps=8, metadata=metadata)

        prog = ''
        total_frames = len(history['ins'])
        print(total_frames)
        f = plt.figure(figsize=[6, 6 * 1.3], dpi=resolution)
        with writer.saving(f, save_dir + movie_title, resolution):
            for i in range(num_frames):
                ix = first_frame + i
                if ix < total_frames:  # prevent loop from trying to process a frame ix greater than rollout length
                    # frame = history['ins'][ix].squeeze().numpy().copy()
                    frame = history['ins'][ix].squeeze().copy()
                    # print(frame.shape)
                    # plt.imshow(frame)
                    # plt.show()
                    # exit(0)
                    frame = gray2rgb(frame)
                    # frame = np.stack((frame, frame, frame), axis=2).reshape(28, 28, 3)
                    # print('f', frame.shape, ix)
                    # frame = history['ins'][ix].squeeze()
                    actor_saliency = score_frame(trained_agent, history, ix, radius, density, interp_func=occlude, mode='actor')
                    # critic_saliency = score_frame(model, history, ix, radius, density, interp_func=occlude, mode='critic')

                    frame = saliency_on_atari_frame(actor_saliency, frame, fudge_factor=meta['actor_ff'], channel=0)
                    # frame = saliency_on_atari_frame(critic_saliency, frame, fudge_factor=meta['critic_ff'], channel=0)

                    plt.imshow(frame)
                    # plt.imshow((frame * 255).astype(np.uint8))
                    # plt.show()
                    # exit(0)
                    plt.title(env_name.lower(), fontsize=15)
                    writer.grab_frame()
                    f.clear()

                    tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
                    print('\ttime: {} | progress: {:.1f}%'.format(tstr, 100 * i / min(num_frames, total_frames)), end='\r')
        print('\nfinished.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env', default='MNIST-v0', type=str, help='gym environment')
    parser.add_argument('-d', '--density', default=5, type=int, help='density of grid of gaussian blurs')
    parser.add_argument('-r', '--radius', default=5, type=int, help='radius of gaussian blur')
    parser.add_argument('-f', '--num_frames', default=500, type=int, help='number of frames in movie')
    parser.add_argument('-i', '--first_frame', default=0, type=int, help='index of first frame')
    parser.add_argument('-dpi', '--resolution', default=75, type=int, help='resolution (dpi)')
    parser.add_argument('-s', '--save_dir', default='movies/', type=str, help='dir to save agent logs and checkpoints')
    parser.add_argument('-p', '--prefix', default='default', type=str, help='prefix to help make video name unique')
    parser.add_argument('-c', '--checkpoint', default='*.pth', type=str, help='checkpoint name (in case there is more than one')
    parser.add_argument('-o', '--overfit_mode', default=False, type=bool, help='analyze an overfit environment (see paper)')
    args = parser.parse_args()

    main(args.env, args.checkpoint, args.num_frames, args.first_frame, args.resolution,
        args.save_dir, args.density, args.radius, args.prefix, args.overfit_mode)
