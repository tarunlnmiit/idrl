import argparse
import time
import numpy as np
import gym
import torch
from agent import DQNAgent
from gym import wrappers
import torch.nn.functional as F
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from get_saliency import *


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
        logits = trained_agent.act(observation)
        action = np.argmax(logits.cpu().data.numpy())
        # print(action, type(action))
        observation, reward, done, info = env.step(action)
        score += reward

        # prob = F.softmax(action)

        history['ins'].append(observation)
        # history['hx'].append(hx.squeeze(0).data.numpy())
        # history['cx'].append(cx.squeeze(0).data.numpy())
        history['logits'].append(logits.data.numpy()[0])
        # history['values'].append(value.data.numpy()[0])
        # history['outs'].append(prob.data.numpy()[0])
        print('\tstep # {}, reward {:.0f}'.format(step, score), end='\r')

        if done:
            print('GAME OVER! score={}'.format(score))
            break
    env.close()
    return history


def main(env_name, checkpoint, num_frames, first_frame, resolution, save_dir, density, radius, prefix, overfit):
    env = gym.make(env_name)
    meta = get_env_meta(env_name)
    N_STEPS_PER_GAME = 10000

    import glob
    model_type = 'dropout'
    models = sorted(glob.glob('train/{}/*'.format(model_type)))
    print(models)

    # for i in range(2, 6):
    # model_name = 'model_17_#{}'.format(i)
    for model_name in models:
        model_name = '{}/{}'.format(model_type, model_name.split('/')[-1])

        # init a new agent
        trained_agent = DQNAgent(state_size=4,
                                 action_size=env.action_space.n,
                                 seed=0)

        # replace the weights with the trained weights
        trained_agent.qnetwork_local.load_state_dict(
            torch.load('train/{}'.format(model_name), map_location=DEVICE))

        # enable inference mode
        trained_agent.qnetwork_local.eval()

        history = test(env, trained_agent, N_STEPS_PER_GAME)

        movie_title = r"{}-{}-{}-{}-{}.mp4".format(model_type, prefix, num_frames, env_name.lower(), model_name.split('/')[-1])
        print('\tmaking movie "{}" using checkpoint at {}'.format(movie_title, 'train/{}'.format(model_name)))

        start = time.time()
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=movie_title, artist='tgupta', comment='atari-saliency-video')
        writer = FFMpegWriter(fps=8, metadata=metadata)

        prog = ''
        total_frames = len(history['ins'])
        print(total_frames)
        num_frames = total_frames
        f = plt.figure(figsize=[6, 6 * 1.3], dpi=resolution)
        with writer.saving(f, save_dir + movie_title, resolution):
            for i in range(num_frames):
                ix = first_frame + i
                if ix < total_frames:  # prevent loop from trying to process a frame ix greater than rollout length
                    frame = history['ins'][ix].squeeze().copy()
                    actor_saliency = score_frame(trained_agent, history, ix, radius, density, interp_func=occlude, mode='actor')
                    # critic_saliency = score_frame(model, history, ix, radius, density, interp_func=occlude, mode='critic')

                    frame = saliency_on_atari_frame(actor_saliency, frame, fudge_factor=meta['actor_ff'], channel=2)
                    # frame = saliency_on_atari_frame(critic_saliency, frame, fudge_factor=meta['critic_ff'], channel=0)

                    plt.imshow(frame)
                    plt.title(env_name.lower(), fontsize=15)
                    writer.grab_frame()
                    f.clear()

                    tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
                    print('\ttime: {} | progress: {:.1f}%'.format(tstr, 100 * i / min(num_frames, total_frames)), end='\r')
        print('\nfinished.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env', default='SpaceInvaders-v0', type=str, help='gym environment')
    parser.add_argument('-d', '--density', default=5, type=int, help='density of grid of gaussian blurs')
    parser.add_argument('-r', '--radius', default=5, type=int, help='radius of gaussian blur')
    parser.add_argument('-f', '--num_frames', default=20, type=int, help='number of frames in movie')
    parser.add_argument('-i', '--first_frame', default=150, type=int, help='index of first frame')
    parser.add_argument('-dpi', '--resolution', default=75, type=int, help='resolution (dpi)')
    parser.add_argument('-s', '--save_dir', default='movies/', type=str, help='dir to save agent logs and checkpoints')
    parser.add_argument('-p', '--prefix', default='default', type=str, help='prefix to help make video name unique')
    parser.add_argument('-c', '--checkpoint', default='*.pth', type=str, help='checkpoint name (in case there is more than one')
    parser.add_argument('-o', '--overfit_mode', default=False, type=bool, help='analyze an overfit environment (see paper)')
    args = parser.parse_args()

    main(args.env, args.checkpoint, args.num_frames, args.first_frame, args.resolution,
        args.save_dir, args.density, args.radius, args.prefix, args.overfit_mode)
