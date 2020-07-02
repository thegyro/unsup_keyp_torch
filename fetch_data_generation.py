import argparse
import os
import pickle
import time

import gym
import numpy as np

import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.transform import resize

import utils

"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
infos = []
frames = []

def get_frame(env, crop=(50,350), size=(64,64)):
    frame = env.render(mode='rgb_array').copy()
    if crop: frame = frame[crop[0]:crop[1], crop[0]:crop[1]]
    frame = img_as_ubyte(resize(frame, size))
    return frame

def main(args):
    #env = gym.make('FetchPickAndPlace-v1')
    env = gym.make('FetchReach-v1')
    env.seed(args.seed)

    numItr = args.num_iter
    env.reset()
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoalReach(env, obs)
        #goToGoal(env, obs)



    if not os.path.isdir(args.dir_name): os.makedirs(args.dir_name)
    save_path = os.path.join(args.dir_name, args.save_path)

    data = {'action': actions, 'obs': observations, 'image': frames, 'info': infos}
    if not args.display:
        np.savez_compressed(save_path, **data)
    #np.savez_compressed(save_path, acs=actions, obs=observations, info=infos, frames=frames) # save the file

def goToGoal(env, lastObs):

    goal = lastObs['desired_goal']
    objectPos = lastObs['observation'][3:6]
    object_rel_pos = lastObs['observation'][6:9]
    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    episodeFrames = []

    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object

    timeStep = 0 #count the total number of timesteps
    episodeObs.append(lastObs)
    if not args.display: episodeFrames.append(get_frame(env))
    else: env.render()

    while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
        #env.render()
        action = [0, 0, 0, 0]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]*6

        action[len(action)-1] = 0.05 #open

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        if not args.display: episodeFrames.append(get_frame(env))
        else: env.render()

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

    #time.sleep(5)

    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
        #env.render()
        action = [0, 0, 0, 0]
        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i]*6

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        if not args.display: episodeFrames.append(get_frame(env))
        else: env.render()

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]


    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
        #env.render()
        action = [0, 0, 0, 0]
        for i in range(len(goal - objectPos)):
            action[i] = (goal - objectPos)[i]*6

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        if not args.display: episodeFrames.append(get_frame(env))
        else: env.render()

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

    while True: #limit the number of timesteps in the episode to a fixed duration
        #env.render()
        action = [0, 0, 0, 0]
        action[len(action)-1] = -0.005 # keep the gripper closed

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        if not args.display: episodeFrames.append(get_frame(env))
        else: env.render()

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

        if timeStep >= env._max_episode_steps: break

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)
    if not args.display: frames.append(episodeFrames)



def goToGoalReach(env, lastObs):

    goal = lastObs['desired_goal']
    objectPos = lastObs['observation'][0:3]
    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    episodeFrames = []

    timeStep = 0 #count the total number of timesteps
    episodeObs.append(lastObs)
    if not args.display: episodeFrames.append(get_frame(env))
    else: env.render()

    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
        #env.render()
        action = [0, 0, 0, 0]
        for i in range(len(goal - objectPos)):
            action[i] = (goal - objectPos)[i]*6

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        if not args.display: episodeFrames.append(get_frame(env))
        else: env.render()

        objectPos = obsDataNew['observation'][0:3]

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)
    if not args.display: frames.append(episodeFrames)

def check(args):
    data = np.load(os.path.join(args.dir_name, args.save_path+".npz"), allow_pickle=True)
    num_seq = data['image'].shape[0]
    print(num_seq)

    for j in range(5):
        fig = plt.figure()
        ax0 = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2)

        i = np.random.randint(0, num_seq)

        img_seq = data['image'][i]

        ax0.imshow(img_seq[0])
        ax0.set_title("Initial")

        ax1.imshow(img_seq[-1])
        ax1.set_title("Final")

        plt.show()

def split(args):
    data = np.load(os.path.join(args.dir_name, args.save_path+".npz"), allow_pickle=True)
    num_seq = data['image'].shape[0]

    target_dir = args.dir_name  + "_sep"
    if not os.path.isdir(target_dir): os.makedirs(target_dir)

    for n in range(num_seq):
        save_path = os.path.join(target_dir, args.save_path + "_" + str(n))

        img_seq = np.stack(data['image'][n][:-1], axis=0)
        action_seq = np.stack(data['action'][n],axis=0)
        obs_seq = data['obs'][n][:-1]

        print(n, img_seq.shape, action_seq.shape, obs_seq[0].keys())
        data_n = {'image': img_seq, 'action': action_seq, 'obs': obs_seq}
        np.savez_compressed(save_path, **data_n)

def test_env(args):

    # env = gym.make('FetchPickAndPlace-v1')
    # env.seed(args.seed)

    env = pickle.load(open('tmp_env_state/env.pickle', 'rb'))

    #env.reset()

    plt.imshow(utils.get_frame(env))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name", default='data/goal/fetch_reach')
    parser.add_argument("--num_iter", type=int, default=1)
    parser.add_argument("--save_path", default="fetch_reach_goal")
    parser.add_argument("--display", action='store_true')
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)

    main(args)
    #check(args)

    #test_env(args)

    #split(args)