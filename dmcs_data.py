import argparse
import os

#import dmc2gym
import imageio

from video_env import VideoRecorder
import numpy as np
import random
import glob
import gym
import fetch_env_custom
from skimage.transform import resize
from skimage.transform import rotate
from skimage.util import img_as_ubyte

def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image

def collect_data(args):

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=args.from_pixels,
        height=args.image_size,
        width=args.image_size,
        frame_skip=args.action_repeat,
        channels_first=False,
        episode_length=256
    )

    video = VideoRecorder(args.dir_vid_name, height=args.image_size, width=args.image_size)

    for i in range(args.num_episodes):
        frames = []
        actions = []
        rewards = []

        obs = env.reset()
        if i < 10: video.init()
        done = False
        while not done:

            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            if i < 10: video.record(env)

            frames.append(obs)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs

        frames, actions, rewards = np.stack(frames), np.stack(actions), np.stack(rewards)
        actions, rewards = actions.astype(np.float32), rewards.astype(np.float32)
        print("Frame:", frames.shape, frames.dtype, "Actions: ", actions.shape)

        data = {'image': frames, 'action': actions, 'rewards': rewards}

        file_name = args.domain_name + "_" + args.task_name + "_" + str(i+1) + \
                    ("_" + args.trial if args.trial else "")
        print(file_name)
        if i < 10: video.save(file_name + ".mp4")

        save_dir = os.path.join(args.dir_name, "orig")
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        save_path = os.path.join(save_dir, file_name)
        np.savez(save_path, **data)


def get_frame(env, crop=(80,350), size=(64,64)):
    frame = env.render(mode='rgb_array')
    if crop: frame = frame[crop[0]:crop[1], crop[0]:crop[1]]
    frame = img_as_ubyte(resize(frame, size))
    return frame

def collect_data_fetch(args):
    #env = gym.make("FetchPushCustom-v1", n_substeps=20)
    #env = gym.make("FetchPickAndPlace-v1")
    env = gym.make("FetchReach-v1")
    #env = SawyerReachPushPickPlaceEnv()

    np.random.seed(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)

    video = VideoRecorder(args.dir_vid_name, height=args.image_size, width=args.image_size)
    crop = (50, 350)
    size = (64, 64)

    for i in range(args.num_episodes):
        frames = []
        actions = []
        rewards = []
        obs_l = []

        obs = env.reset()
        im = get_frame(env, crop, size)
        if i < 10: video.init()
        done = False
        t = 0
        while t < 64:

            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            next_im = get_frame(env, crop, size)
            if i < 10: video.record(env)

            frames.append(im)
            obs_l.append(obs)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs
            im = next_im
            t += 1

        frames, actions, rewards = np.stack(frames), np.stack(actions), np.stack(rewards)
        actions, rewards = actions.astype(np.float32), rewards.astype(np.float32)
        #obs_l = np.stack(obs_l).astype(np.float32)
        print("Frame:", frames.shape, frames.dtype, "Actions: ", actions.shape, "Obs: ", obs_l[-1].keys())

        data = {'image': frames,
                'action': actions,
                'rewards': rewards,
                'obs': obs_l}

        file_name = args.domain_name + "_" + args.task_name + "_" + str(i+1) + \
                    ("_" + args.trial if args.trial else "")
        print(file_name)
        if i < 10: video.save(file_name + ".mp4")

        save_dir = os.path.join(args.dir_name, "orig")
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        save_path = os.path.join(save_dir, file_name)
        np.savez(save_path, **data)


def collect_data_robosuite(args):
    import robosuite as suite
    #env = gym.make("FetchPushCustom-v1", n_substeps=20)
    #env = gym.make("FetchPickAndPlace-v1")
    env = suite.make(
        "SawyerLift",
        has_renderer=False,  # no on-screen renderer
        has_offscreen_renderer=True,  # off-screen renderer is required for camera observations
        ignore_done=True,  # (optional) never terminates episode
        use_camera_obs=True,  # use camera observations
        camera_height=64,  # set camera height
        camera_width=64,  # set camera width
        camera_name='sideview',  # use "agentview" camera
        use_object_obs=False,  # no object feature when training on pixels
        control_freq=60
    )
    #env = SawyerReachPushPickPlaceEnv()

    np.random.seed(args.seed)
    random.seed(args.seed)
    #env.seed(args.seed)

    size = (64, 64)

    for i in range(args.num_episodes):
        frames = []
        video_frames = []
        actions = []
        rewards = []
        obs_l = []

        obs = env.reset()
        im = img_as_ubyte(rotate(obs['image'], 180))
        if i < 10: video_frames.append(im)
        done = False
        t = 0
        while t < 256:

            action = np.random.randn(env.dof)
            next_obs, reward, done, _ = env.step(action)
            next_im = img_as_ubyte(rotate(next_obs['image'], 180))
            if i < 10: video_frames.append(next_im)

            frames.append(im)
            obs_l.append(obs)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs
            im = next_im
            t += 1

        frames, actions, rewards = np.stack(frames), np.stack(actions), np.stack(rewards)
        actions, rewards = actions.astype(np.float32), rewards.astype(np.float32)
        #obs_l = np.stack(obs_l).astype(np.float32)
        print("Frame:", frames.shape, frames.dtype, "Actions: ", actions.shape, "Obs: ", obs_l[-1].keys())

        data = {'image': frames,
                'action': actions,
                'rewards': rewards,
                'obs': obs_l}

        file_name = args.domain_name + "_" + args.task_name + "_" + str(i+1) + \
                    ("_" + args.trial if args.trial else "")
        print(file_name)
        if i < 10: imageio.mimsave(os.path.join(args.dir_vid_name, file_name + ".mp4"), video_frames, fps=30)

        save_dir = os.path.join(args.dir_name, "orig")
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        save_path = os.path.join(save_dir, file_name)
        np.savez(save_path, **data)

def train_test_split(args):
    from shutil import copy
    np.random.seed(args.seed)
    random.seed(args.seed)

    file_name = args.domain_name + "_" + args.task_name + "_" + "*.npz"
    load_dir = os.path.join(args.dir_name, "orig", file_name)
    print("Dir: ", load_dir)
    files = glob.glob(load_dir)

    num_files = len(files)
    random.shuffle(files)

    num_train = int(0.75*num_files)
    num_test  = num_files - num_train

    files_train = files[:num_train]
    files_test  = files[num_train:]

    train_dir = os.path.join(args.dir_name, "train")
    if not os.path.isdir(train_dir): os.makedirs(train_dir)
    test_dir = os.path.join(args.dir_name, "test")
    if not os.path.isdir(test_dir): os.makedirs(test_dir)

    print("Copying train files\n")
    for i,f in enumerate(files_train):
        copy(f, train_dir)

    print("Copying test files\n")
    for i,f in enumerate(files_test):
        copy(f, test_dir)

def create_train_split(args):
    from shutil import copy
    np.random.seed(args.seed)
    random.seed(args.seed)

    file_name = args.domain_name + "_" + args.task_name + "_" + "*.npz"

    train_dir = os.path.join(args.dir_name, "train", file_name)
    train_files = glob.glob(train_dir)
    print("Dir: ", train_dir, len(train_files))

    test_dir = os.path.join(args.dir_name, 'test', file_name)
    test_files = glob.glob(test_dir)

    num_train_files = len(train_files)
    #split_fracs = [2/3.0, 1/3.0, 1/6.0]
    #split_fracs = [1/30.0]
    #split_fracs = [1.0/2.0, 1.0/4.0]
    split_fracs = [1/3.0]
    for frac in split_fracs:
        num_split = int((frac * num_train_files))

        dir_split = args.dir_name + "_" + str(num_split)
        if not os.path.isdir(dir_split): os.makedirs(dir_split)
        print("Dir split: ", dir_split)

        train_dir_split = os.path.join(dir_split, "train")
        if not os.path.isdir(train_dir_split): os.makedirs(train_dir_split)
        test_dir_split  = os.path.join(dir_split, "test")
        if not os.path.isdir(test_dir_split): os.makedirs(test_dir_split)

        train_files_split = random.sample(train_files, num_split)
        print("Len of split: ", len(train_files_split))

        print("Copying train files for split: ", train_dir_split)
        for f in train_files_split:
            copy(f, train_dir_split)

        print("Copying test files for split: ", test_dir_split)
        for f in test_files:
            copy(f, test_dir_split)

        print()

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='acrobot')
    parser.add_argument('--task_name', default='swingup')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--from_pixels', type=bool, default=True)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--action_repeat', type=int, default=1)

    parser.add_argument("--dir_name", default='data/all_envs')
    parser.add_argument("--dir_vid_name", default='vids_env')
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--trial", default="")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # #collect_data(args)
    # train_test_split(args)

    #collect_data_fetch(args)
    #collect_data_robosuite(args)
    #train_test_split(args)
    create_train_split(args)