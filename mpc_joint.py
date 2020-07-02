import glob
import os
import pickle
from itertools import islice

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
from skimage.transform import rotate

import hyperparameters
import train_keyp_inverse_forward
import train_keyp_pred
import utils

import matplotlib.pyplot as plt
import gym

from visualizer import viz_track, viz_imgseq_goal, viz_imgseq

import robosuite as suite


class MPC:
    def __init__(self, model, goal_state, action_dim=4, H=50):
        """

        :param model: f(s_t, a_t) -> del(s_t) ; s_{t+1} = s_t + del(s_t)
        :param state_dim: = 2*num_keyp
        :param action_dim: action_dim
        """
        self.model = model

        self.H = H
        self.num_sample_seq = 1000
        self.goal_state = goal_state
        self.action_dim = action_dim

    def predict_next_states(self, state, action):
        """

        :param state: N x num_keyp x 2
        :param action: N x T x action_dim
        :return: next_state: N x (T-1) x num_keyp x 2
        """
        next_states = self.model.keyp_pred_net.unroll(state, action)

        state = state[:, None, :, :]
        next_states = torch.cat((state, next_states), dim=1)

        return next_states

    def select_min_cost_action(self, state):
        """

        :param state: num_keyp x 2
        :return: action: action_dim,
        """
        # actions = []
        # for n in range(self.num_sample_seq):
        #     #actions.append(np.random.uniform(-1, 1, (self.H, 4)))
        #     l, h = -1, 1
        #     act = (l - h)*torch.rand(self.H, 4) + h
        #     actions.append(act)

        l, h = -1, 1
        actions_batch = (l-h) * torch.rand(self.num_sample_seq, self.H, self.action_dim) + h
        #actions_batch = torch.stack(actions, dim=0) # N x T x 4

        state_batch = state.unsqueeze(0)
        state_batch = state_batch.repeat((self.num_sample_seq, 1, 1))# N x N_K x 2

        next_state_batch = self.predict_next_states(state_batch, actions_batch)

        costs = self.cost_fn(next_state_batch, self.goal_state)
        min_idx = costs.argmin()
        min_cost = costs[min_idx]
        #print(min_cost)

        return actions_batch[min_idx][0]

    def cost_fn(self, state_batch, goal_state):
        """

        :param state_batch: N x T x num_keyp x 2
        :param goal_state: num_keyp x 2
        :return: cost: (N,)
        """

        goal_state = goal_state[None, None, :, :]
        curr_state = state_batch[:, :, :, :]
        T = state_batch.shape[1]
        cost = torch.sum((curr_state - goal_state)**2, dim=(1,2,3))/T
        #cost = torch.sum((curr_state - goal_state) ** 2, dim=(1, 2)) / T
        return cost

    def get_keyp_state(self, im):
        im = im[np.newaxis, :, :, :]
        im = convert_img_torch(im)
        keyp = self.model.img_to_keyp(im.unsqueeze(0))[0, 0, :, :2]
        return keyp

def convert_img_torch(img_seq):
    if not np.issubdtype(img_seq.dtype, np.uint8):
        raise ValueError('Expected image to be of type {}, but got type {}.'.format(
            np.uint8, img_seq.dtype))
    img_seq = img_seq.astype(np.float32) / 255.0 - 0.5

    return torch.from_numpy(img_seq).permute(0,3,1,2)

def convert_to_pixel(object_pos, M):
    object_pos = np.array([object_pos[0], object_pos[1], object_pos[2], 1]).astype(np.float32)
    object_pixel = M.dot(object_pos)[:2] * (64.0/256.0)
    return object_pixel

def load_model(args):
    utils.set_seed_everywhere(args.seed)
    cfg = hyperparameters.get_config(args)
    cfg.data_shapes = {'image': (None, 16, 3, 64, 64)}

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    if not args.inv_fwd:
        model = train_keyp_pred.KeypointModel(cfg).to(device)
    else:
        model = train_keyp_inverse_forward.KeypointModel(cfg).to(device)

    checkpoint_path = os.path.join(args.pretrained_path, "_ckpt_epoch_" + args.ckpt + ".ckpt")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    print("Loading model from: ", checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Load complete")

    return model

def evaluate_control_success_sawyer(args):
    file = glob.glob(os.path.join(args.data_dir, "*.npz"))
    data = np.load(file[0], allow_pickle=True)
    goal_img_seq = data['image']
    grip_pos_seq = data['grip_pos']
    goal_img_seq = convert_img_torch(goal_img_seq)

    model = load_model(args)
    M = np.load('tmp_data/proj_sawyer.npy')

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

    count = 0
    num_goals = goal_img_seq.shape[0]
    for i in range(num_goals):
        with torch.no_grad():
            goal_img = goal_img_seq[i]
            goal_pos_w = grip_pos_seq[i]
            goal_pos_pixel = convert_to_pixel(goal_pos_w, M)

            goal_keyp  = model.img_to_keyp(goal_img[None, None, Ellipsis])[0,0, :,:2]

            mpc = MPC(model, goal_keyp, action_dim=args.action_dim, H = args.horizon)

            x = env.reset()
            cur_pos_w = get_grip_pos(env)
            print("To reach Distance:", np.linalg.norm(cur_pos_w - goal_pos_w))

            keyp = mpc.get_keyp_state( img_as_ubyte(rotate(x['image'], 180)))
            frames = []
            reached = False
            for t in range(args.max_episode_steps):
                action = mpc.select_min_cost_action(keyp).cpu().numpy()
                x, _, done, _ = env.step(action)
                im = img_as_ubyte(rotate(x['image'], 180))
                frames.append(im)
                keyp = mpc.get_keyp_state(im)

                cur_pos_w = get_grip_pos(env)
                dist = np.linalg.norm(cur_pos_w - goal_pos_w)
                #print(dist)
                if dist < 0.05:
                    reached = True
                    break

            frames = np.stack(frames)
            if reached:
                print("Reached")
                count += 1
            else:
                print("Did not reach")

            #l_dir = args.train_dir if args.is_train else args.test_dir
            #save_dir = os.path.join(args.vids_dir, "control", args.vids_path)
            #if not os.path.isdir(save_dir): os.makedirs(save_dir)
            #save_path = os.path.join(save_dir, l_dir + "_{}_{}_seed_{}.mp4".format(i,reached,  args.seed))
            #viz_imgseq_goal(frames, goal_pos_pixel, unnormalize=False, save_path=save_path)

    print("Success Rate: ", float(count) / num_goals)

def get_start_frame(return_pos=False):
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

    x = env.reset()
    start_pos = get_grip_pos(env)
    im = img_as_ubyte(rotate(x['image'], 180))
    return  im if not return_pos else im, start_pos

def get_grip_pos(env):
    return np.array(env.sim.data.site_xpos[env.sim.model.site_name2id("grip_site")]).astype(np.float32)

def check_start_goal(start, goal):
    start_img, start_keyp, start_pos = start
    goal_img, goal_keyp, goal_pos = goal

    start_img = utils.unnormalize_image(start_img)
    goal_img = utils.unnormalize_image(goal_img)

    start_keyp, mu_s = utils.project_keyp(start_keyp)
    goal_keyp, mu_g = utils.project_keyp(goal_keyp)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.imshow(start_img)
    ax1.scatter(start_keyp[:,0], start_keyp[:,1], c=mu_s,cmap='Reds')
    ax1.scatter(start_pos[0], start_pos[1], color='y', marker='x', s=75)
    ax1.set_title("Starting Keypoint State")

    ax2.imshow(goal_img)
    ax2.scatter(goal_keyp[:,0], goal_keyp[:,1], c=mu_g,cmap='Greens')
    ax2.scatter(goal_pos[0], goal_pos[1], color='y', marker='x', s=75)
    ax2.set_title("Goal Keypoint State")

    plt.show()

def test_start_end(args):
    data = np.load(os.path.join(args.data_dir, args.save_path + ".npz"), allow_pickle=True)
    M = np.load('tmp_data/proj_sawyer.npy')

    img_seq = data['image']
    grip_pos_seq = data['grip_pos']

    img_seq = convert_img_torch(img_seq)
    im, start_pos = get_start_frame(True)
    start_img = convert_img_torch(im[None, Ellipsis])[0]
    goal_img = img_seq[-1]

    start_pos = convert_to_pixel(start_pos, M)
    goal_pos = convert_to_pixel(grip_pos_seq[-1], M)

    model = load_model(args)
    with torch.no_grad():
        start_keyp = model.img_to_keyp(start_img[None, None, Ellipsis])[0,0] # num_keyp x 3
        goal_keyp  = model.img_to_keyp(goal_img[None, None, Ellipsis])[0,0]

        start_img_np = utils.img_torch_to_numpy(start_img)
        goal_img_np = utils.img_torch_to_numpy(goal_img)
        check_start_goal((start_img_np, start_keyp.cpu().numpy(), start_pos),
                         (goal_img_np, goal_keyp.cpu().numpy(), goal_pos))

def sample_goal_frames(args):
    files = glob.glob(os.path.join(args.data_dir, "*.npz"))

    goal_imgs = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        img_seq = data['image'] # 256 x 64 x 64 x 3
        goal_imgs.append(img_seq[50])
        goal_imgs.append(img_seq[100])

    goal_imgs = np.stack(goal_imgs)

    dir_name = "data/goal/sawyer_reach_joint"
    if not os.path.isdir(dir_name): os.makedirs(dir_name)


    np.savez(os.path.join(dir_name, args.save_path + ".npz"), **{'image': goal_imgs})


def sample_goal_frames_env(args):
    env = suite.make(
        "SawyerLift",
        has_renderer=False,  # no on-screen renderer
        has_offscreen_renderer=True,  # off-screen renderer is required for camera observations
        ignore_done=True,  # (optional) never terminates episode
        use_camera_obs=True,  # use camera observations
        camera_height=64,  # set camera height
        camera_width=64,  # set camera width
        camera_name='sideview',  # use "agentview" camera
        use_object_obs=True,  # no object feature when training on pixels
        control_freq=60
    )

    goal_imgs = []
    grip_pos_l = []
    for i in range(50):
        x = env.reset()
        for k in range(100): x, _, _, _ = env.step(np.random.randn(env.dof))

        im = img_as_ubyte(rotate(x['image'], 180))
        goal_imgs.append(im)

        grip_pos = np.array(env.sim.data.site_xpos[env.sim.model.site_name2id("grip_site")]).astype(np.float32)
        grip_pos_l.append(grip_pos)


    goal_imgs = np.stack(goal_imgs)
    grip_pos_l = np.stack(grip_pos_l)
    dir_name = "data/goal/sawyer_reach_joint"
    if not os.path.isdir(dir_name): os.makedirs(dir_name)

    data = {'image': goal_imgs, 'grip_pos': grip_pos_l}
    np.savez(os.path.join(dir_name, args.save_path + ".npz"), **data)

if __name__ == "__main__":
    from register_args import get_argparse
    args = get_argparse(False).parse_args()

    #args.data_dir = "data/sawyer_reach_side_75/test"
    args.data_dir = "data/goal/sawyer_reach_joint"
    args.save_path = "sawyer_reach_joint_goal"
    args.max_episode_steps = 100
    args.horizon = 25
    args.inv_fwd = True

    utils.set_seed_everywhere(args.seed)

    #sample_goal_frames_env(args)

    #test_start_end(args)

    evaluate_control_success_sawyer(args)