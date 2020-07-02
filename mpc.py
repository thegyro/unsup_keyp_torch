import glob
import os
import pickle

import imageio
import numpy as np
import torch
import torch.nn.functional as F

import hyperparameters
import train_keyp_inverse_forward
import train_keyp_pred
import utils

import matplotlib.pyplot as plt
import gym

from visualizer import viz_track, viz_imgseq_goal


class MPC:
    def __init__(self, model, goal_state, H=50):
        """

        :param model: f(s_t, a_t) -> del(s_t) ; s_{t+1} = s_t + del(s_t)
        :param state_dim: = 2*num_keyp
        :param action_dim: action_dim
        """
        self.model = model

        self.H = H
        self.num_sample_seq = 500
        self.goal_state = goal_state

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
        actions_batch = (l-h) * torch.rand(self.num_sample_seq, self.H, 4) + h
        #actions_batch = torch.stack(actions, dim=0) # N x T x 4

        state_batch = state.unsqueeze(0)
        state_batch = state_batch.repeat((self.num_sample_seq, 1, 1))# N x N_K x 2

        next_state_batch = self.predict_next_states(state_batch, actions_batch)

        costs = self.cost_fn(next_state_batch, self.goal_state)
        min_idx = costs.argmin()
        min_cost = costs[min_idx]

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

def convert_img_torch(img_seq):
    if not np.issubdtype(img_seq.dtype, np.uint8):
        raise ValueError('Expected image to be of type {}, but got type {}.'.format(
            np.uint8, img_seq.dtype))
    img_seq = img_seq.astype(np.float32) / 255.0 - 0.5

    return torch.from_numpy(img_seq).permute(0,3,1,2)

def check_start_goal(start, goal):
    start_img, start_keyp = start
    goal_img, goal_keyp = goal

    start_img = utils.unnormalize_image(start_img)
    goal_img = utils.unnormalize_image(goal_img)

    start_keyp, mu_s = utils.project_keyp(start_keyp)
    goal_keyp, mu_g = utils.project_keyp(goal_keyp)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.imshow(start_img)
    ax1.scatter(start_keyp[:,0], start_keyp[:,1], c=mu_s,cmap='Reds')
    ax1.set_title("Starting Keypoint State")

    ax2.imshow(goal_img)
    ax2.scatter(goal_keyp[:,0], goal_keyp[:,1], c=mu_g,cmap='Greens')
    ax2.set_title("Goal Keypoint State")

    plt.show()

def test_start_end(args):
    data = np.load(os.path.join(args.data_dir, args.save_path + ".npz"), allow_pickle=True)

    img_seq = data['image']
    action_seq = data['action'].astype(np.float32)

    img_seq = convert_img_torch(img_seq)
    print(img_seq.shape)
    start_img = img_seq[0]
    goal_img = img_seq[-1]

    #top_9_idx = [46, 26, 51, 25, 3, 22, 20, 39, 19]
    #top_9_idx = [46, 26, 51, 25, 3, 22, 20, 39, 19, 35, 52, 14, 27, 38, 34]

    model = load_model(args)
    with torch.no_grad():
        start_keyp = model.img_to_keyp(start_img[None, None, Ellipsis])[0,0] # num_keyp x 3
        goal_keyp  = model.img_to_keyp(goal_img[None, None, Ellipsis])[0,0]

        # start_keyp = start_keyp[top_9_idx]
        # goal_keyp = goal_keyp[top_9_idx]

        start_img_np = utils.img_torch_to_numpy(start_img)
        goal_img_np = utils.img_torch_to_numpy(goal_img)
        check_start_goal((start_img_np, start_keyp.cpu().numpy()),
                         (goal_img_np, goal_keyp.cpu().numpy()))


def convert_to_pixel(object_pos, M):
    object_pos = np.array([object_pos[0], object_pos[1], object_pos[2], 1]).astype(np.float32)
    object_pixel = M.dot(object_pos)[:2] * (64.0/300.0)
    return object_pixel


def evaluate_control_success(args):
    files = glob.glob(os.path.join(args.data_dir, "*.npz"))
    count = 0

    model = load_model(args)
    M = np.load('tmp_data/proj.npy')

    count = 0
    num_steps = 0
    for i,f in enumerate(files):
        data = np.load(f, allow_pickle=True)

        img_seq = data['image']
        action_seq = data['action'].astype(np.float32)
        goal_pos_w = data['obs'][0]['desired_goal']

        print("To reach Distance:", np.linalg.norm(data['obs'][-1]['achieved_goal'] - goal_pos_w))

        goal_pos = convert_to_pixel(goal_pos_w, M)
        img_seq = convert_img_torch(img_seq)

        start_img = img_seq[0]
        goal_img = img_seq[-1]

        with torch.no_grad():
            start_keyp = model.img_to_keyp(start_img[None, None, Ellipsis])[0,0, :,:2] # num_keyp x 2
            goal_keyp  = model.img_to_keyp(goal_img[None, None, Ellipsis])[0,0, :,:2]

            env = gym.make('FetchReach-v1')
            env.seed(args.seed)

            mpc = MPC(model, goal_keyp, H = args.horizon)

            env.reset()
            keyp = start_keyp
            frames = []
            reached = False
            for t in range(args.max_episode_steps):
                action = mpc.select_min_cost_action(keyp).cpu().numpy()
                x, _, done, _ = env.step(action)
                im = utils.get_frame(env)
                frames.append(im)
                keyp = mpc.get_keyp_state(im)

                gripper_pos = x['achieved_goal']
                if np.linalg.norm(gripper_pos - goal_pos_w) <= 0.03:
                    num_steps += t
                    reached = True
                    break

            if reached:
                print("Reached")
                count += 1
                frames = np.stack(frames)

                l_dir = args.train_dir if args.is_train else args.test_dir
                save_dir = os.path.join(args.vids_dir, "control", args.vids_path)
                if not os.path.isdir(save_dir): os.makedirs(save_dir)
                save_path = os.path.join(save_dir, l_dir + "_{}_seed_{}.mp4".format(i, args.seed))
                viz_imgseq_goal(frames, goal_pos, unnormalize=False, save_path=save_path)
            else:
                print("Did not reach")

    print("Success Rate: ", float(count) / len(files))
    print("Average Num of steps: ", float(num_steps)/count)

def check_recon(args):
    data = np.load(os.path.join(args.data_dir, args.save_path + ".npz"), allow_pickle=True)

    img_seq = data['image']
    action_seq = data['action']

    img_seq = convert_img_torch(img_seq).unsqueeze(0)
    action_seq = torch.from_numpy(action_seq.astype(np.float32)).unsqueeze(0)

    model = load_model(args)

    with torch.no_grad():
        keypoints_seq, heatmaps_seq, recon_img_seq, pred_img_seq, pred_keyp_seq = \
            model(img_seq, action_seq)

        print("LOSS:", F.mse_loss(img_seq, recon_img_seq, reduction='sum') / ((img_seq.shape[0]) * img_seq.shape[1]))
        img_seq_np, recon_img_seq_np = utils.img_torch_to_numpy(img_seq), utils.img_torch_to_numpy(recon_img_seq)
        keypoints_seq_np = keypoints_seq.cpu().numpy()

        d = {'img': img_seq_np,
             'pred_img': recon_img_seq_np,
             'keyp': keypoints_seq_np,
             'heatmap': heatmaps_seq.permute(0, 1, 3, 4, 2).cpu().numpy(),
             'action': action_seq.cpu().numpy() if 'action' in data else None}

        tmp_save_path = 'tmp_data/{}_GOAL_data_{}'.format("test", args.save_path)
        print("Save intermediate data path: ", tmp_save_path)
        np.savez(tmp_save_path, **d)

        save_path = 'vids/check.mp4'
        viz_track(img_seq_np[0], recon_img_seq_np[0], keypoints_seq_np[0], True, 100, save_path)

def test_env(args):

    env = gym.make('FetchPickAndPlace-v1')
    env.seed(args.seed)

    #env = pickle.load(open('tmp_env_state/env.pickle', 'rb'))

    #env.reset()

    M = np.load('tmp_data/proj.npy')
    x = env.reset()
    object_pos = x['achieved_goal']
    goal_pos = x['desired_goal']

    object_pos = np.array([object_pos[0], object_pos[1], object_pos[2], 1]).astype(np.float32)
    goal_pos = np.array([goal_pos[0], goal_pos[1], goal_pos[2], 1]).astype(np.float32)

    object_pixel = M.dot(object_pos)[:2] * (64.0/300.0)
    goal_pixel = M.dot(goal_pos)[:2] * (64.0/300.0)

    plt.imshow(utils.get_frame(env))
    plt.scatter(object_pixel[0], object_pixel[1], color='y')
    plt.scatter(goal_pixel[0], goal_pixel[1], color='g')
    plt.show()

if __name__ == "__main__":
    from register_args import get_argparse
    args = get_argparse(False).parse_args()

    args.data_dir = "data/goal/fetch_reach_sep"
    args.save_path = "fetch_reach_goal_0"
    args.max_episode_steps = 50
    args.horizon = 25
    args.inv_fwd = False

    utils.set_seed_everywhere(args.seed)

    #test_start_end(args)
    #main(args)

    evaluate_control_success(args)

    #check_recon(args)
    #test_env(args)



