import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

import sys

import skimage
from skimage.transform import resize
from skimage.util import img_as_ubyte

import gym
import fetch_env_custom


class cpselect_recorder:
    def __init__(self, img1):

        plt.style.use('dark_background')
        self.fig =  plt.figure(figsize=(5, 5))
        self.Ax0 = self.fig.add_subplot(1,1,1)

        self.Ax0.imshow(img1)
        self.Ax0.axis('off')

        self.fig.canvas.mpl_connect('button_press_event', self.handler)
        self.left_x = []
        self.left_y = []

    def handler(self, event):
        circle = plt.Circle((event.xdata, event.ydata), color='r', radius=1)
        if event.inaxes == self.Ax0:
            self.left_x.append(event.xdata)
            self.left_y.append(event.ydata)
            self.Ax0.add_artist(circle)

        self.fig.canvas.draw_idle()
        plt.show()

def cpselect(img1):
    point = cpselect_recorder(img1)
    plt.show()
    point_left = np.concatenate([(np.array(point.left_x))[..., np.newaxis],
                                 (np.array(point.left_y))[..., np.newaxis]], axis=1)

    return point_left.astype(np.float32)

def get_frame(env, crop=(80,350), size=(64,64)):
    frame = env.render(mode='rgb_array')
    if crop: frame = frame[crop[0]:crop[1], crop[0]:crop[1]]
    #frame = img_as_ubyte(resize(frame, size))
    return frame

def collect_projection_data(args):
    num_frames = args.num_frames

    crop = (50, 350)
    size = (64, 64)

    #env = gym.make("FetchPushCustom-v1", n_substeps=20)
    env = gym.make("FetchPickAndPlace-v1")
    env.seed(args.seed)

    source_pts = np.zeros((2*num_frames, 3), dtype=np.float32)
    target_pts = np.zeros((2*num_frames, 2), dtype=np.float32)

    n = 0
    while n < 2*num_frames:
        x = env.reset()
        im = get_frame(env, crop, size)

        points = cpselect(im)

        if len(points) > 0:
            source_pts[n] = x['achieved_goal']
            source_pts[n + 1] = x['desired_goal']

            target_pts[n] = points[0]
            target_pts[n + 1] = points[1]

            n += 2

    data = {}
    data['world_coords'] = source_pts
    data['pixel_coords'] = target_pts
    
    for n in range(2*num_frames):
        s, t = data['world_coords'][n], data['pixel_coords'][n]
        print(n, s, t)

    if not os.path.isdir(args.dir_name): os.makedirs(args.dir_name)
    np.savez(os.path.join(args.dir_name, args.save_path + ".npz") , **data)


def get_grip_pos(env):
    return np.array(env.sim.data.site_xpos[env.sim.model.site_name2id("grip_site")]).astype(np.float32)

def collect_projection_data_sawyer(args):
    num_frames = args.num_frames

    # env = gym.make("FetchPushCustom-v1", n_substeps=20)
    import robosuite as suite
    env = suite.make(
        "SawyerLift",
        has_renderer=False,  # no on-screen renderer
        has_offscreen_renderer=True,  # off-screen renderer is required for camera observations
        use_camera_obs=True,  # use camera observations
        camera_name='sideview',  # use "agentview" camera
        use_object_obs=True,  # no object feature when training on pixels
        gripper_visualization=True,
        ignore_done=True,
        control_freq=30)

    source_pts = []
    target_pts = []

    n = 0
    while n < num_frames:
        x = env.reset()
        for k in range(3):
            for i in range(10):
                x, _, _, _  = env.step(np.random.randn(env.dof))

            im = skimage.img_as_ubyte(skimage.transform.rotate(x['image'], 180))
            points = cpselect(im)

            if len(points) > 0:
                source_pts.append(get_grip_pos(env))
                target_pts.append(points[0])
                print(len(source_pts))

        n += 1

    data = {}
    data['world_coords'] = np.stack(source_pts).astype(np.float32)
    data['pixel_coords'] = np.stack(target_pts).astype(np.float32)

    for n in range(len(source_pts)):
        s, t = data['world_coords'][n], data['pixel_coords'][n]
        print(n, s, t)

    if not os.path.isdir(args.dir_name): os.makedirs(args.dir_name)
    print(os.path.join(args.dir_name, args.save_path + ".npz"))
    np.savez(os.path.join(args.dir_name, args.save_path + ".npz"), **data)

def learn_proj_matrix(args):
    files = glob.glob(os.path.join(args.dir_name, "*.npz"))
    print(files)

    X = []
    y = []
    for file in files:
        data = np.load(file)

        wc = data['world_coords']
        X.append(np.hstack((wc, np.ones((wc.shape[0], 1), dtype=np.float32))))

        pc = data['pixel_coords']
        y.append(np.hstack((pc, np.ones((pc.shape[0], 1), dtype=np.float32))))


    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    n = X.shape[0]
    n_train = int(0.95 * n)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(X.shape, y.shape)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)

    print('Coefficients: \n', reg.coef_)
    # The mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(y_test, reg.predict(X_test)))
    print('Mean squared error: %.2f' % mean_squared_error(y_test, reg.coef_.dot(X_test.T).T))

    # M = np.linalg.pinv(X).dot(y)
    # print(M, M.shape)
    #print(np.sum(np.square(X.dot(M)-y))/X.shape[0])

    return reg.coef_

def check_pred(M, args):
    num_frames = args.num_frames

    crop = (50, 350)
    size = (64, 64)

    #env = gym.make("FetchPushCustom-v1", n_substeps=20)
    env = gym.make('FetchPickAndPlace-v1')
    #env = gym.make('FetchReach-v1')
    env.seed(args.seed)

    for n in range(num_frames):
        x = env.reset()
        im = get_frame(env, crop, size)

        X = np.zeros((2, 3))
        X[0], X[1] = x['achieved_goal'], x['desired_goal']
        X = np.hstack((X, np.ones((2,1), dtype=np.float32)))

        y = (M.dot(X.T)).T

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1,1,1)
        ax.imshow(im)
        ax.axis('off')

        ax.scatter(y[:,0], y[:,1], color='y')
        plt.show()


def check_pred_sawyer(M, args):
    num_frames = args.num_frames


    #env = gym.make("FetchPushCustom-v1", n_substeps=20)
    import robosuite as suite
    env = suite.make(
        "SawyerLift",
        has_renderer=False,  # no on-screen renderer
        has_offscreen_renderer=True,  # off-screen renderer is required for camera observations
        use_camera_obs=True,  # use camera observations
        camera_name='sideview',  # use "agentview" camera
        use_object_obs=True,  # no object feature when training on pixels
        gripper_visualization=True,
        ignore_done=True,
        control_freq=30)

    for n in range(num_frames):
        x = env.reset()
        for i in range(10):
            x, _, _, _ = env.step(np.random.randn(env.dof))
        im = skimage.img_as_ubyte(skimage.transform.rotate(x['image'], 180))

        X = np.zeros((2, 3))
        X[0] = x['cube_pos']
        X[1] = np.array(env.sim.data.site_xpos[env.sim.model.site_name2id("grip_site")])
        X = np.hstack((X, np.ones((2,1), dtype=np.float32)))
        print(X)

        y = (M.dot(X.T)).T

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1,1,1)
        ax.imshow(im)
        ax.axis('off')

        ax.scatter(y[:,0], y[:,1], color='y')
        plt.show()

if __name__ == "__main__":
    # data = np.load('data/fetch_push_25hz/orig/fetch_push_1.npz')
    # img_seq = data['image']
    # cpselect(img_seq[0])

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name", default='data/projection/fetch_pick')
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--save_path", default="fetch_pick_proj")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    #collect_projection_data_sawyer(args)

    args.save_path = "sawyer_reach_joint_proj"
    M = learn_proj_matrix(args)
    np.save("tmp_data/proj_sawyer.npy", M)
    check_pred_sawyer(M, args)