import argparse
import enum
import os
from itertools import islice

import numpy as np
from scipy.spatial import distance
from npeet import entropy_estimators as ee

from visualizer import viz_track, viz_dynamic_img_top_vid, project_keyp, viz_hmap, viz_img_keyp_history


def align_keypoints(keyp1, keyp2):
    """
    :param keyp1: N x 2
    :param keyp2: N x 2
    :return: match: N x 1 , keyp1[i] aligns with keyp2[match[i]]
    """

    pair_dist = distance.cdist(keyp1, keyp2)
    n = len(keyp1)
    match = np.zeros(n, dtype=np.int)
    for i in range(n):
        min_idx = np.argmin(pair_dist[i])
        match[i] = min_idx
        pair_dist[:, min_idx] = np.inf
    return match

def responsiveness(keyp, actions):
    """

    :param keyp: T x 2
    :param actions: T x 1
    :return:
    """
    assert len(keyp.shape) == 2 and keyp.shape[1] == 2
    resp = ee.mi(keyp, actions)

    return resp

def responsiveness_1(keyp, keyp_diff, actions):

    assert len(keyp.shape) == 2 and keyp.shape[1] == 2

    keyp_combine = np.hstack((actions, keyp))
    print(keyp_combine.shape)
    resp = ee.mi(keyp_combine, keyp_diff)

    return resp


def viz_keyp(args):
    #data = np.load('test_data_80steps.npz')
    #data = np.load('test_data_80steps_robot_push.npz')
    #data = np.load('tmp_data/test_data_80steps_bair_push.npz')
    #data = np.load("tmp_data/test_data_80steps_fetch_push_seed_0.npz")
    #data = np.load("tmp_data/test_data_acro_nofirst_r1_seed_0.npz")
    #data = np.load('tmp_data/test_data_fetch_push_r1_epoch_25_seed_5.npz')
    #data = np.load('tmp_data/test_data_bair_new_hmap_r1_seed_0.npz')
    data = np.load(args.data_path)

    img_seq = data['img']
    recon_img_seq = data['pred_img']
    keyp_seq = data['keyp']
    action_seq = data['action']
    hmap_seq = data['heatmap']

    # k = 7
    # b = 0
    # path = "vids/vid_bair_push_hmap_{}_{}.mp4"
    # viz_hmap(img_seq[b],recon_img_seq[b] , keyp_seq[b], hmap_seq[b],
    #          k,
    #          unnormalize=True, delay=100, save_path=path.format(b,k))

    # b = 1
    # k = 29
    # path = "vids/fetch_nofirst_hmap_k_{}_{}.mp4".format(k, b)
    # viz_dynamic_img(img_seq[b],recon_img_seq[b] , keyp_seq[b], hmap_seq[b],
    #          k,
    #          unnormalize=True, delay=100, save_path=path)

    #viz_dynamic_img_top(img_seq[0],recon_img_seq[0],keyp_seq[0],save_path="vids/vid_dyn_{}.png".format(k))

    # b = 1
    # path = "vids/acro_nofirst_hmap_{}.mp4".format(b)
    # viz_dynamic_img_top_vid(img_seq[b],
    #                     recon_img_seq[b],
    #                     keyp_seq[b], delay=100, save_path=path.format(b))
    b = 1
    #path = "vids/vid_acrobot_80_{}.mp4"
    #path = "vids/bair_push_bottom_resp_{}.mp4".format(b)
    path = "vids/fetch_nofirst_bottom_resp_{}_epoch150.mp4".format(b)
    #path = "vids/acro_nofirst_bottom_resp_{}.mp4".format(b)
    viz_dynamic_img_top_vid(img_seq[b],
                        recon_img_seq[b],
                        keyp_seq[b], delay=100, save_path=path)

    b = 2
    #path = "vids/vid_fetch_push_track_{}.mp4".format(b)
    #path = "vids/vid_acro_nofirst_track_{}.mp4".format(b)
    # path = "vids/bair_push_track_{}.mp4".format(b)
    # viz_track(img_seq[b], recon_img_seq[b], keyp_seq[b], True, 200, path)

def calc_resp_k(keyp_seq, action_seq):
    """
    :param keyp_seq: T x 3
    :param action_seq: T x action_dim
    :return: resp (float)
    """

    keyp_seq, mu = project_keyp(keyp_seq)

    keyp_seq_t = keyp_seq[:-1]
    keyp_seq_t1 = keyp_seq[1:]
    keyp_seq_diff = keyp_seq_t1 - keyp_seq_t
    action_seq = action_seq[:-1]

    return responsiveness(keyp_seq_diff, action_seq)

def calc_resp_all(keyp_seq, action_seq):
    """

    :param keyp_seq: T x N x 3
    :param action_seq: T x action_dim
    :return: resp: (N,1)
    """

    num_keyp = keyp_seq.shape[1]
    resps = []
    for k in range(num_keyp):
        keyp_seq_k = keyp_seq[:, k]
        R = calc_resp_k(keyp_seq_k, action_seq)
        resps.append(R)

    return np.array(resps)

def viz_recon(args):
    data = np.load(args.data_path)

    img_seq_batch = data['img']
    recon_img_seq_batch = data['pred_img']
    keyp_seq_batch = data['keyp']
    action_seq_batch = data['action']
    hmap_seq_batch = data['heatmap']

    batch_size = min(img_seq_batch.shape[0], 3)
    for b in range(batch_size):
        img_seq, keyp_seq = img_seq_batch[b], keyp_seq_batch[b]  # T x N x 3
        recon_img_seq, action_seq = recon_img_seq_batch[b], action_seq_batch[b]  # T x 1

        save_path = args.save_dir + "/{}_recon_b_{}.mp4".format(args.exp_name, b)
        print(save_path)
        viz_track(img_seq, recon_img_seq, keyp_seq, save_path=save_path, annotate=args.annotate)

def viz_recon_keyp(args):
    data = np.load(args.data_path)

    img_seq_batch = data['img']
    recon_img_seq_batch = data['pred_img']
    keyp_seq_batch = data['keyp']
    action_seq_batch = data['action']
    hmap_seq_batch = data['heatmap']

    batch_size = img_seq_batch.shape[0]
    for b in islice(range(batch_size), 20):
        img_seq, keyp_seq = img_seq_batch[b], keyp_seq_batch[b]  # T x N x 3
        recon_img_seq, action_seq = recon_img_seq_batch[b], action_seq_batch[b]  # T x 1
        hmap_seq = hmap_seq_batch[b]

        k = args.k
        resp_all = calc_resp_all(keyp_seq, action_seq)
        top_idx = np.argsort(-resp_all)
        resp_k = resp_all[k]
        rank = np.where(top_idx == k)[0]

        save_dir = os.path.join(args.save_dir, "specific/{}/{}".format(args.exp_name, args.k))
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "{}_recon_keyp_{}_b_{}.mp4".format(args.exp_name, args.k, b))
        print(save_path)

        viz_hmap(img_seq, recon_img_seq, keyp_seq, hmap_seq, args.k, resp=(resp_k, rank),
                 save_path=save_path)

def viz_top_resp(args):
    data = np.load(args.data_path)

    img_seq_batch = data['img']
    #recon_img_seq_batch = data['pred_img']
    keyp_seq_batch = data['keyp']
    action_seq_batch = data['action']
    hmap_seq_batch = data['heatmap']

    batch_size = min(img_seq_batch.shape[0], 3)
    for b in range(batch_size):
        img_seq, keyp_seq = img_seq_batch[b], keyp_seq_batch[b]  # T x N x 3
        #recon_img_seq, action_seq = recon_img_seq_batch[b], action_seq_batch[b]  # T x 1
        action_seq = action_seq_batch[b]

        mu = keyp_seq[0, :, 2]
        top_9_mu = np.argsort((-mu))[:9]
        print(top_9_mu)

        T, num_keyp = keyp_seq.shape[:2]
        print("Timesteps:", T, "Num keypoints: ", num_keyp)

        keyp_seq_proj = np.zeros((T, num_keyp, 2))
        for i in range(keyp_seq.shape[0]):
            keyps, _ = project_keyp(keyp_seq[i])
            keyp_seq_proj[i] = keyps

        resps = []
        for k in range(num_keyp):
            keypoints_k = keyp_seq_proj[:-1, k] # (T-1 x 2)
            keypoints_k_1 = keyp_seq_proj[1:, k] # (T-1 x 2)
            keypoints_diff = keypoints_k_1 - keypoints_k
            actions = action_seq[:-1]

            #R = responsiveness_1(keypoints_diff.copy(), keypoints_k.copy(), actions.copy())
            R = responsiveness(keypoints_diff.copy(), actions.copy())
            print(k, R, keypoints_k.shape, actions.shape)
            resps.append(R)

        resps = np.array(resps)
        top_9 = np.argsort(-resps)[:9]
        bottom_9 = np.argsort(resps)[:9]
        print("Top 9:", list(top_9), resps[top_9])

        save_path = args.save_dir + "/{}_{}_resp_b_{}".format(args.exp_name, "top", b) + ".mp4" # <exp_name>_{}_resp_batch_{}
        print("save path", save_path)
        viz_dynamic_img_top_vid(img_seq,
                                #recon_img_seq,
                                img_seq,
                                keyp_seq, top_9_idx=top_9, delay=100, save_path=save_path)

        print("Bottom 9: ", list(bottom_9), resps[bottom_9])
        save_path = args.save_dir + "/{}_{}_resp_b_{}".format(args.exp_name, "bottom", b) + ".mp4"
        viz_dynamic_img_top_vid(img_seq,
                                #recon_img_seq,
                                img_seq,
                                keyp_seq, top_9_idx=bottom_9, delay=100, save_path=save_path)


def viz_keyp_history_training(args):

    args.data_dir = 'data/fetch_reach_25/test'
    args.file_name = 'fetch_reach'
    args.keyp_path = 'exp_data/fetch_reach_16kp_track_s_0' \
                     '/2020-06-02-03-52-29/training_logs/file_144_frame_37/keyps_history.npy'
    args.file_id = 144
    args.frame_id =37
    args.exp_name = 'fetch_reach_16'

    data = np.load(os.path.join(args.data_dir, args.file_name + "_{}.npz".format(args.file_id)))
    img_seq = data['image']
    img = img_seq[args.frame_id]

    keyp_history = np.load(args.keyp_path)
    num_keyp = keyp_history.shape[1]
    for k in range(num_keyp):
        print("Keypoint ", k)
        save_dir = os.path.join(args.save_dir, "keyp_history")
        if not os.path.isdir(save_dir): os.makedirs(save_dir)

        save_path = save_dir + "/{}_k_{}_keyp_history_{}_{}".format(
            args.exp_name, k, args.file_id, args.frame_id) + ".mp4"
        viz_img_keyp_history(img, keyp_history[:, k], k, save_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='tmp_data/random.npz')
    parser.add_argument('--exp_name', default='test')
    parser.add_argument('--save_dir', default='vids')
    parser.add_argument("--k", default=0)
    parser.add_argument("--annotate", action='store_true')
    #viz_keyp()

    #calc_resp()
    args = parser.parse_args()
    #args.data_path = 'tmp_data/test_data_pred_keyp_loss_reg1e-1_seed_0.npz'
    #args.data_path = 'tmp_data/test_data_fetch_cpu_action_UNROLL100_e_50_seed_0.npz'
    #args.data_path = 'tmp_data/test_data_fetch_pick_action_e_75_seed_0.npz'
    #args.data_path = 'tmp_data/_data_fetch_GOAL_sep_action_e_75_seed_0.npz'
    #args.data_path = 'tmp_data/test_data_fetch_reach_150_seed_0.npz'
    #args.data_path = 'tmp_data/test_data_fetch_reach_25_seed_0.npz'
    #args.data_path = 'tmp_data/test_data_fetch_reach_5_seed_0.npz'
    #args.data_path = 'tmp_data/test_data_fetch_reach_16kp_seed_0.npz'
    #args.data_path = 'tmp_data/test_data_bair_push_32kp_seed_0.npz'
    #args.data_path = 'tmp_data/test_data_bair_push_15k_64kp_seed_0.npz'
    #args.data_path = 'tmp_data/test_data_sawyer_reach_75_32kp_seed_0.npz'
    #args.data_path = 'tmp_data/test_data_sawyer_reach_side_seed_0.npz'
    args.data_path = 'tmp_data/test_data_fetch_reach_inverse_forward_hmap2_seed_0.npz'
    #args.exp_name = "fetch_pick_action_e_75_test"
    #args.exp_name = "sawyer_reach_side"
    args.exp_name = "fetch_reach_inverse_forward"
    args.annotate = True
    args.k = 23

    #viz_recon_keyp(args)
    viz_top_resp(args)
    #viz_recon(args)

    #viz_keyp_history_training(args)
