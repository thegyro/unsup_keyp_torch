import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np

#from keyp_resp import align_keypoints

IMG_W, IMG_H = (64, 64)
HMAP_W, HMAP_H = (16, 16)


def unnormalize_image(img):
    img = np.clip(img, -0.5, 0.5)
    return ((img + 0.5) * 255).astype(np.uint8)

def project_keyp(keyp):
    x, y, mu = keyp[:, 0], keyp[:, 1], keyp[:, 2]
    #x, y = x[mu >= 0.5], y[mu >= 0.5]
    x, y = 8 * x, 8 * y
    x, y = x + 8, 8 - y
    x, y = (64 / 16) * x, (64 / 16) * y

    N = x.shape[0]

    #return np.hstack((x.reshape((N, 1)), y.reshape((N, 1)), mu.reshape(N,1)))
    return np.hstack((x.reshape((N, 1)), y.reshape((N, 1)))), mu
    #return np.hstack((x.reshape((N, 1)), y.reshape((N, 1))))

def project_keyp_batch(keyp):
    x, y, mu = keyp[:, :, 0], keyp[:, :, 1], keyp[:, :, 2]
    x, y = 8 * x, 8 * y
    x, y = x + 8, 8 - y
    x, y = (64 / 16) * x, (64 / 16) * y

    b_s, N = x.shape[:2]

    return np.concatenate((x.reshape((b_s, N, 1)), y.reshape((b_s, N, 1))), axis=2), mu

def save_img_keyp(img, keyp_history, save_path, k, step_num):
    # img is H x W x 3 and keyp is (x, y, mu)

    num_steps = keyp_history.shape[0]
    img = unnormalize_image(img)
    keyp_history, mu = project_keyp(keyp_history)

    color = np.zeros((num_steps, 4))
    color[:, 0] = 1.0
    color[:, 3] = np.linspace(0.1, 1, num_steps)

    fig = plt.figure()
    plt.imshow(img)
    plt.scatter(keyp_history[:, 0], keyp_history[:, 1], c=color)
    plt.title("Keypoint={}, Step No: {}".format(k, step_num))
    fig.savefig(save_path)
    plt.close()

def viz_img_keyp_history(img, keyp_history, k, save_path):
    # img is H x W x 3 and keyp_history is (N, num_keyp, 3)
    num_steps = keyp_history.shape[0]
    keyp_history, mu = project_keyp(keyp_history)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    f1 = ax1.imshow(img)

    def animate(i):
        keyp = keyp_history[i]

        alpha = (0.9/(num_steps-1)) * i + 0.1
        f2 = ax1.scatter(keyp[0], keyp[1], color='r', alpha=alpha)
        ax1.set_title('k={}, step={}, mu={:.4f}'.format(k, i*500, mu[i]))

        return [f1, f2]

    ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=200, blit=True)

    ani.save(save_path)

def viz_imgseq(image_seq, unnormalize=False, delay=100, save_path=None):
    print(image_seq.shape)
    N = image_seq.shape[0]

    fig = plt.figure()
    frames = []
    for i in range(N):
        img = image_seq[i]

        if unnormalize: img = unnormalize_image(img)

        f1 = plt.imshow(img)
        frames.append([f1])

    ani = animation.ArtistAnimation(fig, frames, interval=delay, blit=True)

    if not save_path:
        plt.show()
    else:
        ani.save(save_path)

def viz_imgseq_goal(image_seq, goal, unnormalize=True, delay=100, save_path=None):
    print(image_seq.shape)
    N = image_seq.shape[0]

    fig = plt.figure()
    frames = []
    for i in range(N):
        img = image_seq[i]

        if unnormalize:
            img = unnormalize_image(img)

        f1 = plt.imshow(img)
        f2 = plt.scatter(goal[0], goal[1], color='y', marker='x', s=75)

        frames.append([f1, f2])

    ani = animation.ArtistAnimation(fig, frames, interval=delay, blit=True)

    if not save_path:
        plt.show()
    else:
        ani.save(save_path)

def viz_keypoints(img_seq, keyp_seq, unnormalize=True, delay=100, save_path=None, annotate=False):
    """
    Args:
      image_seq: seq_length * H * W * 3 (image normalized (-0.5, 0.5))
      keyp_seq: seq_length * num_keypoints * 3
    """
    print(img_seq.shape, keyp_seq.shape)
    n = img_seq.shape[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    def animate(i):
        img = img_seq[i]
        if unnormalize: img = unnormalize_image(img)
        keypoints = keyp_seq[i]

        keypoints, mu = project_keyp(keypoints)

        ax1.clear()

        f1 = ax1.imshow(img)
        f2 = ax1.scatter(keypoints[:, 0], keypoints[:, 1], c=mu, cmap='Reds')

        f3 = []
        if annotate:
            num_keyp = keypoints.shape[0]
            for i in range(num_keyp):
                f3.append(ax1.annotate(str(i), keypoints[i]))

        ax1.set_title("Input Img and Keypoints")

        return [f1] + [f2] + f3

    ani = animation.FuncAnimation(fig, animate, frames=n, interval=delay, blit=True)

    if save_path:
        ani.save(save_path)
    else:
        plt.show()


def viz_keyp_hmap(img_seq, keyp_seq, hmap_seq,k, unnormalize=True, delay=100, save_path=None):
    n = img_seq.shape[0]
    print(hmap_seq.shape)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    def animate(i):
        img = img_seq[i]
        img = unnormalize_image(img)
        keypoints = keyp_seq[i]

        keypoints, mu = project_keyp(keypoints)

        ax1.clear()
        ax2.clear()

        f1 = ax1.imshow(img)
        f2 = ax1.scatter([keypoints[k, 0]], [keypoints[k, 1]], color='r', s=10)

        f3 = ax2.imshow(hmap_seq[i, :, :, k], cmap='coolwarm')

        ax1.set_title("mu={:.4f}".format(mu[k]))

        ax2.set_title("{} hmap. Max = {:.4f}".format(k, hmap_seq[i, :, :, k].max()))

        return [f1, f2, f3]

    ani = animation.FuncAnimation(fig, animate, frames=n, interval=delay, blit=True)

    if save_path:
        ani.save(save_path)
    else:
        plt.show()

def viz_hmap_all(img_seq, pred_img_seq, keyp_seq, hmap_seq ,
             unnormalize=True, delay=100, save_path=None):

    print(img_seq.shape, pred_img_seq.shape, keyp_seq.shape)
    print("Loss Seq: ", np.sum(np.square(img_seq - pred_img_seq))/(img_seq.shape[0]))
    n = img_seq.shape[0]

    fig = plt.figure()
    #plt.subplots_adjust(hspace=1.0)
    fig.tight_layout()


    top_5_idx = [10, 23, 40, 61, 45]
    num = len(top_5_idx)
    gs = fig.add_gridspec(num, 2)
    axes = []
    for j in range(num):
        axes.append([fig.add_subplot(gs[j, 0]), fig.add_subplot(gs[j, 1])])


    def animate(i):
        img, pred_img = img_seq[i], pred_img_seq[i]
        img, pred_img = unnormalize_image(img), unnormalize_image(pred_img)
        keypoints = keyp_seq[i]

        keypoints, mu = project_keyp(keypoints)

        #print(keypoints, '\n')
        frames = []
        for j in range(num):
            k = top_5_idx[j]
            ax1, ax2 = axes[j]

            ax1.clear()
            f1 = ax1.imshow(img)

            hmax = hmap_seq[i, :, :, k].max()
            print(k, mu[k], hmax , (keypoints[k,0], keypoints[k,1]))
            f2 = ax1.scatter([keypoints[k, 0]], [keypoints[k, 1]], s=10, color='r')
            f3 = ax2.imshow(hmap_seq[i, :, :, k], cmap='coolwarm')

            ax1.set_title("mu={:.4f}".format(mu[k]), fontdict={'fontsize': 'small'})
            ax2.set_title("{} hmap. Max = {:.4f}".format(k, hmax), fontdict={'fontsize': 'small'})

            frames.extend([f1, f2, f3])
        print()
        return frames

    ani = animation.FuncAnimation(fig, animate, frames=n, interval=delay, blit=True)

    if save_path:
        ani.save(save_path)
    else:
        plt.show()


def viz_dynamic_img(img_seq, pred_img_seq, keyp_seq, hmap_seq , k,
             unnormalize=True, delay=100, save_path=None):

    print(img_seq.shape, pred_img_seq.shape, keyp_seq.shape)
    print("Loss Seq: ", np.sum(np.square(img_seq - pred_img_seq))/(img_seq.shape[0]))
    n = img_seq.shape[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2)

    def animate(i):
        img, pred_img = img_seq[i], pred_img_seq[i]
        img, pred_img = unnormalize_image(img), unnormalize_image(pred_img)
        keypoints = keyp_seq[i]

        keypoints, mu = project_keyp(keypoints)

        #print(keypoints, '\n')

        f1 = ax1.imshow(img)

        print(np.argsort(-mu)[:10])

        f2 = ax1.scatter([keypoints[k, 0]], [keypoints[k, 1]], color='r', alpha=mu[k])

        f4 = ax3.imshow(hmap_seq[i, :, :, k], cmap='coolwarm')

        ax1.set_title("mu={:.4f}".format(mu[k]))
        ax3.set_title("{} hmap. Max = {:.4f}".format(k, hmap_seq[i, :, :, k].max()))

        return [f1] + [f2] + [f4]

    ani = animation.FuncAnimation(fig, animate, frames=n, interval=delay, blit=True)

    if save_path:
        ani.save(save_path)
    else:
        plt.show()

def viz_dynamic_img_top(img_seq, pred_img_seq, keyp_seq,save_path=None):

    print(img_seq.shape, pred_img_seq.shape, keyp_seq.shape)
    print("Loss Seq: ", np.sum(np.square(img_seq - pred_img_seq))/(img_seq.shape[0]))
    n = img_seq.shape[0]

    fig = plt.figure(figsize=(20,20))
    plt.subplots_adjust(hspace=0.5)
    fig.tight_layout()

    top_5_idx = None
    gs = fig.add_gridspec(3, 3)
    axes = []
    for i in range(3):
        res = [fig.add_subplot(gs[i, j]) for j in range(3)]
        axes.append(res)

    for i in range(n):
        img, pred_img = img_seq[i], pred_img_seq[i]
        img, pred_img = unnormalize_image(img), unnormalize_image(pred_img)
        keypoints = keyp_seq[i]

        keypoints, mu = project_keyp(keypoints)

        print(np.argsort(-mu)[:10])
        if i == 0:
            top_5_idx = list(np.argsort(-mu)[:10])

        for idx, k in enumerate(top_5_idx):
            ax1 = axes[idx//3][idx%3]

            if i == 0: ax1.imshow(img)

            ax1.scatter([keypoints[k, 0]], [keypoints[k, 1]], color='r', alpha=mu[k])
            ax1.set_title("Channel={}".format(k))

    plt.savefig(save_path)


def viz_dynamic_img_top_vid(img_seq, pred_img_seq, keyp_seq, top_9_idx = None, delay=100, save_path=None):

    print(img_seq.shape, pred_img_seq.shape, keyp_seq.shape)
    print("Loss Seq: ", np.sum(np.square(img_seq - pred_img_seq))/(img_seq.shape[0]))
    n = img_seq.shape[0]

    fig = plt.figure(figsize=(20,20))
    plt.subplots_adjust(hspace=0.1)
    fig.tight_layout()

    gs = fig.add_gridspec(3, 3)
    axes = []
    for i in range(3):
        res = [fig.add_subplot(gs[i, j]) for j in range(3)]
        axes.append(res)

    frames = []
    def animate(i):
        nonlocal  top_9_idx
        img, pred_img = img_seq[i], pred_img_seq[i]
        img, pred_img = unnormalize_image(img), unnormalize_image(pred_img)
        keypoints = keyp_seq[i]

        keypoints, mu = project_keyp(keypoints)

        print(i, np.argsort(-mu)[:10])
        if i == 0 and top_9_idx is None:
            top_9_idx = list(np.argsort(-mu)[:9])

        for idx, k in enumerate(top_9_idx):
            ax1 = axes[idx//3][idx%3]

            ax1.clear()

            ax1.imshow(img)
            f = ax1.scatter([keypoints[k, 0]], [keypoints[k, 1]], s=50.0, color='r') #alpha=mu[k])
            ax1.set_title("Channel={}, mu={:.4f}".format(k, mu[k]))

            frames.append(f)

        return frames

    ani = animation.FuncAnimation(fig, animate, frames=n, interval=delay, blit=True)

    ani.save(save_path)

def viz_hmap(img_seq, pred_img_seq, keyp_seq, hmap_seq ,
             k, resp=None,unnormalize=True, delay=100, save_path=None):
    print(img_seq.shape, pred_img_seq.shape, keyp_seq.shape)
    print("Loss Seq: ", np.sum(np.square(img_seq - pred_img_seq))/(img_seq.shape[0]))
    n = img_seq.shape[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    def animate(i):
        img, pred_img = img_seq[i], pred_img_seq[i]
        img, pred_img = unnormalize_image(img), unnormalize_image(pred_img)
        keypoints = keyp_seq[i]

        keypoints, mu = project_keyp(keypoints)

        ax1.clear()
        ax2.clear()
        ax3.clear()

        f1 = ax1.imshow(img)
        f2 = ax1.scatter([keypoints[k, 0]], [keypoints[k, 1]], color='r', s=10)

        f3 = ax2.imshow(pred_img)
        f4 = ax3.imshow(hmap_seq[i, :, :, k], cmap='coolwarm')

        if resp: ax1.set_title("mu={:.4f}, R={:.4f},{}".format(mu[k], resp[0], resp[1]))
        else: ax1.set_title("mu={:.4f}".format(mu[k]))

        ax2.set_title("Recon Img")
        ax3.set_title("{} hmap. Max = {:.4f}".format(k, hmap_seq[i, :, :, k].max()))

        return [f1, f2, f3, f4]

    ani = animation.FuncAnimation(fig, animate, frames=n, interval=delay, blit=True)

    if save_path:
        ani.save(save_path)
    else:
        plt.show()

def viz_track(img_seq, pred_img_seq, keyp_seq, unnormalize=False, delay=100, save_path=None, annotate=False):
    print(img_seq.shape, pred_img_seq.shape, keyp_seq.shape)
    print("Loss Seq: ", np.sum(np.square(img_seq - pred_img_seq))/(img_seq.shape[0]))
    n = img_seq.shape[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    def animate(i):
        img, pred_img = img_seq[i], pred_img_seq[i]
        img, pred_img = unnormalize_image(img), unnormalize_image(pred_img)
        keypoints = keyp_seq[i]

        keypoints, mu = project_keyp(keypoints)

        ax1.clear()

        f1 = ax1.imshow(img)
        f2 = ax1.scatter(keypoints[:, 0], keypoints[:, 1], c=mu, cmap='Greens')

        f4 = []
        if annotate:
            num_keyp = keypoints.shape[0]
            for i in range(num_keyp):
                f4.append(ax1.annotate(str(i), keypoints[i]))

        # colors = np.zeros((len(keypoints), 4))
        # colors[:, 0] = 1.0
        # colors[:, 3] = mu
        #f2 = ax1.scatter(keypoints[:, 0], keypoints[:, 1], color=colors)

        f3 = ax2.imshow(pred_img)

        ax1.set_title("Input Img and Keypoints")
        ax2.set_title("Reconstructeed Img")

        return [f1] + [f2] + [f3] + f4

    ani = animation.FuncAnimation(fig, animate, frames=n, interval=delay, blit=True)

    if save_path:
        ani.save(save_path)
    else:
        plt.show()

def viz_all(img_seq, pred_img_seq, keyp_seq, unnormalize=False, delay=100, save_path=None):
    print(img_seq.shape, pred_img_seq.shape, keyp_seq.shape)
    print("Loss Seq: ", np.sum(np.square(img_seq - pred_img_seq))/(img_seq.shape[0]))
    n = img_seq.shape[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    def animate(i):
        img, pred_img = img_seq[i], pred_img_seq[i]
        img, pred_img = unnormalize_image(img), unnormalize_image(pred_img)
        keypoints = keyp_seq[i]

        keypoints, mu = project_keyp(keypoints)

        ax1.clear()

        f1 = ax1.imshow(img)
        # f2s = []
        # for k in range(len(keypoints)):
        #     mu = keypoints[k, 2]
        #     f2 = ax1.scatter(keypoints[k, 0], keypoints[k, 1], c='r', alpha=mu)
        #     f2s.append(f2)

        f2 = ax1.scatter(keypoints[:,0], keypoints[:,1], c=mu, cmap='coolwarm')

        f3 = ax2.imshow(pred_img)

        ax1.set_title("Input Img and Keypoints")
        ax2.set_title("Reconstructeed Img")

        return [f1] + [f2] + [f3]

    ani = animation.FuncAnimation(fig, animate, frames=n, interval=delay, blit=True)

    if save_path:
        ani.save(save_path)
    else:
        plt.show()

def viz_all_unroll(img_seq, pred_img_seq, keyp_seq, unnormalize=False, delay=100, save_path=None):
    T = img_seq.shape[0]
    T_obs = T//2
    T_future = pred_img_seq.shape[0] - T_obs
    print(img_seq.shape, pred_img_seq.shape, keyp_seq.shape, 'T_obs: ', T_obs, 'T_future:', T_future)

    error = np.sum(np.square(img_seq - pred_img_seq[:T])/T)
    print("Loss Seq: ", error)

    n = pred_img_seq.shape[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    def animate(i):
        if i < T:
            img, pred_img = img_seq[i], pred_img_seq[i]
        else:
            img, pred_img = img_seq[-1], pred_img_seq[i]

        img, pred_img = unnormalize_image(img), unnormalize_image(pred_img)
        keypoints = keyp_seq[i]

        keypoints = project_keyp(keypoints)

        ax1.clear()
        ax2.clear()

        f1 = ax1.imshow(img)
        # f2s = []
        # for k in range(len(keypoints)):
        #     mu = keypoints[k, 2]
        #     f2 = ax1.scatter(keypoints[k, 0], keypoints[k, 1], c='r', alpha=mu)
        #     f2s.append(f2)
        f2 = ax2.scatter(keypoints[:,0], keypoints[:,1], c='r')

        f3 = ax2.imshow(pred_img)

        if i < T_obs:
            ax1.set_title("OBS: Input Img and Keypoints: t={}".format(i))
            ax2.set_title("OBS: Recon Img: t={}".format(i))
        elif T_obs <= i < T:
            ax1.set_title("PRED: Input Img and Keypoints: t={}".format(i))
            ax2.set_title("PRED: Recon Img: t={}".format(i))
        else:
            ax1.set_title("FUTURE: Input: t={}".format(T-1))
            ax2.set_title("FUTURE: Future pred Img: t={}".format(i))

        return [f1] + [f2] + [f3]

    ani = animation.FuncAnimation(fig, animate, frames=n, interval=delay, blit=True)

    if save_path:
        ani.save(save_path)
    else:
        plt.show()

def viz_keyp_action_pred(img_seq, recon_img_seq, keyp_seq, pred_img_seq, pred_keyp_seq, delay=100, save_path=None):

    T = img_seq.shape[0]
    print(img_seq.shape, pred_img_seq.shape, keyp_seq.shape)

    recon_error = np.sum(np.square(img_seq - recon_img_seq))/T
    pred_recon_error = np.sum(np.square(img_seq[1:]-pred_img_seq))/(T-1)
    print("Recon Loss: ", recon_error, "Pred Recon Loss: ", pred_recon_error)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    def animate(i):
        if i == 0:
            img, recon_img, pred_img = img_seq[i], recon_img_seq[i], recon_img_seq[i]
            keypoints, pred_keypoints = keyp_seq[i], keyp_seq[i]
        else:
            img, recon_img, pred_img = img_seq[i], recon_img_seq[i], pred_img_seq[i-1]
            keypoints, pred_keypoints = keyp_seq[i], pred_keyp_seq[i-1]

        img, recon_img, pred_img = unnormalize_image(img), unnormalize_image(recon_img), unnormalize_image(pred_img)

        keypoints, mu = project_keyp(keypoints)
        pred_keypoints, _ = project_keyp(pred_keypoints)

        ax1.clear()
        ax2.clear()
        ax3.clear()

        f1, f2, f3 = ax1.imshow(img), ax2.imshow(recon_img), ax3.imshow(pred_img)
        f4 = ax1.scatter(keypoints[:,0], keypoints[:,1], c=mu, cmap='Reds')
        f5 = ax1.scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], c=mu, cmap='Greens')

        ax1.set_title("Input: t={}".format(i+1))
        ax2.set_title("Recon: t={}".format(i+1))
        ax3.set_title("Pred:  t={}".format(i+1))

        return [f1, f2, f3, f4, f5]

    ani = animation.FuncAnimation(fig, animate, frames=T, interval=delay, blit=True)

    if save_path:
        ani.save(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # f = np.load("data/acrobot/orig/acrobot_swingup_random_repeat40_00006887be28ecb8.npz")
    # # f = np.load("data/acrobot/train_25/acrobot_swingup_random_repeat40_train_25.npz")
    # img_seq = f['image']
    # viz_imgseq(img_seq)

    import utils
    utils.set_seed_everywhere(5)
    #
    # import datasets
    # d, s = datasets.get_sequence_dataset("data/acrobot_big/train", 32, 32, shuffle=True)
    # data = next(iter(d))

    # print(data['frame_ind'][0])
    # viz_imgseq(data['image'][0].permute(0,2,3,1).numpy(), delay=50, unnormalize=True)

    #data = np.load('test_data.npz')
    #data = np.load('data/fetch_push_25hz/orig/fetch_push_1.npz')
    #data = np.load('data/fetch_pick/orig/fetch_pick_1.npz', allow_pickle=True)
    #data = np.load('data/goal/fetch_pick_sep/fetch_pick_goal_5.npz', allow_pickle=True)
    #data = np.load('data/goal/fetch_reach_sep/fetch_reach_goal_0.npz', allow_pickle=True)
    data = np.load('data/bair_push/orig/traj_9662_to_9917_30.npz')
    #data = np.load('data/fetch_reach/orig/fetch_reach_1.npz', allow_pickle=True)
    #data = np.load('data/bair_push/orig/traj_9662_to_9917.tfrecords_5.npz')
    #img_seq = data['img'][2]
    #img_seq = data['image']
    #img_seq = data['image'][2]
    img_seq = data['image']
    viz_imgseq(img_seq, delay=100, unnormalize=False)