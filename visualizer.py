import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

IMG_W, IMG_H = (64, 64)
HMAP_W, HMAP_H = (16, 16)


def unnormalize_image(img):
    return ((img + 0.5) * 255).astype(np.uint8)

def project_keyp(keyp):
    """
    Args:
      keyp: N x 2

    Returns:
      keyp: N x 2
    """
    x, y, mu = keyp[:, 0], keyp[:, 1], keyp[:, 2]
    x, y = x[mu >= 0.5], y[mu >= 0.5]
    x, y = 8 * x, 8 * y
    x, y = x + 8, 8 - y
    x, y = (64 / 16) * x, (64 / 16) * y

    N = x.shape[0]

    #return np.hstack((x.reshape((N, 1)), y.reshape((N, 1)), mu.reshape(N,1)))
    return np.hstack((x.reshape((N, 1)), y.reshape((N, 1))))


def viz_imgseq(image_seq, unnormalize=False, delay=100, save_path=None):
    import scipy.ndimage
    print(image_seq.shape)
    N = image_seq.shape[0]

    fig = plt.figure()
    frames = []
    for i in range(N):
        img = image_seq[i]
        #print(img.shape)

        if unnormalize: img = unnormalize_image(img)

        f1 = plt.imshow(img)
        frames.append([f1])

    ani = animation.ArtistAnimation(fig, frames, interval=delay, blit=True)

    if not save_path:
        plt.show()
    else:
        ani.save(save_path)


def viz_keypoints(image_seq, keyp_seq):
    """
    Args:
      image_seq: seq_length * H * W * 3 (image normalized (-0.5, 0.5))
      keyp_seq: seq_length * num_keypoints * 3
    """
    print(image_seq.shape, keyp_seq.shape)
    n = image_seq.shape[0]

    fig = plt.figure()
    frames = []
    for i in range(n):
        img = image_seq[i]
        img = unnormalize_image(img)
        keypoints = keyp_seq[i]

        keypoints = project_keyp(keypoints)
        # print('x_min: ', keypoints[:, 0].min(), ' x_max: ', keypoints[:, 0].max())
        # print('y_min: ', keypoints[:, 1].min(), ' y_max: ', keypoints[:, 1].max())

        f1 = plt.imshow(img)
        f2 = plt.scatter(keypoints[:, 0], keypoints[:, 1])

        frames.append([f1, f2])

    ani = animation.ArtistAnimation(fig, frames, interval=500, blit=True)

    ani.save('test.mp4')


def viz_all(img_seq, pred_img_seq, keyp_seq, unnormalize=False, delay=100, save_path=None):
    print(img_seq.shape, pred_img_seq.shape, keyp_seq.shape)
    print("Loss Seq: ", np.sum(np.abs(img_seq - pred_img_seq))/(img_seq.shape[0]))
    n = img_seq.shape[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    def animate(i):
        img, pred_img = img_seq[i], pred_img_seq[i]
        img, pred_img = unnormalize_image(img), unnormalize_image(pred_img)
        keypoints = keyp_seq[i]

        keypoints = project_keyp(keypoints)

        ax1.clear()

        f1 = ax1.imshow(img)
        # f2s = []
        # for k in range(len(keypoints)):
        #     mu = keypoints[k, 2]
        #     f2 = ax1.scatter(keypoints[k, 0], keypoints[k, 1], c='r', alpha=mu)
        #     f2s.append(f2)

        f2 = ax1.scatter(keypoints[:,0], keypoints[:,1], c='r')

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


if __name__ == "__main__":
    # f = np.load("data/acrobot/orig/acrobot_swingup_random_repeat40_00006887be28ecb8.npz")
    # # f = np.load("data/acrobot/train_25/acrobot_swingup_random_repeat40_train_25.npz")
    # img_seq = f['image']
    # viz_imgseq(img_seq)

    import datasets
    d, s = datasets.get_sequence_dataset("data/acrobot/train", 1, 50, shuffle=False)
    data = next(iter(d))

    print(data['frame_ind'][0])
    viz_imgseq(data['image'][0].permute(0,2,3,1).numpy(), unnormalize=True)
