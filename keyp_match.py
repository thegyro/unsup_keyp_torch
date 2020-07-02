from scipy.signal import convolve2d
import numpy as np
from annoy import AnnoyIndex

import matplotlib.pyplot as plt

def unnormalize_image(img):
    img = np.clip(img, -0.5, 0.5)
    return ((img + 0.5) * 255).astype(np.uint8)

def project_keyp(keyp):
    x, y, mu = keyp[:, 0], keyp[:, 1], keyp[:, 2]
    #x, y = x[mu >= 0.2], y[mu >= 0.2]
    x, y = 8 * x, 8 * y
    x, y = x + 8, 8 - y
    x, y = (64 / 16) * x, (64 / 16) * y

    N = x.shape[0]

    #return np.hstack((x.reshape((N, 1)), y.reshape((N, 1)))), mu[mu>=0.2]
    return np.hstack((x.reshape((N, 1)), y.reshape((N, 1)))), mu

def rgb2gray(I_rgb):
  r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
  I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return I_gray

def GaussianPDF_1D(mu, sigma, length):
    # create an array
    half_len = length / 2

    if np.remainder(length, 2) == 0:
        ax = np.arange(-half_len, half_len, 1)
    else:
        ax = np.arange(-half_len, half_len + 1, 1)

    ax = ax.reshape([-1, ax.size])
    denominator = sigma * np.sqrt(2 * np.pi)
    nominator = np.exp(-np.square(ax - mu) / (2 * sigma * sigma))

    return nominator / denominator


def GaussianPDF_2D(mu, sigma, row, col):
    # create row vector as 1D Gaussian pdf
    g_row = GaussianPDF_1D(mu, sigma, row)
    # create column vector as 1D Gaussian pdf
    g_col = GaussianPDF_1D(mu, sigma, col).transpose()

    return convolve2d(g_row, g_col, 'full')


def feat_desc(img, x, y):
    # Your Code Here

    H, W = img.shape

    G = GaussianPDF_2D(0, 0.5, 3, 3)
    img = convolve2d(img, G, mode='same')

    N = x.shape[0]

    ii, jj = np.meshgrid(np.arange(-2, 2), np.arange(-2, 2))
    points = np.array(list(zip(ii.flatten(), jj.flatten())))  # 64 x 2
    points_N = np.repeat(points[:, :, np.newaxis], N, axis=2)  # 64 x 2 x N
    points_N = points_N.transpose((2, 1, 0))  # N x 2 x 64

    xy = np.hstack((x.reshape((N, 1)), y.reshape((N, 1))))  # N x 2 x 64
    xy = xy[:, :, np.newaxis]
    points_N = points_N + xy  # N x 2 x 64

    points_N[:, 0, :] = np.clip(points_N[:, 0, :], 0, W - 1)
    points_N[:, 1, :] = np.clip(points_N[:, 1, :], 0, H - 1)

    feats = img[points_N[:, 1, :], points_N[:, 0, :]].copy()  # N x 64

    mean, std = np.mean(feats, axis=0), np.std(feats, axis=0)
    feats -= mean
    feats /= std

    return feats.T

def feat_match(descs1, descs2):
    # Your Code Here
    D, N1 = descs1.shape
    D, N2 = descs2.shape

    annoy_index = AnnoyIndex(D, metric='euclidean')
    for i in range(N2):
        annoy_index.add_item(i, descs2[:, i])

    annoy_index.build(20)

    output_match = np.zeros(N1)

    for i in range(N1):
        v = descs1[:, i]
        uidx, distances = annoy_index.get_nns_by_vector(v, 2, include_distances=True)

        if distances[0] / distances[1] < 0.7:
            output_match[i] = uidx[0]
        else:
            output_match[i] = uidx[0]

    print("Annoy: Total number of matching points: ", np.sum(output_match != -1))

    return output_match

def main():
    from matplotlib.patches import ConnectionPatch

    data = np.load('test_data.npz')

    img_seq_batch = data['img']
    recon_img_seq_batch = data['pred_img']
    keyp_seq_batch = data['keyp']
    action_seq_batch = data['action']

    img_seq = img_seq_batch[0]
    keyp_seq = keyp_seq_batch[0]

    t1, t2 = 0, 5

    img = unnormalize_image(img_seq[t1])
    img2 = unnormalize_image(img_seq[t2])

    np.save("Mosaicing/images/k1.npy", img)
    np.save("Mosaicing/images/k2.npy", img2)

    img_g = rgb2gray(img)
    img_g2 = rgb2gray(img2)

    keyp, mu = project_keyp(keyp_seq[t1])
    x, y = keyp[:, 0].copy(), keyp[:, 1].copy()
    x, y = np.round(x).astype(int), np.round(y).astype(int)
    feats = feat_desc(img_g, x, y)

    keyp2, mu2 = project_keyp(keyp_seq[t2])
    x2, y2 = keyp2[:, 0].copy(), keyp2[:, 1].copy()
    x2, y2 = np.round(x2).astype(int), np.round(y2).astype(int)
    feats2 = feat_desc(img_g2, x2, y2)

    output_match = feat_match(feats, feats2)
    output_match = output_match.squeeze()
    output_match = output_match.astype(int)

    x1 = x[output_match > -1]
    y1 = y[output_match > -1]

    x2 = x2[output_match[output_match > -1]]
    y2 = y2[output_match[output_match > -1]]

    f = plt.figure(figsize=(10, 10))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    ax1.imshow(img)
    ax2.imshow(img2)

    ax1.plot(x1, y1, 'ro', markersize=3)
    ax2.plot(x2, y2, 'ro', markersize=3)

    ax1.scatter(keyp[:,0], keyp[:,1], c=mu, cmap='coolwarm')
    ax2.scatter(keyp2[:,0], keyp2[:,1], c=mu2, cmap='coolwarm')

    for i in range(x1.shape[0]):
        con = ConnectionPatch(xyA=(x2[i], y2[i]), xyB=(x1[i], y1[i]), coordsA="data", coordsB="data", axesA=ax2,
                              axesB=ax1, color="yellow")
        ax2.add_artist(con)

    plt.show()

if __name__ == "__main__":
    main()