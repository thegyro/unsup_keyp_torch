import functools
import glob
import os
import numpy as np

# Data fields used by the model:
REQUIRED_DATA_FIELDS = ['image', 'true_object_pos', 'action']

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, ChainDataset, ConcatDataset

class ImgSequenceDatasetGen(Dataset):
    def __init__(self, filename, num_timesteps, shuffle=True):
        self.filename = filename
        self.num_timesteps = num_timesteps
        self.shuffle = shuffle

        self.data_len = self._find_length(filename, num_timesteps, shuffle)

    def __getitem__(self, idx):
        seq_data_dict = self._process_data(self.filename, self.num_timesteps, self.shuffle)

        img_seq_dict = {k: v[idx] for k, v in seq_data_dict.items()}
        return img_seq_dict

    def _process_data(self, filename, num_timesteps, shuffle=True):
        data_dict = _read_numpy_sequences(filename)
        data_dict.pop('filename')

        seq_data_dict = _chunk_sequence(data_dict, num_timesteps, random_offset=False, shuffle=shuffle)

        for k in seq_data_dict.keys():
            if k == 'image':
                seq_data_dict[k] = torch.from_numpy(seq_data_dict[k]).permute(0, 1, 4, 2, 3)
            else:
                seq_data_dict[k] = torch.from_numpy(seq_data_dict[k])

        return seq_data_dict

    def _find_length(self, filename, num_timesteps, shuffle=True):
        print("Finding length")
        data_dict = self._process_data(filename, num_timesteps, shuffle)
        return data_dict['image'].shape[0]

    def __len__(self):
        return self.data_len


class ImgDataset(Dataset):
    def __init__(self, filename):
        super(ImgDataset, self).__init__()
        self.filename = filename
        self.data_dict = self._process_data(filename)

    def __getitem__(self, idx):
        img_dict = {k: v[idx] for k, v in self.data_dict.items()}
        return img_dict

    def _process_data(self, filename):
        data_dict = _read_numpy_sequences(filename)
        data_dict.pop('filename')

        for k in data_dict.keys():
            if k == 'image':
                data_dict[k] = torch.from_numpy(data_dict[k]).permute(0, 3, 1, 2)
            else:
                data_dict[k] = torch.from_numpy(data_dict[k])

        return data_dict

    def __len__(self):
        return len(self.data_dict['image'])


class ImgSequenceDataset(Dataset):
    def __init__(self, filename, num_timesteps, shuffle=True):
        self.filename = filename
        self.num_timesteps = num_timesteps
        self.seq_data_dict = self._process_data(filename, num_timesteps, shuffle)

    def __getitem__(self, idx):
        img_seq_dict = {k: v[idx] for k, v in self.seq_data_dict.items()}
        return img_seq_dict

    def _process_data(self, filename, num_timesteps, shuffle=True):
        data_dict = _read_numpy_sequences(filename)
        #data_dict.pop('filename')

        seq_data_dict = _chunk_sequence(data_dict, num_timesteps, random_offset=False, shuffle=shuffle)

        for k in seq_data_dict.keys():
            if k == 'image':
                seq_data_dict[k] = torch.from_numpy(seq_data_dict[k]).permute(0, 1, 4, 2, 3)
            else:
                seq_data_dict[k] = torch.from_numpy(seq_data_dict[k])

        return seq_data_dict

    def __len__(self):
        return len(self.seq_data_dict['image'])

def get_dataset(data_dir,
                batch_size,
                file_glob='*.npz',
                shuffle=True,
                num_workers=2,
                seed=0):
    # Find files for dataset. Each file contains a sequence of arbitrary length:
    file_glob = file_glob if '.npz' in file_glob else file_glob + '.npz'
    filenames = sorted(glob.glob(os.path.join(data_dir, file_glob)))
    print("Loading data from: ", data_dir)
    if not filenames:
        raise RuntimeError('No data files match {}.'.format(os.path.join(data_dir, file_glob)))

    # Deterministic in-place shuffle:
    np.random.RandomState(seed).shuffle(filenames)

    # Create dataset:
    dtypes, pre_chunk_shapes = _read_data_types_and_shapes(filenames[0])

    print("Shapes Before: ", pre_chunk_shapes)
    print("Dtypes Before: ", dtypes)

    # dsets = [ImgDatasetIter(fname) for fname in filenames]
    # dataset = ChainDataset(dsets)
    dsets = [ImgDataset(fname) for fname in filenames]
    dataset = ConcatDataset(dsets)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    shapes = {k: (None,) + v.shape[1:] for k, v in iter(dataloader).next().items()}

    print("Shapes After: ", shapes)
    print()

    return dataloader, shapes


def get_sequence_dataset(data_dir,
                         batch_size,
                         num_timesteps,
                         file_glob='*.npz',
                         random_offset=False,
                         shuffle=True,
                         num_workers=4,
                         seed=0):
    # Find files for dataset. Each file contains a sequence of arbitrary length:
    file_glob = file_glob if '.npz' in file_glob else file_glob + '.npz'
    filenames = sorted(glob.glob(os.path.join(data_dir, file_glob)))
    print("Loading data from: ", data_dir)
    if not filenames:
        raise RuntimeError('No data files match {}.'.format(os.path.join(data_dir, file_glob)))

    # Deterministic in-place shuffle:
    np.random.RandomState(seed).shuffle(filenames)

    # Create dataset:
    dtypes, pre_chunk_shapes = _read_data_types_and_shapes(filenames[0])

    print("Shapes Before: ", pre_chunk_shapes)
    print("Dtypes Before: ", dtypes)

    dsets = [ImgSequenceDataset(fname, num_timesteps, shuffle=shuffle) for fname in filenames]
    dataset = ConcatDataset(dsets)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=shuffle)

    shapes = {k: (None,) + v.shape[1:] for k, v in iter(dataloader).next().items()}

    print("Shapes After: ", shapes)
    print()

    return dataloader, shapes


def _read_numpy_sequences(filename):
    try:
        with open(filename, 'rb') as f:
            sequence_dict = {k: v for k, v in np.load(f, allow_pickle=True).items()}
    except IOError as e:
        print('Caught IOError: "{}". Skipping file {}.'.format(e, filename))

    # Format data:
    sequence_dict = _choose_data_fields(sequence_dict)
    sequence_dict = {k: _adjust_precision(v) for k, v in sequence_dict.items()}
    sequence_dict['image'] = _format_image_data(sequence_dict['image'])

    # Add filename and frame arrays for traceability:
    num_frames = list(sequence_dict.values())[0].shape[0]
    sequence_dict['frame_ind'] = np.arange(num_frames, dtype=np.int32)
    sequence_dict['file_idx'] = np.full(num_frames, int(filename.split("_")[-1].split(".")[0]))

    return sequence_dict


def _choose_data_fields(data_dict):
    import scipy.ndimage
    """Returns a new dict containing only fields required by the model."""
    output_dict = {}
    for k in REQUIRED_DATA_FIELDS:
        if k in data_dict:
            output_dict[k] = data_dict[k]
        elif k == 'true_object_pos':
            # Create dummy ground truth if it's not in the dict:
            # print('\nFound no true_object_pos in data, adding dummy.\n')
            num_timesteps = data_dict['image'].shape[0]
            output_dict['true_object_pos'] = np.zeros([num_timesteps, 0, 2])
        else:
            raise ValueError(
                'Required key "{}" is not in the  dict with keys {}.'.format(
                    k, list(data_dict.keys())))
    return output_dict


def _adjust_precision(array):
    """Adjusts precision."""
    if array.dtype == np.float64:
        return array.astype(np.float32)
    if array.dtype == np.int64:
        return array.astype(np.int32)
    return array


def _format_image_data(image):
    """Formats the uint8 input image to float32 in the range [-0.5, 0.5]."""
    if not np.issubdtype(image.dtype, np.uint8):
        raise ValueError('Expected image to be of type {}, but got type {}.'.format(
            np.uint8, image.dtype))
    return image.astype(np.float32) / 255.0 - 0.5


def _read_data_types_and_shapes(filename):
    """Gets dtypes and shapes for all keys in the dataset."""
    sequence = _read_numpy_sequences(filename)
    dtypes = {k: v.dtype for k, v in sequence.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in sequence.items()}
    return dtypes, shapes


def _chunk_sequence(sequence_dict, chunk_length, shuffle=True, random_offset=False):
    length = np.shape(list(sequence_dict.values())[0])[0]

    if random_offset:
        num_chunks = np.maximum(1, length // chunk_length - 1)
        output_length = num_chunks * chunk_length
        max_offset = length - output_length
        offset = np.random.randint(0, max_offset + 1)
    else:
        num_chunks = length // chunk_length
        output_length = num_chunks * chunk_length
        offset = 0

    chunked = {}
    for key, tensor in sequence_dict.items():
        tensor = tensor[offset:offset + output_length]
        chunked_shape = [num_chunks, chunk_length] + list(tensor.shape[1:])
        chunked[key] = np.reshape(tensor, chunked_shape)

    if shuffle:
        chunked = shuffle_dict(chunked, num_chunks)

    return chunked


def shuffle_dict(d, length):
    idx = np.arange(0, length)
    np.random.shuffle(idx)
    s_d = {}
    for key, tensor in d.items():
        s_d[key] = tensor[idx]
    return s_d


if __name__ == "__main__":
    #d, s = get_dataset("data/acrobot/train", 32)
    #d, s = get_sequence_dataset("data/acrobot_big/train", 32, 16, shuffle=False)
    #d, s = get_sequence_dataset("data/fetch_push/train", 32, 16, shuffle=False)
    #d, s = get_sequence_dataset("data/bair_push/orig", 1, 30, shuffle=False)
    #d, s = get_sequence_dataset("data/fetch_push/train", 1, 16, shuffle=True)
    #d, s = get_sequence_dataset("data/fetch_pick/train", 1, 16, shuffle=True)
    #d, s = get_sequence_dataset("data/goal/fetch_pick_sep/", 1, 16, shuffle = True)
    #d, s = get_sequence_dataset("data/fetch_reach_25/train", 32, 16, shuffle=False)
    #d, s = get_sequence_dataset("data/bair_push/orig", 32, 30, shuffle=False)
    d, s = get_sequence_dataset("data/fetch_reach_1/train", 32, 16, shuffle=False)
    # d, s = get_sequence_dataset("data/robot_push/train", 32, 8,
    #                             num_workers=5,
    #                             shuffle=False)

    ind = []
    for (i, a) in enumerate(d):
        print(i, a['image'].shape, a['frame_ind'].shape, a['action'].shape, a['file_idx'].shape)
        #ind.extend(list(a['frame_ind'].numpy().flatten()))

    print(sorted(ind))