import argparse

import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from tensorflow.python.platform import gfile

import os

import numpy as np
from PIL import Image


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='', help='base directory to save processed data')
# opt = parser.parse_args()

FRAMES_PER_VIDEO = 30
IMG_SHAPE = (64, 64, 3)

def get_seq(dname):
    data_dir = '%s/%s' % ("data", dname)

    filenames = gfile.Glob(os.path.join(data_dir, '*'))
    if not filenames:
        raise RuntimeError('No data files found.')

    for f in filenames:
        k = 0
        for serialized_example in tf.python_io.tf_record_iterator(f):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            image_seq = []
            for i in range(30):
                image_name = str(i) + '/image_aux1/encoded'
                byte_str = example.features.feature[image_name].bytes_list.value[0]
                # img = Image.open(io.BytesIO(byte_str))
                img = Image.frombytes('RGB', (64, 64), byte_str)
                arr = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
                image_seq.append(arr.reshape(1, 64, 64, 3))
            image_seq = np.concatenate(image_seq, axis=0)
            k = k + 1
            yield f, k, image_seq


def generate_examples(filedir):
    files = tf.gfile.Glob(os.path.join(filedir, "*.tfrecords"))
    # For each file
    file_id = 0
    for filepath in sorted(files):
        # For each video inside the file
        for video_id, example_str in enumerate(
            tf.io.tf_record_iterator(filepath)):
            example = tf.train.SequenceExample.FromString(example_str)

            # Merge all frames together
            all_frames = {'image': [],
                          'action': [],
                          'endeffector_pos': [],
                          'frame_ind': [],
                          'file_idx': []}

            for frame_id in range(FRAMES_PER_VIDEO):
                # Extract all features from the original proto context field
                frame_feature = {  # pylint: disable=g-complex-comprehension
                    out_key: example.context.feature[in_key.format(frame_id)]  # pylint: disable=g-complex-comprehension
                    for out_key, in_key in [
                        #("image_main", "{}/image_main/encoded"),
                        ("image", "{}/image_aux1/encoded"),
                        ("endeffector_pos", "{}/endeffector_pos"),
                        ("action", "{}/action"),
                    ]
                }

                all_frames['frame_ind'].append(frame_id)
                all_frames['file_idx'].append(file_id)
                # Decode float
                for key in ("endeffector_pos", "action"):
                    values = frame_feature[key].float_list.value
                    frame_feature[key] = [values[i] for i in range(len(values))]
                    all_frames[key].append(frame_feature[key])

                # Decode images (from encoded string)
                for key in ["image"]:
                    img = frame_feature[key].bytes_list.value[0]
                    img = np.frombuffer(img, dtype=np.uint8)
                    img = np.reshape(img, IMG_SHAPE)
                    frame_feature[key] = img

                    all_frames[key].append(frame_feature[key])

            for key in all_frames:
                all_frames[key] = np.stack(all_frames[key])

            yield "%s_%s" % (filepath.split(".")[0], file_id), all_frames

            file_id += 1

def save_seq(args):
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)

    gen = generate_examples(args.data_dir)

    for f, data_seq in gen:
        f_save = f.split("/")[-1] + ".npz"
        np.savez(os.path.join(args.save_dir, f_save), **data_seq)
        print(f_save)

#gen = get_seq("bair_push")

#print(tf.gfile.Glob("data/bair_push/*.tfrecords"))
#save_seq("data/bair_push/raw_test", "data/bair_push/test")

def play_video():
    gen = generate_examples("data/bair_push/raw")
    _, d = next(gen)
    img_seq = d['image']
    print(img_seq.shape, d['action'].shape, d['endeffector_pos'].shape)
    from  visualizer import viz_imgseq
    viz_imgseq(img_seq, unnormalize=False)

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='bair')
    parser.add_argument('--task_name', default='push')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument("--data_dir", default='data/bair_push/raw')
    parser.add_argument("--save_dir", default='data/bair_push/orig')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    save_seq(args)
