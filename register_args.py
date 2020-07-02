import argparse
import json, os


def get_argparse(force_exp_name=True):
    parser = argparse.ArgumentParser(description='Unsup Keyp')

    exp_name = None if force_exp_name else "test-run"
    parser.add_argument("--exp_name", type=str, help='Name of the experiment', required=force_exp_name,
                        default=exp_name)
    parser.add_argument("--data_dir", type=str, help='Base data directory', default="data/acrobot_big")
    parser.add_argument("--train_dir", type=str, default="train", help='Train directory under data_dir')
    parser.add_argument("--test_dir", type=str, default="test", help='Test directory under data_dir')
    parser.add_argument("--base_dir", type=str, default="exp_data",
                        help='Base directory containing data for an experiment')

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--pretrained_path", type=str, default=None, help='Load pretrained model from this path')
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--seed", type=int, default=0, help='Random seeed for a run')
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument("--num_keypoints", type=int, default=64, help="Number of keypoints to encode")
    parser.add_argument("--timesteps", type=int, default=8, help="Number of observed steps")
    parser.add_argument("--action_dim", type=int, default=4)

    parser.add_argument("--heatmap_reg", type=float, default=0.1, help="Coeff for L1 loss on heatmap")
    parser.add_argument("--clipnorm", type=float, default=10.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
    parser.add_argument("--temp_reg", type=float, default=1.0, help="temporal loss scale")
    parser.add_argument("--temp_width", type=float, default=0.01, help="temporal separation loss width")
    parser.add_argument("--kl_reg", type=float, default=0.003, help="temporal separation loss width")
    parser.add_argument("--keyp_reg", type=float, default=1.0)
    parser.add_argument("--action_reg", type=float, default=1.0)

    parser.add_argument("--num_epochs", type=int, default=50, help="Number of Epochs to train")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of Epochs to train")
    parser.add_argument("--steps_per_epoch", type=int, default=50, help="Number of Steps Per Epoch to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for model")


    parser.add_argument("--log_training", action='store_false')
    parser.add_argument("--log_training_path", default='training_logs')

    parser.add_argument('--no-cuda', action='store_true', help='enables CUDA training')
    parser.add_argument('--is_train', action='store_true')

    parser.add_argument('--vids_dir', type=str, default='vids')
    parser.add_argument('--vids_path', type=str, default='vid')

    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--no_first', action='store_true')
    parser.add_argument('--keyp_pred', action='store_true')
    parser.add_argument('--keyp_inverse', action='store_true')
    parser.add_argument('--unroll', action='store_true')
    parser.add_argument('--annotate', action='store_true')

    return parser


def save_config(cfg, dir_path, filename):
    print("Experiment Configuration:", json.dumps(cfg, indent=4))

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    path = os.path.join(dir_path, filename)
    with open(path, "w") as write_file:
        write_file.write(json.dumps(cfg, indent=4))