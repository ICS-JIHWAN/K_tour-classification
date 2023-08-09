import argparse
import os

# ======= DATASET INFORMATION =======
DATASET_DICT = dict()
DATASET_DICT['train'] = '/storage/jhchoi/tour/open/train.csv'
DATASET_DICT['train_img'] = '/storage/jhchoi/tour/open/image/train/'
DATASET_DICT['test'] = '/storage/jhchoi/tour/open/test.csv'
DATASET_DICT['test_img'] = '/storage/jhchoi/tour/open/image/test/'
DATASET_DICT['width'] = 128
DATASET_DICT['height'] = 128
# ========== PATH INFORMATION ==============
SAMPLE_SUBMIT_PATH = '/storage/jhchoi/tour/open/sample_submission.csv'
SAVE_PATH = './checkpoints'
# ==========================================

class config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        #####
        self.parser.add_argument('--epochs', type=int, default=30)
        self.parser.add_argument('--batch_size', type=int, default=50)
        self.parser.add_argument('--train', type=bool, default=False)
        self.parser.add_argument('--continue_train', type=bool, default=False)
        #####
        self.parser.add_argument('--dataset_dict', type=str, default=DATASET_DICT)
        #####
        self.parser.add_argument('--gpuid', type=str, default='1')
        self.parser.add_argument('--lr', type=float, default=3e-4)
        self.parser.add_argument('--lr_min', type=float, default=3e-6)
        #####
        self.parser.add_argument('--freq_show_batch', type=int, default=100)
        self.parser.add_argument('--freq_show_img', type=int, default=1000)
        self.parser.add_argument('--freq_show_loss', type=int, default=100)
        self.parser.add_argument('--save_path', type=str, default=SAVE_PATH)
        #####
        self.parser.add_argument('--sample_submit_path', type=str, default=SAMPLE_SUBMIT_PATH)

        self.opt, _ = self.parser.parse_known_args()
    def print_options(self):
        """Print and save options
                It will print both current options and default values(if different).
                It will save options into a text file / [checkpoints_dir] / opt.txt
                """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)