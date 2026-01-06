import argparse
import glob
import os
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def run(src, dst, test_size=0.1):
    try:
        for image_path in glob.glob(os.path.join(src, '**', '*.jpeg'), recursive=True):
            os.rename(image_path, os.path.splitext(image_path)[0] + '.jpg')
    except:
        pass
    for folder in os.listdir(src):
        dst_train = os.path.join(dst, 'train')
        dst_test = os.path.join(dst, 'val')
        makedirs(os.path.join(dst_train, folder))
        makedirs(os.path.join(dst_test, folder))
        x_train, x_test = train_test_split(glob.glob(os.path.join(src, folder, '**', '*.jpg'), recursive=True), test_size=test_size)
        for img in tqdm(x_train, desc=f'Split {folder} to train'):
            shutil.copy(os.path.join(src, folder, img), os.path.join(dst_train, folder))
        for img in tqdm(x_test, desc=f'Split {folder} to val'):
            shutil.copy(os.path.join(src, folder, img), os.path.join(dst_test, folder))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--test-size', type=float, default=0.1)
    args = parser.parse_args()
    return args

def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)