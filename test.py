import os
import numpy as np
import pandas as pd
import torch
import random
import albumentations as A
import pickle

from sklearn.feature_extraction.text import CountVectorizer

from albumentations.pytorch.transforms import ToTensorV2
from options.Config import config

# ======= Fixed RandomSeed =======
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(41) # Seed 고정

# ======= SETUP DATASET =======
def setup_dst(opt):
    from data_loader.tour_data_load import tour_data_load as dts
    dst_dict = opt.dataset_dict

    test_df = pd.read_csv(dst_dict['test'])

    # ======= Vectorizer =======
    vectorizer = CountVectorizer(max_features=4096)

    test_vectors = vectorizer.fit_transform(test_df['overview'])
    test_vectors = test_vectors.todense()

    # ======= Data augmentation =======
    test_transform = A.Compose([
        A.Resize(dst_dict['width'], dst_dict['height']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                    p=1.0),
        ToTensorV2()
    ])

    obj_te = dts(test_df['img_path'].values, test_vectors, None, test_transform, opt.train)

    test_loader = torch.utils.data.DataLoader(obj_te, batch_size=opt.batch_size, num_workers=0)
    print('test Loader Completed')

    return test_loader

if __name__ == "__main__":
    options = config()
    options.print_options()
    device = torch.device("cuda:{}".format(options.opt.gpuid) if torch.cuda.is_available() else "cpu")

    #
    print('======= LOAD DATASET =======')
    test_loader = setup_dst(options.opt)

    #
    print('======= SETUP NETWORK =======')
    from model.basic import CustomModel
    net = CustomModel(128).to(device)
    net = net.load_weights(net, device, 'basic_30.pth', options.opt.save_path)

    #
    print('======= START TEST =======')
    net.eval()

    net_preds = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            #
            input_dict = dict()

            input_dict['img'] = data['img'].to(device)
            input_dict['text'] = data['text'].to(device)

            pred = net.forward(input_dict['img'], input_dict['text'])
            net_preds += pred.argmax(1).detach().cpu().numpy().tolist()

            #
            batch_step = batch_idx * options.opt.batch_size
            if batch_step % options.opt.freq_show_batch == 0:
                print(f'## Test Rate : [{batch_step} / {round(7280)}] ##')

    submit = pd.read_csv(options.opt.sample_submit_path)

    with open('./data/le.pickle', 'rb') as fw:
        le = pickle.load(fw)

    submit['cat3'] = le.inverse_transform(net_preds)

    submit.to_csv('/storage/jhchoi/tour/open/submit_basic_30.csv', index=False)
    print('Complete making submit file !!')