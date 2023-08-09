import os
import numpy as np
import pandas as pd
import torch
import random
import albumentations as A
import pickle

from torch.optim import lr_scheduler

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from albumentations.pytorch.transforms import ToTensorV2
from options.Config import config
from utils.wandb_utils import WBLogger

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

    # ======= Data validation =======
    all_df = pd.read_csv(dst_dict['train'])
    train_df, val_df = train_test_split(all_df, test_size=0.2, random_state=41)

    # ======= Label-Encoding =======
    le = preprocessing.LabelEncoder()
    le.fit(train_df['cat3'].values)

    with open('./data/le.pickle', 'wb') as fw:
        pickle.dump(le, fw)

    train_df['cat3'] = le.transform(train_df['cat3'].values)
    val_df['cat3'] = le.transform(val_df['cat3'].values)

    # ======= Vectorizer =======
    vectorizer = CountVectorizer(max_features=4096)

    train_vectors = vectorizer.fit_transform(train_df['overview'])
    train_vectors = train_vectors.todense()

    val_vectors = vectorizer.transform(val_df['overview'])
    val_vectors = val_vectors.todense()

    # ======= Data augmentation =======
    train_transform = A.Compose([
        A.Resize(dst_dict['width'], dst_dict['height']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                    p=1.0),
        ToTensorV2()
    ])

    obj_tr = dts(train_df['img_path'].values, train_vectors, train_df['cat3'].values, train_transform, train=opt.train)
    obj_val = dts(val_df['img_path'].values, val_vectors, val_df['cat3'].values, train_transform, train=opt.train)

    train_loader = torch.utils.data.DataLoader(obj_tr, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    print('Train Loader Completed')
    test_loader = torch.utils.data.DataLoader(obj_val, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    print('validation Loader Completed')

    return train_loader, test_loader, len(le.classes_)

# ======= VALIDATION CALCULATE LOSS & SCORE =======
def validation_score(net, criterion, val_loader, device):
    net.eval()

    net_preds = []
    true_labels = []

    val_loss = []

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            val_dict = dict()

            val_dict['img'] = data['img'].to(device)
            val_dict['text'] = data['text'].to(device)
            val_dict['label'] = data['label'].to(device)

            pred = net.forward(val_dict['img'], val_dict['text'])

            loss = criterion(pred, val_dict['label'].type(torch.cuda.LongTensor))

            val_loss.append(loss.item())

            net_preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += val_dict['label'].detach().cpu().numpy().tolist()
            #
            batch_step = batch_idx * options.opt.batch_size
            if batch_step % options.opt.freq_show_img == 0:
                wb_logger.log_tour_img_to_wandb(
                    'validation', val_dict['img'], val_dict['label'],
                    pred.argmax(1).detach().cpu().numpy().tolist(),
                    batch_idx, options.opt)
            #
            if batch_step % options.opt.freq_show_batch == 0:
                print(f'## Test Rate : [{batch_step} / {round(16986 * 0.2)}] ##')

    test_score = f1_score(true_labels, net_preds, average='weighted')
    return np.mean(val_loss), test_score

if __name__ == "__main__":
    options = config()
    options.print_options()
    wb_logger = WBLogger(options.opt)
    device = torch.device("cuda:{}".format(options.opt.gpuid) if torch.cuda.is_available() else "cpu")

    #
    print('======= LOAD DATASET =======')
    train_loader, test_loader, n_class = setup_dst(options.opt)

    #
    print('======= SETUP NETWORK =======')
    from model.basic import CustomModel
    net = CustomModel(n_class).to(device)
    if not options.opt.continue_train:
       net.init_weights()
    else:
        net = net.load_weights(net, device, options.opt.save_path)

    #
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=options.opt.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=options.opt.epochs, eta_min=options.opt.lr_min)

    #
    print('======= START TRAIN =======')
    loss_dict = dict()
    for epoch in range(1, options.opt.epochs + 1):
        net.train()
        loss_dict['wb_loss'] = []
        loss_dict['tr_loss'] = []
        print('------- epoch {} -------'.format(epoch))
        for batch_idx, data in enumerate(train_loader):
            #
            input_dict = dict()

            input_dict['img'] = data['img'].to(device)
            input_dict['text'] = data['text'].to(device)
            input_dict['label'] = data['label'].to(device)

            pred = net.forward(input_dict['img'], input_dict['text'])

            loss = criterion(pred, input_dict['label'].type(torch.cuda.LongTensor))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_dict['wb_loss'].append(loss.item())
            loss_dict['tr_loss'].append(loss.item())

            batch_step = batch_idx * options.opt.batch_size
            #
            if batch_step % options.opt.freq_show_img == 0:
                wb_logger.log_tour_img_to_wandb(
                    'train', input_dict['img'], input_dict['label'],
                    pred.argmax(1).detach().cpu().numpy().tolist(),
                    batch_idx, options.opt)
            #
            if batch_step % options.opt.freq_show_loss == 0:
                wb_logger.log('train', metrics_dict=loss_dict)
                loss_dict['wb_loss'] = []
            #
            if batch_step % options.opt.freq_show_batch == 0:
                print(f'## Train Rate : [{batch_step} / {round(16986*0.8)}] ##')

        tr_loss = np.mean(loss_dict['tr_loss'])
        val_loss, val_score = validation_score(net, criterion, test_loader, device)

        print(f'Epoch : [{epoch}], Train Loss : [{tr_loss:.5f}], Val loss : [{val_loss:.5f}], Val Score : [{val_score:.5f}]')
        #
        print('Update Learning Rate')
        scheduler.step()
        print('learning rate = %.7f' % optimizer.param_groups[0]['lr'])
        #
        if epoch % 2 == 0:
            save_file = options.opt.save_path + '/basic_{}.pth'.format(epoch)
            torch.save({'net':net.state_dict()}, save_file)
            print('Save networks... File_path : ' + str(save_file))