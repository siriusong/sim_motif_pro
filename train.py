import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import torch.optim as optim
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from data_preprocessing_warm_start import CustomData
from MM_KG_normal import MotifData
from dataset import load_ddi_dataset
from log.train_logger import TrainLogger
from model import SimMotifPro
import argparse
from metrics import *
from utils import *
from tqdm import  tqdm
import warnings
import json
warnings.filterwarnings("ignore")

def val(model, criterion, dataloader, device,mm_graph):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in dataloader:
        head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]

        with torch.no_grad():

            pred = model((head_pairs, tail_pairs, rel,mm_graph),False)
            loss = criterion(pred, label)
            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy().tolist())
            label_list.append(label.detach().cpu().numpy().tolist())
            running_loss.update(loss.item(), label.size(0))

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)
    epoch_loss = running_loss.get_average()
    running_loss.reset()
    model.train()
    return epoch_loss, acc, auroc, f1_score, precision, recall, ap

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--n_iter', type=int, default=6, help='number of iterations')
    parser.add_argument('--fold', type=int, default=0, help='[0, 1, 2]')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=4e-4, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--save_model', action='store_true',default=True, help='whether save model or not')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    args = parser.parse_args()

    params = dict(
        model='SimMotifPro',
        data_root='data/preprocessed/',
        save_dir='save',
        dataset='drugbank',
        epochs=args.epochs,
        fold=args.fold,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        weight_decay=args.weight_decay
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    save_model = params.get('save_model')
    batch_size = params.get('batch_size')
    data_root = params.get('data_root')
    data_set = params.get('dataset')
    fold = params.get('fold')
    epochs = params.get('epochs')
    n_iter = params.get('n_iter')
    lr = params.get('lr')
    weight_decay = params.get('weight_decay')
    data_path = os.path.join(data_root, data_set)

    train_loader, val_loader, test_loader,motifs,mm_graph = load_ddi_dataset(root=data_path, batch_size=batch_size, fold=fold)
    data = next(iter(train_loader))
    node_dim = data[0]['atom'].x.size(-1)
    device = torch.device('cuda:0')

    model = SimMotifPro(70, 6, n_layer=n_iter).cuda()

    optimizer = optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1-epoch/epochs)

    running_loss = AverageMeter()
    running_acc = AverageMeter()
    # model.load_state_dict(torch.load('/home/user12/mmddi/drugbank/save/20240517_185434_drugbank/model/s1_checkpoints.pt',map_location='cuda'))
    motifs = motifs.to(device)
    mm_graph = mm_graph.to(device)
    val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap = val(model, criterion, val_loader,device, mm_graph)
    # breakpoint()
    model.train()
    for epoch in range(epochs):
        for data in tqdm(train_loader):
            motifs = motifs.to(device)
            mm_graph = mm_graph.to(device)
            head_pairs, tail_pairs, rel, label = [d.to(device) for d in data]
            pred = model((head_pairs, tail_pairs, rel,mm_graph),True)
            if torch.isnan(pred).any():
                print('error')
                return
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_cls = (torch.sigmoid(pred) > 0.5).detach().cpu().numpy()
            acc = accuracy(label.detach().cpu().numpy(), pred_cls)
            running_acc.update(acc)
            running_loss.update(loss.item(), label.size(0)) 

        epoch_loss = running_loss.get_average()
        epoch_acc = running_acc.get_average()
        running_loss.reset()
        running_acc.reset()

        val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap  = val(model, criterion, val_loader, device,mm_graph)

        msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, s1-%.4f, val_acc-%.4f, val_auroc-%.4f, val_f1_score-%.4f, val_prec-%.4f, val_rec-%.4f, val_ap-%.4f" % (epoch, epoch_loss, epoch_acc, val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap)
        logger.info(msg)
        if save_model:
            msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, s1-%.4f, val_acc-%.4f" % (epoch, epoch_loss, epoch_acc, val_loss, val_acc)
            # del_file(logger.get_model_dir())
            save_model_dict(model, logger.get_model_dir(), msg)
        #cold_start
        val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap  = val(model, criterion, test_loader, device,mm_graph)

        msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, s2-%.4f, val_acc-%.4f, val_auroc-%.4f, val_f1_score-%.4f, val_prec-%.4f, val_rec-%.4f, val_ap-%.4f" % (epoch, epoch_loss, epoch_acc, val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap)
        logger.info(msg)

        scheduler.step()

        if save_model:
            msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, s2-%.4f, val_acc-%.4f" % (epoch, epoch_loss, epoch_acc, val_loss, val_acc)
            save_model_dict(model, logger.get_model_dir(), msg)

if __name__ == "__main__":
    main()
