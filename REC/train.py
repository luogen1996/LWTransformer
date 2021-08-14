import argparse
import sys
from config.base_config import cfg_from_file, cfg, print_cfg, get_models_dir
import os.path as osp
import numpy as np
from utils.dictionary import Dictionary
from model.baseline import baseNet
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import get_vocab_size
import torch.utils.data as Data
from dataset.dataset import RefDataset
from utils.optim import get_optim,adjust_lr
from utils.bbox_transform import bbox_transform_inv,clip_boxes,calc_iou
from utils.loss import SmoothL1Loss
from model.transformer import Transformer
from model.ltransformer import LTransformer
import os,datetime,time

model_dict={
    'baseline':baseNet,
    'transformer':Transformer,
    'lwtransformer':LTransformer
}


def test_net(net,split):
    dataset= RefDataset(split)
    dataloader = Data.DataLoader(
        dataset,
        batch_size=cfg.BATCHSIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEM,
        drop_last=True
    )
    num_query=dataset.__len__()
    count=0.
    num_right=0.
    net.eval()
    for step, (gt_bbox, qvec, img_feat, bbox, img_shape,
               spt_feat, query_score_targets, query_score_mask,
               query_bbox_targets, query_bbox_inside_weights, query_bbox_outside_weights,
               valid_data, iid
               ) in enumerate(dataloader):
        qvec = qvec.cuda().long()
        img_feat = img_feat.cuda().float()
        bbox_ = bbox.cuda().float()

        with torch.no_grad():
            query_score_pred, query_bbox_pred = net(qvec, img_feat, bbox_)

        query_score_pred=query_score_pred.cpu().detach().numpy()
        query_bbox_pred=query_bbox_pred.cpu().detach().numpy()

        if cfg.USE_REG:
            if cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                query_bbox_pred = query_bbox_pred*np.array(cfg.BBOX_NORMALIZE_STDS)+np.array(cfg.BBOX_NORMALIZE_MEANS)
            query_bbox_pred = bbox_transform_inv(np.reshape(bbox.numpy(), (-1, 4)),
                                                 np.reshape(query_bbox_pred, (-1, 4)))
        else:
            query_bbox_pred = bbox.reshape(-1, 4).numpy()

        query_inds = np.argsort(-query_score_pred, axis=1)

        batchsize=query_score_pred.shape[0]
        bbox=bbox.numpy()
        gt_bbox=gt_bbox.numpy()
        query_bbox_pred=np.reshape(query_bbox_pred,bbox.shape)
        # print(bbox.shape,query_bbox_pred.shape)
        for i in range(batchsize):
            if valid_data[i] != 0:
                t_query_bbox_pred = clip_boxes(query_bbox_pred[i], img_shape[i])
                t_rois = clip_boxes(bbox[i], img_shape[i])
                # for j in range(topk):
                query_ind = query_inds[i, 0]

                # print(t_query_bbox_pred[query_ind],gt_bbox[i])
                iou = calc_iou(t_query_bbox_pred[query_ind], gt_bbox[i])
                if iou>=cfg.OVERLAP_THRESHOLD:
                    num_right+=1
                # debug pred
                # if vis:
                #     debug_dir = 'visual_pred_%s_%s' % (cfg.IMDB_NAME, test_split)
                #     img_path = dp.get_img_path(int(iid_list[i]))
                #     img = cv2.imread(img_path)
                #     img.shape
                #     debug_pred(debug_dir, count, tp_qvec[i], tp_cvec[i], img, gt_boxes[i], t_rois[query_ind],
                #                t_query_bbox_pred[query_ind], iou)

            # percent = 100 * float(count) / num_query
            # print('\r' + ('%.2f' % percent) + '%')
            # count += 1
    accuracy = num_right / float(num_query)
    print('accuracy: %f\n' % accuracy)
    return accuracy

def train():
    # train_loss = np.zeros(MAX_ITERATIONS + 1)
    dataset= RefDataset('train')
    dataloader = Data.DataLoader(
        dataset,
        batch_size=cfg.BATCHSIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEM,
        drop_last=True
    )
    vocab_size=get_vocab_size(cfg.QUERY_DIR)
    net = model_dict[cfg.MODEL](vocab_size,dataset.pretrain_embed)
    net.cuda()
    #log file
    logfile = open(
        cfg.MODEL_DIR +
        '/log_run_' + cfg.MODEL + '.txt',
        'a+'
    )
    logfile.write(str(cfg))
    logfile.close()
    # F.cross_entropy
    if cfg.USE_KLD:
        criterion_conf = nn.KLDivLoss(reduction='batchmean')
    else:
        criterion_conf = nn.CrossEntropyLoss(reduction='batchmean',ignore_index=-1)
    criterion_regr=SmoothL1Loss
    # nn.SmoothL1Loss
    start_epoch=0
    best_acc=0.
    optim = get_optim(cfg, net, dataloader.__len__()*cfg.BATCHSIZE)
    # Training script
    for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
        time_start = time.time()
        logfile = open(
            cfg.MODEL_DIR +
            '/log_run_' + cfg.MODEL + '.txt',
            'a+'
        )
        logfile.write(
            '=====================================\nnowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n'
        )
        logfile.close()
        net.train()
        # Learning Rate Decay
        if epoch in cfg.TRAIN.LR_DECAY_LIST:
            adjust_lr(optim, cfg.TRAIN.LR_DECAY_R)

        # Iteration
        for step, (gt_bbox, qvec, img_feat, bbox, img_shape,
                   spt_feat, query_score_targets, query_score_mask,
                   query_bbox_targets, query_bbox_inside_weights, query_bbox_outside_weights,
                   valid_data, iid
        ) in enumerate(dataloader):
            optim.zero_grad()
            qvec=qvec.cuda().long()
            img_feat=img_feat.cuda().float()
            bbox=bbox.cuda().float()

            query_score_targets=query_score_targets.cuda().float()
            query_bbox_targets=query_bbox_targets.cuda().float()
            query_bbox_inside_weights = query_bbox_inside_weights.cuda().float()
            query_bbox_outside_weights = query_bbox_outside_weights.cuda().float()

            query_score_pred,query_bbox_pred=net(qvec,img_feat,bbox)

            # KLD loss  have backward,predict score
            if cfg.USE_KLD:
                #softmaxKldLoss
                query_score_pred = F.log_softmax(query_score_pred,-1)
            else:
                query_score_targets=query_score_targets.cuda().long()
            # print(query_score_targets)
            # print(query_score_pred)
            loss_query_score = criterion_conf(query_score_pred, query_score_targets)
            # regeression loss
            loss_query_bbox=criterion_regr(query_bbox_pred,query_bbox_targets,query_bbox_inside_weights,query_bbox_outside_weights)

            loss=loss_query_bbox*0.5+loss_query_score
            print("\r[Version %s][Model %s][Epoch %2d][Step %4d/%4d] Loss: %.4f, Conf: %.4f, Box: %.4f, Lr: %.2e" % (
                cfg.VERSION,
                cfg.MODEL,
                epoch + 1,
                step,
                int(dataloader.__len__()),
                loss.cpu().detach().numpy(),
                loss_query_score.cpu().detach().numpy(),
                loss_query_bbox.cpu().detach().numpy(),
                optim._rate
            ), end='          ')
            loss.backward()

            optim.step()
            # break
        time_end = time.time()
        elapse_time = time_end-time_start
        print('\nFinished in {}s'.format(int(elapse_time)))
        print('\nstart testing')
        acc=test_net(net,'val')
        state = {
            'state_dict': net.state_dict(),
            'optimizer': optim.optimizer.state_dict(),
            'lr_base': optim.lr_base,
            'epoch': epoch
        }
        if acc>best_acc:
            best_acc=acc
            torch.save(
                state,
                os.path.join(cfg.MODEL_DIR,'best.pkl')
            )
        # Logging
        logfile = open(
            cfg.MODEL_DIR +
            '/log_run_' + cfg.MODEL + '.txt',
            'a+'
        )
        logfile.write(
            'Epoch: ' + str(epoch) +
            ', Lr: ' + str(optim._rate) + '\n' +
            ', valAcc: %.4f'%acc +
            'Elapsed time: ' + str(int(elapse_time)) +
            ', Speed(s/batch): ' + str(elapse_time / step) +
            '\n\n'
        )
        logfile.close()
        # torch.save(
        #     state,
        #     os.path.join(cfg.MODEL_DIR,str(epoch) +
        #     '.pkl')
        # )


    # # predict bbox
train()