import argparse
import sys
# import cv2
from config.base_config import cfg_from_file, cfg, print_cfg, get_models_dir
import os.path as osp
import numpy as np
from utils.dictionary import Dictionary
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
import os
from model.baseline import baseNet
from model.transformer import Transformer
from model.ltransformer import LTransformer
import os,datetime,time

model_dict={
    'baseline':baseNet,
    'transformer':Transformer,
    'lwtransformer':LTransformer,
}

def debug_pred(debug_dir, count, qvec, cvec, img, gt_bbox, roi, bbox_pred, iou):
    # debug
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    qdic_dir = osp.join(cfg.DATA_DIR, cfg.IMDB_NAME, 'query_dict')

    qdic = Dictionary(qdic_dir)
    qdic.load()
    q_str = []
    for idx in qvec:
        if int(idx) != 0:
            q_str.append(qdic.get_token(idx))
    q_str = ' '.join(q_str)
    if iou >= 0.5:
        save_dir = os.path.join(debug_dir, 'right/' + str(count))
    else:
        save_dir = os.path.join(debug_dir, 'wrong/' + str(count))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # with open(save_dir + '/%.3f' % iou, 'w') as f:
    #     f.write(' ')
    # json.dump(list(cont.astype(np.int)), open(save_dir+'/%s'%q_str, 'w'))
    # json.dump(list(query.astype(np.int)), open(save_dir+'/query.json', 'w'))
    with open(save_dir + '/query.txt', 'w') as f:
        f.write(q_str)
    pred = img.copy()
    box = gt_bbox.astype(np.int)
    cv2.rectangle(pred, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    box = roi.astype(np.int)
    cv2.rectangle(pred, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
    box = bbox_pred.astype(np.int)
    cv2.rectangle(pred, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    cv2.imwrite(save_dir + '/pred_%.2f'%iou+'__'+str(count)+'.jpg', pred)

def test_net(split,vis=False):
    dataset= RefDataset(split)
    dataloader = Data.DataLoader(
        dataset,
        batch_size=cfg.BATCHSIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEM,
        drop_last=True
    )
    vocab_size = get_vocab_size(cfg.QUERY_DIR)
    net = model_dict[cfg.MODEL](vocab_size, dataset.pretrain_embed)
    net.load_state_dict(state_dict=torch.load(cfg.EVAL_MODEL)['state_dict'])
    net.cuda()
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
            t_query_bbox_pred = clip_boxes(query_bbox_pred[i], img_shape[i])
            t_rois = clip_boxes(bbox[i], img_shape[i])
            # for j in range(topk):
            query_ind = query_inds[i, 0]

            # print(t_query_bbox_pred[query_ind],gt_bbox[i])
            iou = calc_iou(t_query_bbox_pred[query_ind], gt_bbox[i])
            if iou>=cfg.OVERLAP_THRESHOLD:
                num_right+=1
            # debug pred

            if vis:
                debug_dir = 'visual_pred_%s_%s' % (cfg.IMDB_NAME, split)
                img_path = dataset.get_img_path(int(iid[i].numpy()))
                img = cv2.imread(img_path)
                # print(img.shape)
                debug_pred(debug_dir, count, qvec[i].cpu().numpy(), None, img, gt_bbox[i], t_rois[query_ind],
                           t_query_bbox_pred[query_ind], iou)

            # percent = 100 * float(count) / num_query
            # print('\r' + ('%.2f' % percent) + '%')
            # count += 1
    accuracy = num_right / float(num_query)
    print('accuracy: %f\n' % accuracy)
    return accuracy

parser = argparse.ArgumentParser(description='Lightweight Transformer')
parser.add_argument('--version', type=str)
args = parser.parse_args()
cfg.EVAL_MODEL=osp.join(cfg.ROOT_DIR, 'models',cfg.MODEL+'_'+args.version,'best.pkl')
print(test_net('val'))
print(test_net('testA'))
print(test_net('testB'))
