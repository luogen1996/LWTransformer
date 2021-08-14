from config.base_config import cfg
import os
import numpy as np
import skimage.io
from utils.dictionary import Dictionary
from utils.data_utils import save,load
from utils.bbox_transform import bbox_transform
from utils.bbox_transform import bbox_overlaps
import random
import torch.utils.data as Data
import en_vectors_web_lg
import re
class RefDataset(Data.Dataset):
    def __init__(self, data_split):
        print('init DataProvider for %s : %s : %s' % (cfg.IMDB_NAME, cfg.PROJ_NAME, data_split))
        self.is_ss = cfg.FEAT_TYPE == 'ss'
        self.ss_box_dir = cfg.SS_BOX_DIR
        self.ss_feat_dir = cfg.SS_FEAT_DIR
        self.feat_type = cfg.FEAT_TYPE
        self.pretrain_embed=None
        if 'refcoco' in cfg.IMDB_NAME or cfg.IMDB_NAME == 'refclef':
            self.is_mscoco_prefix = True
        else:
            self.is_mscoco_prefix = False

        self.use_kld = cfg.USE_KLD
        # self.mscoco_prefix = cfg.MSCOCO_PREFIX
        self.rpn_topn = cfg.RPN_TOPN
        if self.is_ss:
            self.bottomup_feat_dim = cfg.SS_FEAT_DIM
        else:
            self.bottomup_feat_dim = cfg.BOTTOMUP_FEAT_DIM

        self.query_maxlen = cfg.QUERY_MAXLEN
        # self.data_paths = cfg.DATA_PATHS
        self.image_ext = '.jpg'
        data_splits = data_split.split(cfg.SPLIT_TOK)
        if 'train' in data_splits:
            self.mode = 'train'
        else:
            self.mode = 'test'
        self.image_dir = cfg.IMAGE_DIR
        self.feat_dir = cfg.FEAT_DIR
        self.dict_dir = cfg.QUERY_DIR  # osp.join(cfg.DATA_DIR, cfg.IMDB_NAME, 'query_dict')
        self.anno = self.load_data(data_splits)
        self.qdic = Dictionary(self.dict_dir)
        self.qdic.load()
        if cfg.USE_GLOVE:
            self.pretrain_embed=self.init_glove_embeding()
        self.index = 0
        self.batch_len = None
        self.num_query = len(self.anno)

    def __getitem__(self, index):
        if self.batch_len is None:
            self.n_skipped = 0
            qid_list = self.get_query_ids()
            if self.mode == 'train':
                random.shuffle(qid_list)
            self.qid_list = qid_list
            self.batch_len = len(qid_list)
            self.epoch_counter = 0
            # print('mode %s has %d data' % (self.mode, self.batch_len))

        qid = self.qid_list[index]
        gt_bbox = np.zeros(4)
        qvec = np.zeros(self.query_maxlen)
        img_feat = np.zeros((self.rpn_topn, self.bottomup_feat_dim))
        bbox = np.zeros((self.rpn_topn, 4))
        img_shape = np.zeros(2)
        spt_feat = np.zeros((self.rpn_topn, 5))
        if self.use_kld:
            query_label = np.zeros(self.rpn_topn)

        query_label_mask = 0
        query_bbox_targets = np.zeros((self.rpn_topn, 4))
        query_bbox_inside_weights = np.zeros((self.rpn_topn, 4))
        query_bbox_outside_weights = np.zeros((self.rpn_topn, 4))

        valid_data = 1

        t_qstr = self.anno[qid]['qstr']
        t_qvec = self.str2list(t_qstr, self.query_maxlen)
        qvec[...] = t_qvec

        # try:
        t_gt_bbox = self.anno[qid]['boxes']
        gt_bbox[...] = t_gt_bbox[0]
        t_img_feat, t_num_bbox, t_bbox, t_img_shape = self.get_topdown_feat(self.anno[qid]['iid'])
        # t_img_feat = t_img_feat.transpose((1, 0))
        # t_img_feat = (t_img_feat / np.sqrt((t_img_feat ** 2).sum()))

        img_feat[:t_num_bbox, :] = t_img_feat
        bbox[:t_num_bbox, :] = t_bbox

        # spt feat
        img_shape[...] = np.array(t_img_shape)
        t_spt_feat = self.get_spt_feat(t_bbox, t_img_shape)
        spt_feat[:t_num_bbox, :] = t_spt_feat

        # query label, mask
        t_gt_bbox = np.array(self.anno[qid]['boxes'])
        t_query_label, t_query_label_mask, t_query_bbox_targets, t_query_bbox_inside_weights, t_query_bbox_outside_weights = \
            self.get_labels(t_bbox, t_gt_bbox)

        if self.use_kld:
            query_label[:t_num_bbox] = t_query_label
            query_label_mask = t_query_label_mask
        else:
            query_label = t_query_label

        query_bbox_targets[:t_num_bbox, :] = t_query_bbox_targets
        query_bbox_inside_weights[:t_num_bbox, :] = t_query_bbox_inside_weights
        query_bbox_outside_weights[:t_num_bbox, :] = t_query_bbox_outside_weights

        # except Exception as e:
        #     print(e)
        #     valid_data = 0
        #     if not self.use_kld:
        #         query_label = -1
        #     query_label_mask = 0
        #     query_bbox_inside_weights[...] = 0
        #     query_bbox_outside_weights[...] = 0
        #     print('data not found for iid: %s' % str(self.anno[qid]['iid']))

        if self.index >= self.batch_len - 1:
            self.epoch_counter += 1
            qid_list = self.get_query_ids()
            random.shuffle(qid_list)
            self.qid_list = qid_list
            print('a epoch passed')

        return gt_bbox, qvec, img_feat, bbox, img_shape, spt_feat, query_label, query_label_mask, \
               query_bbox_targets, query_bbox_inside_weights, query_bbox_outside_weights, valid_data, int(self.anno[qid]['iid'])

    def __len__(self):
        return self.num_query
    def init_glove_embeding(self):
        pretrained_emb = []
        spacy_tool=None
        if cfg.USE_GLOVE:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('BOS').vector)
            pretrained_emb.append(spacy_tool('EOS').vector)
        for i in range(4,self.qdic.size()):
            word=self.qdic.get_token(i)
            pretrained_emb.append(spacy_tool(word).vector)
        return np.array(pretrained_emb)
    def get_image_ids(self):
        qid_list = self.get_query_ids()
        iid_list = set()
        for qid in qid_list:
            iid_list.add(self.anno[qid]['iid'])
        return list(iid_list)

    def get_query_ids(self):
        return list(self.anno.keys())

    def get_num_query(self):
        return self.num_query

    def load_data(self, data_splits):
        anno = {}
        for data_split in data_splits:
            # data_path = osp.join(cfg.DATA_DIR, cfg.IMDB_NAME, 'format_%s.pkl'%str(data_split))
            data_path = cfg.ANNO_PATH % str(data_split)
            t_anno = load(data_path)
            anno.update(t_anno)
        return anno

    def get_vocabsize(self):
        return self.qdic.size()

    def get_iid(self, qid):
        return self.anno[qid]['iid']

    def get_img_path(self, iid):
        if self.is_mscoco_prefix:
            return os.path.join(self.image_dir, 'COCO_train2014_' + str(iid).zfill(12) + self.image_ext)
        else:
            return os.path.join(self.image_dir, str(iid) + self.image_ext)

    def str2list(self, qstr, query_maxlen):
        q_list = qstr.split()
        qvec = np.zeros(query_maxlen, dtype=np.int64)
        # cvec = np.zeros(query_maxlen, dtype=np.int64)
        for i, _ in enumerate(range(min(query_maxlen,len(q_list)))):

            # w = q_list[i - (query_maxlen - len(q_list))]
            w = q_list[i]
            qvec[i] = self.qdic.lookup(w)
                # cvec[i] = 0 if i == query_maxlen - len(q_list) else 1

        # return qvec, cvec
        return qvec

    def load_ss_box(self, ss_box_path):
        boxes = np.loadtxt(ss_box_path)
        if len(boxes) == 0:
            raise Exception("boxes is None!")
        boxes = boxes - 1
        boxes[:, [0, 1]] = boxes[:, [1, 0]]
        boxes[:, [2, 3]] = boxes[:, [3, 2]]
        return boxes

    def get_topdown_feat(self, iid):

        if self.is_ss:
            img_path = self.get_img_path(iid)
            im = skimage.io.imread(img_path)
            img_h = im.shape[0]
            img_w = im.shape[1]
            feat_path = os.path.join(self.ss_feat_dir, str(iid) + '.npz')
            ss_box_path = os.path.join(self.ss_box_dir, str(iid) + '.txt')
            bbox = self.load_ss_box(ss_box_path)
            num_bbox = bbox.shape[0]
            img_feat = np.transpose(np.load(feat_path)['x'], (1, 0))
        else:
            if self.is_mscoco_prefix:  # zfill(12) insert 0 before the str
                feat_path = os.path.join(self.feat_dir, 'COCO_train2014_' + str(iid).zfill(12) +  '.npy')
            else:
                feat_path = os.path.join(self.feat_dir, str(iid) +  '.npy')
            feat_dict = np.load(feat_path,allow_pickle=True)
            # print(feat_dict.item())
            img_feat = feat_dict.item(0)['features']
            num_bbox = len(feat_dict.item(0)['boxes'])
            bbox = feat_dict.item(0)['boxes']
            img_h,img_w,_=feat_dict.item(0)['img_shape']
            # import cv2
            # img=cv2.imread(img_path)
            # i=0
            # for gt_bbox in bbox:
            #     i+=1
            #     gt_bbox=np.array(gt_bbox).astype(np.int)
            #     img = cv2.rectangle(img, tuple(gt_bbox[:2]), tuple(gt_bbox[2:]), color=(0, 255, 0), thickness=3)
            #     if i>10:
            #         break
            # cv2.imwrite('./test.jpg',img)
            # print(img_feat.shape,bbox.shape)
        return img_feat, num_bbox, bbox, (img_h,img_w)


    def create_batch_rpn(self, iid):
        img_path = self.get_img_path(iid)
        # img = cv2.imread(img_path)
        img_feat, num_bbox, bbox = self.get_topdown_feat(iid)
        return num_bbox, bbox, img_path

    def create_batch_recall(self, qid):
        iid = self.anno[qid]['iid']
        gt_bbox = self.anno[qid]['boxes']
        img_path = self.get_img_path(iid)
        # img = cv2.imread(img_path)
        img_feat, num_bbox, bbox, img_shape = self.get_topdown_feat(iid)
        return num_bbox, bbox, gt_bbox, img_path

    def compute_targets(self, ex_rois, gt_rois, query_label):
        """Compute bounding-box regression targets for an image."""
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 4

        targets = bbox_transform(ex_rois, gt_rois)
        if cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - np.array(cfg.BBOX_NORMALIZE_MEANS))
                       / np.array(cfg.BBOX_NORMALIZE_STDS))
        query_bbox_target_data = np.hstack((query_label[:, np.newaxis], targets)).astype(np.float32, copy=False)
        return query_bbox_target_data

    def get_query_bbox_regression_labels(self, query_bbox_target_data):
        query_label = query_bbox_target_data[:, 0]
        query_bbox_targets = np.zeros((query_label.size, 4), dtype=np.float32)
        query_bbox_inside_weights = np.zeros(query_bbox_targets.shape, dtype=np.float32)
        inds = np.where(query_label > 0)[0]
        if len(inds) != 0:
            for ind in inds:
                query_bbox_targets[ind, :] = query_bbox_target_data[ind, 1:]
                if query_label[ind] == 1:
                    query_bbox_inside_weights[ind, :] = cfg.BBOX_INSIDE_WEIGHTS
                elif query_label[ind] == 2:
                    query_bbox_inside_weights[ind, :] = 0.2

        return query_bbox_targets, query_bbox_inside_weights

    # 获取 query score和 bbox regression 的 label, mask
    def get_labels(self, rpn_rois, gt_boxes):
        # overlaps: (rois x gt_boxes)
        # overlaps = bbox_overlaps(np.ascontiguousarray(rpn_rois, dtype=np.float), np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        overlaps = bbox_overlaps(np.ascontiguousarray(rpn_rois, dtype=np.float), np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        # print(overlaps.shape)
        if self.use_kld:
            query_label = np.zeros(rpn_rois.shape[0])
        query_label_mask = 0
        bbox_label = np.zeros(rpn_rois.shape[0])
        # keep_inds = []
        # 找出 query = 1 的 gt_box 的 index
        query_gt_ind = 0
        query_overlaps = overlaps[:, query_gt_ind].reshape(-1)

        if self.use_kld:
            # kld: 根据 iou 设置权重
            if query_overlaps.max() >= 0.5:
                query_label_mask = 1
                query_inds = np.where(query_overlaps >= cfg.THRESHOLD)[0]
                for ind in query_inds:
                    query_label[ind] = query_overlaps[ind]
                if query_label.sum() == 0:
                    print(query_overlaps.max())
                # query_label = query_label / float(query_label.sum())
        else:
            # softmax
            if query_overlaps.max() >= 0.5:
                query_label = int(query_overlaps.argmax())
            else:
                query_label = -1
        rois = rpn_rois
        gt_assignment = overlaps.argmax(axis=1)
        # print(rpn_rois[overlaps.argmax(axis=0)])
        # print(gt_boxes[0, :4])
        gt_target_boxes = gt_boxes[gt_assignment, :4]
        bbox_label[np.where(overlaps.max(axis=1) >= 0.5)[0]] = 2
        if query_overlaps.max() >= 0.5:
            query_inds = np.where(query_overlaps >= cfg.THRESHOLD)[0]
            bbox_label[query_inds] = 1
            gt_target_boxes[query_inds] = gt_boxes[query_gt_ind, :4]

        # print(gt_boxes.shape)
        # print(gt_assignment)
        # print(overlaps[overlaps.argmax(axis=0)])
        bbox_target_data = self.compute_targets(rois, gt_target_boxes, bbox_label)
        query_bbox_targets, query_bbox_inside_weights = self.get_query_bbox_regression_labels(bbox_target_data)
        query_bbox_outside_weights = np.array(query_bbox_inside_weights > 0).astype(np.float32)

        return query_label, query_label_mask, query_bbox_targets, query_bbox_inside_weights, query_bbox_outside_weights

    def get_spt_feat(self, bbox, img_shape):
        spt_feat = np.zeros((bbox.shape[0], 5), dtype=np.float)

        spt_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        spt_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        spt_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        spt_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        spt_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])
        return spt_feat