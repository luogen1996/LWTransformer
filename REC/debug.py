from dataset.dataset import RefDataset
from config.base_config import cfg_from_file, cfg
import torch.utils.data as Data
import torch
from utils.dictionary import Dictionary
import cv2
split='val'
dataset = RefDataset(split)
dataloader = Data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True
)
qdic = Dictionary(cfg.QUERY_DIR)
qdic.load()
print(
qdic.get_token(0),
qdic.get_token(1),
qdic.get_token(2),
qdic.get_token(3),
qdic.get_token(4))
# for step, (gt_bbox, qvec, img_feat, bbox, img_shape,
#            spt_feat, query_score_targets, query_score_mask,
#            query_bbox_targets, query_bbox_inside_weights, query_bbox_outside_weights,
#            valid_data, iid
#            ) in enumerate(dataloader):
#     qvec=qvec.long().numpy()
#     for i in range(len(qvec[0])):
#         print(qdic.get_token(qvec[0,i]))
#     # print(gt_bbox)
#     # print(query_bbox_targets.size(),query_score_targets.argmax(-1).size())
#     print(bbox.size())
#     tbox=bbox[0,query_score_targets.argmax(-1)].long().numpy()[0]
#     # print(tbox.size())
#     print(query_score_targets[0,query_score_targets.argmax(-1)[0]])
#
#     print(query_bbox_inside_weights[0,query_score_targets.argmax(-1)[0]])
#     print(query_bbox_outside_weights[0,query_score_targets.argmax(-1)[0]])
#     gt_bbox=gt_bbox.long().numpy()[0]
#     path=dataset.get_img_path(iid.long().numpy()[0])
#     img=cv2.imread(path)
#     img = cv2.rectangle(img, tuple(gt_bbox[:2]), tuple(gt_bbox[2:]), color=(0, 255, 0), thickness=3)
#     img = cv2.rectangle(img, tuple(tbox[:2]), tuple(tbox[2:]), color=(0, 0, 255), thickness=3)
#     cv2.imwrite('./test.jpg',img)
#     # print(gt_bbox.shape)
#     # print(query_bbox_inside_weights)
#     # print(torch.sum(query_bbox_inside_weights==query_bbox_outside_weights))
#     break