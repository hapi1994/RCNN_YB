import pre_train_data
import selectivesearch
import cv2
import os
import numpy as np

IMG_PATH = 'ILSVRC2012_img_train_t3'
PROPOSAL_SAVE_PATH = 'proposal_trainingset'
SVM_SAVE_PATH = 'svm_trainingset'
SVM = True

def get_proposals(img_path):
    img = cv2.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    #print(img_lbl)
    #print('---')
    #print(regions)
    return regions


def roi(prop, bbox):
    min_x = np.maximum(prop[0], bbox[:, 0]).astype(np.float32)
    min_y = np.maximum(prop[1], bbox[:, 1]).astype(np.float32)
    max_x = np.minimum(prop[2] + prop[0], bbox[:, 2]).astype(np.float32)
    max_y = np.minimum(prop[3] + prop[1], bbox[:, 3]).astype(np.float32)
    overlap = (min_x < max_x) & (min_y < max_y)
    area_bbox = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    area_prop = prop[2] * prop[3]
    area_overlap = (max_x - min_x) * (max_y - min_y)
    res = np.zeros_like(min_x)
    res[overlap] = area_overlap[overlap] / (area_bbox[overlap] + area_prop - area_overlap[overlap])
    res = sorted(res)
    return res[-1]


if __name__ == '__main__':
    gt_bboxes = pre_train_data.get_bboxes()
    all_gt_bboxes = list(gt_bboxes.items())

    # save train.txt and proposal training images
    prop_train_txt = open('prop_train.txt', 'w')
    prop_img_idx = 0

    svm_train_txt = open('svm_train.txt', 'w')
    svm_img_idx = 0

    for each_gt_bbox in all_gt_bboxes:
        filename = each_gt_bbox[0]
        gt_bbox = each_gt_bbox[1]

        typename = gt_bbox[0]._typename
        bbox = []
        for b in gt_bbox:
            bbox.append([b._min_x, b._min_y, b._max_x, b._max_y])
        bbox = np.array(bbox)
        folder = filename.split('_')[0]
        img_path = os.path.join(IMG_PATH, folder + "/" + filename + ".jpeg")
        img = cv2.imread(img_path)
        regions = get_proposals(img_path)
        roi_score = []
        regions_tmp = []
        s = set()
        for r in regions:
            rect = r['rect']
            if r['size'] < 220 or rect[2] == 0 or rect[3] == 0 or rect in s:
                continue
            s.add(rect)
            region_oi = roi([rect[0], rect[1], rect[2], rect[3]], bbox)
            roi_score.append(region_oi)
            regions_tmp.append(r)
    #    idx = 0
    #    while True:
    #        img = cv2.imread(img_path)
    #        cv2.rectangle(img, (bbox[0][0], bbox[0][1]), (bbox[0][2], bbox[0][3]), (0, 255, 0), 3)
    #        r = regions[idx]['rect']
    #        cv2.rectangle(img, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 3)
    #        region_oi = roi([r[0], r[1], r[2], r[3]], bbox)
    #        print(region_oi)
    #        cv2.imshow('image', img)
    #        k = cv2.waitKey(0) & 0xFFFF
    #        if k == 108:#l
    #            idx += 1
    #        if k == 113:#q
    #            break
    #    cv2.destroyAllWindows()
        prop_labs = []
        prop_bboxes = []
        prop_labs_svm = []
        prop_bboxes_svm = []
        for i in range(len(roi_score)):
            rect = regions_tmp[i]['rect']
            if roi_score[i] > 0.5:
                prop_labs.append(typename)
            else:
                prop_labs.append('background')
            prop_bboxes.append([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]])
            if SVM:
                if roi_score[i] > 0.5:
                    prop_labs_svm.append(typename)
                    prop_bboxes_svm.append([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]])
                if roi_score[i] < 0.3:
                    prop_labs_svm.append('background')
                    prop_bboxes_svm.append([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]])
        # choose only 10 background proposals for each image
        prop_labs = np.array(prop_labs)
        prop_bboxes = np.array(prop_bboxes)
        bg_indexes = np.where(prop_labs == 'background')[0]
        np.random.shuffle(bg_indexes)
        bg_indexes = bg_indexes[:10]
        non_bg_indexes = np.where(prop_labs != 'background')[0]
        prop_labs = np.concatenate((prop_labs[bg_indexes], prop_labs[non_bg_indexes]))
        prop_bboxes = np.concatenate((prop_bboxes[bg_indexes], prop_bboxes[non_bg_indexes]))

        prop_labs_svm = np.array(prop_labs_svm)
        prop_bboxes_svm = np.array(prop_bboxes_svm)
        bg_indexes_svm = np.where(prop_labs_svm == 'background')[0]
        np.random.shuffle(bg_indexes_svm)
        bg_indexes_svm = bg_indexes_svm[:10]
        non_bg_indexes_svm = np.where(prop_labs_svm != 'background')[0]
        prop_labs_svm = np.concatenate((prop_labs_svm[bg_indexes_svm], prop_labs_svm[non_bg_indexes_svm]))
        prop_bboxes_svm = np.concatenate((prop_bboxes_svm[bg_indexes_svm], prop_bboxes_svm[non_bg_indexes_svm]))

        for i in range(len(prop_labs)):
            img_save_path = os.path.join(PROPOSAL_SAVE_PATH, str(prop_img_idx) + '.jpeg')
            box = prop_bboxes[i]
            img_save_data = img[box[1] : box[3]+1, box[0]: box[2]+1]
            cv2.imwrite(img_save_path, img_save_data)
            prop_train_txt.write(img_save_path + " " + prop_labs[i] + "\n")
            prop_img_idx += 1
        
        for i in range(len(prop_labs_svm)):
            img_save_path = os.path.join(SVM_SAVE_PATH, str(svm_img_idx) + '.jpeg')
            box = prop_bboxes_svm[i]
            img_save_data = img[box[1] : box[3]+1, box[0]: box[2]+1]
            cv2.imwrite(img_save_path, img_save_data)
            svm_train_txt.write(img_save_path + " " + prop_labs_svm[i] + "\n")
            svm_img_idx += 1
        print(filename + " DONE!")
        #break
    prop_train_txt.close()
    svm_train_txt.close()
