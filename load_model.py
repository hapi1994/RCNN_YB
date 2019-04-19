import selectivesearch
import tensorflow as tf
import cv2
import numpy as np
from sklearn import svm

svm_trainingset_path = 'svm_train_convert.txt'

def get_svm_trainingset(train_path):
    imgs = []
    lbls = []

    f = open(train_path, 'w')
    lines = f.read().splitlines()

    for line in lines:
        img_data = cv2.imread(line[0])
        img_data = cv2.resize(img_data, (227, 227))
        imgs.append(img_data)
        lbls.append(line[1])

    return np.array(imgs), np.array(lbls)


def generate_proposals(img_path):
    img_data = cv2.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(img_data, scale=500, sigma=0.9, min_size=10)
    imgs = []
    bonds = []
    for r in regions:
        rect = r['rect']
        bound = [rect[1], rect[1]+rect[3], rect[0], rect[0]+rect[2]]
        if r['size'] < 220 or rect[2] == 0 or rect[3] == 0:
            continue
        img = img_data[bound[0]: bound[1], bound[2]: bound[3]]
        imgs.append(cv2.resize(img, (227, 227)))
        bonds.append([rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]])
    imgs = np.array(imgs)
    bonds = np.array(bonds)
    return imgs, bonds


if __name__ == '__main__':
    img_path = 'test.jpeg'
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('finetune_alexnet/checkpoints/model_epoch100.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('finetune_alexnet/checkpoints/'))
        graph = tf.get_default_graph()

        img_tensor = graph.get_tensor_by_name('Placeholder:0')
        keep_prop_tensor = graph.get_tensor_by_name('Placeholder_2:0')
        output_tensor = graph.get_tensor_by_name('fc7/fc7:0')

        svm_train_img, svm_train_lbl = get_svm_trainingset(svm_trainingset_path)
        svm_train_features = sess.run(output_tensor, feed_dict={img_tensor: svm_train_img, keep_prop_tensor: 0.5})
        #start to train svm model
        model = svm.SVC(kernel='linear')
        model.fit(svm_train_features, svm_train_lbl)

        # start to predict
        image_data = cv2.imread(img_path)

        imgs,bonds = generate_proposals(img_path)
        allpred_bonds = []

        num_cnt = imgs.shape[0] / 128
        for i in range(int(num_cnt)):
            bonds_tmp = bonds[i*128: i*128+128]
            res = sess.run(output_tensor, feed_dict={img_tensor: imgs[i*128: i*128+128], keep_prop_tensor: 0.5})
            predict_lbls = model.predict(res)
            indexes = np.where(predict_lbls!=0)
            allpred_bonds.append(bonds_tmp[indexes])
#            choose_index = np.argmax(res, axis=1)
#            choose_index = np.where(choose_index==1)
#            pred_bonds = bonds_tmp[choose_index]
#            if pred_bonds.shape[0] != 0:
#                allpred_bonds.append(pred_bonds)
#
        allpred_bonds = np.vstack(allpred_bonds)

        for item_bond in allpred_bonds:
            cv2.rectangle(image_data, (item_bond[0], item_bond[1]), (item_bond[2], item_bond[3]), (0, 255, 0), 3)
        cv2.imshow('images', image_data)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



