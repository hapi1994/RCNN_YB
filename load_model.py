import selectivesearch
import tensorflow as tf
import cv2
import numpy as np
from sklearn import svm

svm_trainingset_path = 'svm_train_convert.txt'

def get_svm_trainingset(train_path):
    imgs = []
    lbls = []

    f = open(train_path, 'r')
    lines = f.read().splitlines()

    for line in lines:
        comps = line.split(' ')
        img_data = cv2.imread(comps[0])
        img_data = cv2.resize(img_data, (227, 227))
        imgs.append(img_data)
        lbls.append(comps[1])

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
        svm_train_features = []
        NUM_RUNS = svm_train_img.shape[0] // 128
        for RUN in range(NUM_RUNS):
            output_feature = sess.run(output_tensor, feed_dict={img_tensor: svm_train_img[RUN * 128: RUN * 128 + 128], keep_prop_tensor: 0.5})
            svm_train_features.append(output_feature)
        svm_train_features = np.vstack(svm_train_features)
        svm_train_lbl = svm_train_lbl[:2048]
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
            indexes = np.where(predict_lbls!=0)[0]
            allpred_bonds.append(bonds_tmp[indexes])
        allpred_bonds = np.vstack(allpred_bonds)

        for item_bond in allpred_bonds:
            cv2.rectangle(image_data, (item_bond[0], item_bond[1]), (item_bond[2], item_bond[3]), (0, 255, 0), 3)
        cv2.imshow('images', image_data)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # with tf.Session() as sess:
        # saver = tf.train.import_meta_graph('finetune_alexnet/checkpoints/model_epoch100.ckpt.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('finetune_alexnet/checkpoints/'))
        # graph = tf.get_default_graph()

        # img_tensor = graph.get_tensor_by_name('Placeholder:0')
        # keep_prop_tensor = graph.get_tensor_by_name('Placeholder_2:0')
        # output_tensor = graph.get_tensor_by_name('fc8/fc8:0')

        # imgs, bonds = generate_proposals('test.jpeg')
        # num_loops = imgs.shape[0] // 128
        # res_bonds = []
        # for loop in range(num_loops):
            # temping_boxes = bonds[128 * loop: 128 * loop + 128]
            # classfy_res = sess.run(output_tensor, feed_dict={img_tensor: imgs[128 * loop:128 * loop + 128], keep_prop_tensor: 0.5})
            # classfy_res = np.argmax(classfy_res, axis=1)
            # choose = np.where(classfy_res != 0)[0]
            # res_bonds.append(temping_boxes[choose])
        # res_bonds = np.vstack(res_bonds)
        # image_data = cv2.imread('test.jpeg')
        # for item_bond in res_bonds:
            # cv2.rectangle(image_data, (item_bond[0], item_bond[1]), (item_bond[2], item_bond[3]), (0, 255, 0), 3)
        # cv2.imshow('image', image_data)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

