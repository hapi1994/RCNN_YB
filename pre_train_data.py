import glob
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os

NUM_CLASSES = 3
NUM_IMAGES_PER_CLASS = 50
BB_PATHS = 'ILSVRC2012_bbox_train_dogs'

class BBOX(object):
    def __init__(self, typename, min_x, min_y, max_x, max_y):
        self._typename = typename
        self._min_x = min_x
        self._min_y = min_y
        self._max_x = max_x
        self._max_y = max_y


def parseXML(xmlfile_path):
    tree = ET.parse(xmlfile_path)
    root = tree.getroot()
    filename = root.find('filename').text
    bboxes = []
    for member in root.findall('object'):
        bbox = BBOX(member[0].text,
                 int(member[4][0].text),
                 int(member[4][1].text),
                 int(member[4][2].text),
                 int(member[4][3].text)
                 )
        bboxes.append(bbox)
    return filename, bboxes


def get_bboxes():
    BB_CLASSES_PATHS = os.listdir(BB_PATHS)
    res = {}
    for cat in range(NUM_CLASSES):
        CLASS_XML_FILES = glob.glob(os.path.join(BB_PATHS, BB_CLASSES_PATHS[cat] + "/*.xml"))
        #print(CLASS_XML_FILES)
        for xml_idx in range(NUM_IMAGES_PER_CLASS):
            filename, bboxes = parseXML(CLASS_XML_FILES[xml_idx])
            res[filename] = bboxes
    #print(res)
    #file_ = open('pre_train_data.txt', 'w')
    #file_.write('filename\tclass\tmin_x\tmin_y\tmax_x\tmax_y\n')
    #for filename in res:
    #    bboxes = res[filename]
    #    for bbox in bboxes:
    #        file_.write(filename+"\t"+str(bbox._typename)+"\t"+str(bbox._min_x)+"\t"+str(bbox._min_y)+"\t"+str(bbox._max_x)+"\t"+str(bbox._max_y)+"\n")
    #file_.close()
    return res
