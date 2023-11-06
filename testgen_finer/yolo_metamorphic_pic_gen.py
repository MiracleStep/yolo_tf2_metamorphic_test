from yolo_coverage import YoloV5Model
import csv
import os
import argparse
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
from yolo_coverage import *


def clear_folder(folder_path):
    # 使用shutil模块的rmtree函数删除文件夹及其内部的所有文件
    shutil.rmtree(folder_path)

    # 使用os模块的makedirs函数重新创建一个同名的空文件夹
    os.makedirs(folder_path)

# 生成蜕变测试图像
def gen_metamorphic_pic(index, VOCdevkit_path, image_index_path, VOCdevkit_out_path):
    image_ids = open(os.path.join(VOCdevkit_path, image_index_path)).read().strip().split()
    image_ids = image_ids[:100]
    if index == -1:
        # clear all file
        clear_folder(os.path.join(VOCdevkit_out_path, "VOC2007/ImageSets/Main/"))
        clear_folder(os.path.join(VOCdevkit_out_path, "VOC2007/JPEGImages/"))
        clear_folder(os.path.join(VOCdevkit_out_path, "VOC2007/Annotations/"))

    if index == 0:
        with open(os.path.join(VOCdevkit_out_path, "VOC2007/ImageSets/Main/" + str(index) + ".txt"), "w") as f:
            for image_id in tqdm(image_ids):
                image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                seed_image = cv2.imread(image_path)
                for p in range(1, 11):
                    save_image = seed_image
                    file_name = image_id + "_" + str(p) +"_index_" + str(index)
                    cv2.imwrite(os.path.join(VOCdevkit_out_path, "VOC2007/JPEGImages/"+ file_name +".jpg"), save_image)
                    shutil.copyfile(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+ image_id +".xml"),
                                    os.path.join(VOCdevkit_out_path, "VOC2007/Annotations/"+ file_name +".xml"))
                    f.write(file_name+"\n")
    elif index == 1: # 平移
        with open(os.path.join(VOCdevkit_out_path, "VOC2007/ImageSets/Main/" + str(index) + ".txt"), "w") as f:
            for image_id in tqdm(image_ids):
                image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                seed_image = cv2.imread(image_path)
                for p in range(1, 11):
                    params = [p * 10, p * 10]  # 图片平移参数（10， 10） -> (100, 100) stride = (10, 10)
                    save_image = image_translation(seed_image, params)

                    file_name = image_id + "_" + str(p) +"_index_" + str(index)
                    cv2.imwrite(os.path.join(VOCdevkit_out_path, "VOC2007/JPEGImages/"+ file_name +".jpg"), save_image)
                    shutil.copyfile(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+ image_id +".xml"),
                                    os.path.join(VOCdevkit_out_path, "VOC2007/Annotations/"+ file_name +".xml"))
                    f.write(file_name+"\n")
    elif index == 2:# 缩放
        with open(os.path.join(VOCdevkit_out_path, "VOC2007/ImageSets/Main/" + str(index) + ".txt"), "w") as f:
            for image_id in tqdm(image_ids):
                image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                seed_image = cv2.imread(image_path)
                for p in range(1, 11):
                    params = [p*0.5+1, p*0.5+1] # 图片缩放参数
                    save_image = image_scale(seed_image, params)

                    file_name = image_id + "_" + str(p) +"_index_" + str(index)
                    cv2.imwrite(os.path.join(VOCdevkit_out_path, "VOC2007/JPEGImages/"+ file_name +".jpg"), save_image)
                    shutil.copyfile(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+ image_id +".xml"),
                                    os.path.join(VOCdevkit_out_path, "VOC2007/Annotations/"+ file_name +".xml"))
                    f.write(file_name+"\n")
    elif index == 3: # 剪切
        with open(os.path.join(VOCdevkit_out_path, "VOC2007/ImageSets/Main/" + str(index) + ".txt"), "w") as f:
            for image_id in tqdm(image_ids):
                image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                seed_image = cv2.imread(image_path)
                for p in range(1, 11):
                    params = 0.1 * p
                    save_image = image_scale(seed_image, params)

                    file_name = image_id + "_" + str(p) +"_index_" + str(index)
                    cv2.imwrite(os.path.join(VOCdevkit_out_path, "VOC2007/JPEGImages/"+ file_name +".jpg"), save_image)
                    shutil.copyfile(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+ image_id +".xml"),
                                    os.path.join(VOCdevkit_out_path, "VOC2007/Annotations/"+ file_name +".xml"))
                    f.write(file_name+"\n")
    elif index == 4: # 旋转
        with open(os.path.join(VOCdevkit_out_path, "VOC2007/ImageSets/Main/" + str(index) + ".txt"), "w") as f:
            for image_id in tqdm(image_ids):
                image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                seed_image = cv2.imread(image_path)
                for p in range(1, 11):
                    params = p * 3
                    save_image = image_rotation(seed_image, params)

                    file_name = image_id + "_" + str(p) +"_index_" + str(index)
                    cv2.imwrite(os.path.join(VOCdevkit_out_path, "VOC2007/JPEGImages/"+ file_name +".jpg"), save_image)
                    shutil.copyfile(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+ image_id +".xml"),
                                    os.path.join(VOCdevkit_out_path, "VOC2007/Annotations/"+ file_name +".xml"))
                    f.write(file_name+"\n")
    elif index == 5: # 对比度
        with open(os.path.join(VOCdevkit_out_path, "VOC2007/ImageSets/Main/" + str(index) + ".txt"), "w") as f:
            for image_id in tqdm(image_ids):
                image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                seed_image = cv2.imread(image_path)
                for p in range(1, 11):
                    params = 1 + p * 0.2
                    save_image = image_contrast(seed_image, params)

                    file_name = image_id + "_" + str(p) +"_index_" + str(index)
                    cv2.imwrite(os.path.join(VOCdevkit_out_path, "VOC2007/JPEGImages/"+ file_name +".jpg"), save_image)
                    shutil.copyfile(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+ image_id +".xml"),
                                    os.path.join(VOCdevkit_out_path, "VOC2007/Annotations/"+ file_name +".xml"))
                    f.write(file_name+"\n")
    elif index == 6: # 亮度
        with open(os.path.join(VOCdevkit_out_path, "VOC2007/ImageSets/Main/" + str(index) + ".txt"), "w") as f:
            for image_id in tqdm(image_ids):
                image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                seed_image = cv2.imread(image_path)
                for p in range(1, 11):
                    params = p * 10
                    save_image = image_brightness(seed_image, params)

                    file_name = image_id + "_" + str(p) +"_index_" + str(index)
                    cv2.imwrite(os.path.join(VOCdevkit_out_path, "VOC2007/JPEGImages/"+ file_name +".jpg"), save_image)
                    shutil.copyfile(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+ image_id +".xml"),
                                    os.path.join(VOCdevkit_out_path, "VOC2007/Annotations/"+ file_name +".xml"))
                    f.write(file_name+"\n")
    elif index == 7: # 模糊
        with open(os.path.join(VOCdevkit_out_path, "VOC2007/ImageSets/Main/" + str(index) + ".txt"), "w") as f:
            for image_id in tqdm(image_ids):
                image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                seed_image = cv2.imread(image_path)
                for p in range(1, 11):
                    params = p
                    save_image = image_blur(seed_image, params)

                    file_name = image_id + "_" + str(p) +"_index_" + str(index)
                    cv2.imwrite(os.path.join(VOCdevkit_out_path, "VOC2007/JPEGImages/"+ file_name +".jpg"), save_image)
                    shutil.copyfile(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+ image_id +".xml"),
                                    os.path.join(VOCdevkit_out_path, "VOC2007/Annotations/"+ file_name +".xml"))
                    f.write(file_name+"\n")


if __name__ == '__main__':
    index = 7
    VOCdevkit_path = "../VOCdevkit"
    image_index_path = "VOC2007/ImageSets/Main/test.txt"
    VOCdevkit_out_path = "../VOC_metamorphic"
    gen_metamorphic_pic(index, VOCdevkit_path, image_index_path, VOCdevkit_out_path)