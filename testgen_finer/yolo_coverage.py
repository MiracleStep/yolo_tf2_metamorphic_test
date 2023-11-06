import argparse
import csv
import os
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

from nets.yolo import yolo_body
from yolo import YOLO
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class YoloV5Model(object):
    def __init__(self, model_path="model_data/yolov5_s.h5", anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], num_classes=80, phi="s", nactivated_threshold=0.4):
        self.crop = False # 指定了是否在单张图片预测后对目标进行截取
        self.count = False # 指定了是否进行目标的计数
        self.model_path = model_path
        self.anchors_mask = anchors_mask
        self.num_classes = num_classes
        self.phi = phi
        self.nactivated_threshold = nactivated_threshold


        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        self.model = yolo_body([None, None, 3], self.anchors_mask, self.num_classes, self.phi)
        self.model.load_weights(self.model_path)
        print('{} model, anchors, and classes loaded.'.format(model_path))
        print(self.model.summary())


        # 神经元覆盖类
        # self.nc_yolo = NCoverage(self.model, self.nactivated_threshold, only_layer=only_layer)

    def scale(self, layer_outputs, rmax=1, rmin=0):
        '''
        scale the intermediate layer's output between 0 and 1
        :param layer_outputs: the layer's output tensor
        :param rmax: the upper bound of scale4
        :param rmin: the lower bound of scale
        :return:
        '''
        divider = (layer_outputs.max() - layer_outputs.min())
        if divider == 0:
            return np.zeros(shape=layer_outputs.shape)
        X_std = (layer_outputs - layer_outputs.min()) / divider
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    def only_predict(self):
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        self.yolo = YOLO()
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = self.yolo.detect_image(image, crop=self.crop, count=self.count)
                r_image.show()

    # 用于计算总神经元和覆盖神经元的个数并预测最终结果的函数
    def predict_fn(self, img, img_size=640):
        r_image = self.yolo.detect_image(img, crop=self.crop, count=self.count)

        ndict = {}
        include_layer = ['focus', 'conv', 'spp', 'bottleneck_csp']

        for layer in self.model.layers[1:]:
            if any(inc in layer.name for inc in include_layer):
                # print('calculating layer: ', layer.name, layer.output_shape)
                # 构建需要进行神经元覆盖测试的子类,考虑复合层与单层两种情况
                if len(layer.submodules):
                    name_outputs_dict = {}
                    for (name, outputs) in layer.get_neuron().items():
                        name_outputs_dict[layer.name + '-' + name] = outputs.numpy()
                else:
                    layer_outputs = tf.keras.models.Model(inputs=self.model.input, outputs=layer.output).predict(img1)
                    name_outputs_dict = {layer.name: layer_outputs}

                # 按照输出值最后一维的维度，计算神经元与否
                cov_dict = {}
                # print('nactivated_threshold :', self.nactivated_threshold)
                for (name, outputs) in name_outputs_dict.items():
                    for i, layer_output in enumerate(outputs):
                        scaled = self.scale(layer_output)
                        #print("sub_Layer name:", name, "\tscaled shape:", scaled.shape)

                        for neuron_idx in range(scaled.shape[-1]):
                            if np.mean(scaled[..., neuron_idx]) > self.nactivated_threshold:
                                cov_dict[(str(i + 1) + '-' + name, neuron_idx + 1)] = True
                            else:
                                cov_dict[(str(i + 1) + '-' + name, neuron_idx + 1)] = False
                print(layer.name + '\'s sub_Layer neurons number:', len(cov_dict))
                print("========")
                ndict.update(cov_dict)

        covered_neurons = len([v for v in ndict.values() if v])
        total_neurons = len(ndict)
        return ndict, covered_neurons, total_neurons, pred_bbox


def image_translation(img, params):
    rows, cols, ch = img.shape

    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def image_scale(img, params):
    res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
    return res


def image_shear(img, params):
    rows, cols, ch = img.shape
    factor = params * (-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def image_rotation(img, params):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def image_contrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))  # mul_img = img*alpha
    # new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

    return new_img


def image_brightness(img, params):
    beta = params
    new_img = cv2.add(img, beta)  # new_img = img*alpha + beta

    return new_img


def image_blur(img, params):
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    if params == 9:
        blur = cv2.blur(img, (6, 6))
    if params == 10:
        blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


def yolo_coverage(index, data_path):
    nactivated_threshold = 0.4 # 激活阈值
    index = int(index)

    model = YoloV5Model(nactivated_threshold=0.5)

    file_list = os.listdir(data_path)

    with open("result/yolo_images_coverage.csv", "a") as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['index', 'image', 'tranformation', 'param_name', 'param_value',
                         'threshold', 'coverage_rate', 'covered_neurons', 'total_neurons', 'covered_detail'])

        # input: index = 0, 1, 2, 3, ... , 141

        # index = 0, 1
        if index / 2 == 0:
            if index % 2 == 0:  # index== 0
                input_images = range(200, 250)
            else:
                input_images = range(250, 300)

            # input_images = range(200, 300)


            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(data_path, file_list[j]))
                # cv2.imwrite("./img_save/0.jpg", seed_image)
                ndict, covered_neurons, total_neurons, pred_bbox = model.predict_fn(seed_image)

                # 准备打印数据
                tempk = []
                for k in ndict.keys():
                    if ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                # 清空神经元覆盖
                # model.nc_yolo.reset_cov_dict()

                csvrecord.append(j)  # index
                csvrecord.append(str(file_list[j]))  # image
                csvrecord.append('-')
                csvrecord.append('-')
                csvrecord.append('-')
                csvrecord.append(model.nactivated_threshold)
                csvrecord.append(str(round(100.0 * covered_neurons / total_neurons, 2)) + '%')
                csvrecord.append(covered_neurons)
                csvrecord.append(total_neurons)
                # csvrecord.append(covered_detail)


                print(csvrecord[:8])  # 输出到控制台
                # print(covered_neurons)
                # covered_neurons = nc.get_neuron_coverage(test_x)
                # print('input covered {} neurons'.format(covered_neurons))
                # print('total {} neurons'.format(total_neurons))
                writer.writerow(csvrecord)
            print("seed input done")

        # Translation 2 - 20
        if index / 2 >= 1 and index / 2 <= 10:
            # for p in range(1,11):
            p = index // 2
            params = [p * 10, p * 10]  # 图片平移参数（10， 10） -> (100, 100) stride = (10, 10)

            if index % 2 == 0:  # index== 0
                input_images = range(200, 250)
            else:
                input_images = range(250, 300)

            # input_images = range(200, 300)


            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(data_path, file_list[j]))
                seed_image = image_translation(seed_image, params)
                # cv2.imwrite("./img_save/0.jpg", seed_image)
                ndict, covered_neurons, total_neurons, pred_bbox = model.predict_fn(seed_image)

                # 准备打印数据
                tempk = []
                for k in ndict.keys():
                    if ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                # 清空神经元覆盖
                # model.nc_yolo.reset_cov_dict()

                csvrecord.append(j)  # index
                csvrecord.append(str(file_list[j]))  # image
                csvrecord.append('translation')
                csvrecord.append('x:y')
                csvrecord.append(':'.join(str(x) for x in params))
                csvrecord.append(model.nactivated_threshold)
                csvrecord.append(str(round(100.0 * covered_neurons / total_neurons, 2)) + '%')
                csvrecord.append(covered_neurons)
                csvrecord.append(total_neurons)
                # csvrecord.append(covered_detail)

                print(csvrecord[:8])
                # print(covered_neurons)
                # covered_neurons = nc.get_neuron_coverage(test_x)
                # print('input covered {} neurons'.format(covered_neurons))
                # print('total {} neurons'.format(total_neurons))
                writer.writerow(csvrecord)

        print("translation done")

        # Scale 22-40
        if index / 2 >= 11 and index / 2 <= 20:
            # for p in range(1,11):
            p = index // 2
            params = [p * 0.5 + 1, p * 0.5 + 1]  # 图片缩放参数

            if index % 2 == 0:  # index== 0
                input_images = range(200, 250)
            else:
                input_images = range(250, 300)

            # input_images = range(200, 300)

            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(data_path, file_list[j]))
                seed_image = image_scale(seed_image, params)
                # cv2.imwrite("./img_save/0.jpg", seed_image)
                ndict, covered_neurons, total_neurons, pred_bbox = model.predict_fn(seed_image)

                # 准备打印数据
                tempk = []
                for k in ndict.keys():
                    if ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                # 清空神经元覆盖
                model.nc_yolo.reset_cov_dict()

                csvrecord.append(j)  # index
                csvrecord.append(str(file_list[j]))  # image
                csvrecord.append('scale')
                csvrecord.append('x:y')
                csvrecord.append(':'.join(str(x) for x in params))
                csvrecord.append(model.nactivated_threshold)
                csvrecord.append(covered_neurons)
                csvrecord.append(total_neurons)
                csvrecord.append(covered_detail)

                print(csvrecord[:8])
                # print(covered_neurons)
                # covered_neurons = nc.get_neuron_coverage(test_x)
                # print('input covered {} neurons'.format(covered_neurons))
                # print('total {} neurons'.format(total_neurons))
                writer.writerow(csvrecord)

        print("scale done")

        # Shear 42-60
        if index / 2 >= 21 and index / 2 <= 30:
            # for p in range(1,11):
            p = index // 2 - 20
            # for p in range(1,11):
            params = 0.1 * p

            if index % 2 == 0:  # index== 0
                input_images = range(200, 250)
            else:
                input_images = range(250, 300)

            # input_images = range(200, 300)


            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(data_path, file_list[j]))
                seed_image = image_shear(seed_image, params)

                ndict, covered_neurons, total_neurons, pred_bbox = model.predict_fn(seed_image)

                # 准备打印数据
                tempk = []
                for k in ndict.keys():
                    if ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                # 清空神经元覆盖
                # model.nc_yolo.reset_cov_dict()

                csvrecord.append(j)  # index
                csvrecord.append(str(file_list[j]))  # image
                csvrecord.append('shear')
                csvrecord.append('factor')
                csvrecord.append(params)
                csvrecord.append(model.nactivated_threshold)
                csvrecord.append(str(round(100.0 * covered_neurons / total_neurons, 2)) + '%')
                csvrecord.append(covered_neurons)
                csvrecord.append(total_neurons)
                # csvrecord.append(covered_detail)

                print(csvrecord[:8])  # print(covered_neurons)
                # covered_neurons = nc.get_neuron_coverage(test_x)
                # print('input covered {} neurons'.format(covered_neurons))
                # print('total {} neurons'.format(total_neurons))

                writer.writerow(csvrecord)

        print("sheer done")

        # Rotation 62-80
        if index / 2 >= 31 and index / 2 <= 40:
            p = index // 2 - 30
            # for p in range(1,11):
            params = p * 3

            if index % 2 == 0:  # index== 0
                input_images = range(200, 250)
            else:
                input_images = range(250, 300)

            # input_images = range(200, 300)


            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(data_path, file_list[j]))
                seed_image = image_rotation(seed_image, params)

                ndict, covered_neurons, total_neurons, pred_bbox = model.predict_fn(seed_image)

                # 准备打印数据
                tempk = []
                for k in ndict.keys():
                    if ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                # 清空神经元覆盖
                # model.nc_yolo.reset_cov_dict()

                csvrecord.append(j)  # index
                csvrecord.append(str(file_list[j]))  # image
                csvrecord.append('rotation')
                csvrecord.append('angle')
                csvrecord.append(params)
                csvrecord.append(model.nactivated_threshold)
                csvrecord.append(str(round(100.0 * covered_neurons / total_neurons, 2)) + '%')
                csvrecord.append(covered_neurons)
                csvrecord.append(total_neurons)
                # csvrecord.append(covered_detail)

                print(csvrecord[:8])
                writer.writerow(csvrecord)

        print("rotation done")

        # Contrast 82-100
        if index / 2 >= 41 and index / 2 <= 50:
            p = index // 2 - 40
            # or p in range(1,11):
            params = 1 + p * 0.2

            if index % 2 == 0:  # index== 0
                input_images = range(200, 250)
            else:
                input_images = range(250, 300)

            # input_images = range(200, 300)

            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(data_path, file_list[j]))
                seed_image = image_contrast(seed_image, params)

                ndict, covered_neurons, total_neurons, pred_bbox = model.predict_fn(seed_image)

                # 准备打印数据
                tempk = []
                for k in ndict.keys():
                    if ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                # 清空神经元覆盖
                # model.nc_yolo.reset_cov_dict()

                csvrecord.append(j)  # index
                csvrecord.append(str(file_list[j]))  # image
                csvrecord.append('contrast')
                csvrecord.append('gain')
                csvrecord.append(params)
                csvrecord.append(model.nactivated_threshold)
                csvrecord.append(str(round(100.0 * covered_neurons / total_neurons, 2)) + '%')
                csvrecord.append(covered_neurons)
                csvrecord.append(total_neurons)
                # csvrecord.append(covered_detail)

                print(csvrecord[:8])
                writer.writerow(csvrecord)

        print("contrast done")

        # Brightness 102-120
        if index / 2 >= 51 and index / 2 <= 60:
            p = index // 2 - 50
            # for p in range(1,11):
            params = p * 10

            if index % 2 == 0:  # index== 0
                input_images = range(200, 250)
            else:
                input_images = range(250, 300)

            # input_images = range(200, 300)

            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(data_path, file_list[j]))
                seed_image = image_brightness(seed_image, params)

                ndict, covered_neurons, total_neurons, pred_bbox = model.predict_fn(seed_image)

                # 准备打印数据
                tempk = []
                for k in ndict.keys():
                    if ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                # 清空神经元覆盖
                # model.nc_yolo.reset_cov_dict()

                csvrecord.append(j)
                csvrecord.append(str(file_list[j]))
                csvrecord.append('brightness')
                csvrecord.append('bias')
                csvrecord.append(params)
                csvrecord.append(model.nactivated_threshold)
                csvrecord.append(str(round(100.0 * covered_neurons / total_neurons, 2)) + '%')
                csvrecord.append(covered_neurons)
                csvrecord.append(total_neurons)
                # csvrecord.append(covered_detail)

                print(csvrecord[:8])
                writer.writerow(csvrecord)

        print("brightness done")

        # blur 122-141
        if index / 2 >= 61 and index / 2 <= 70:
            p = index // 2 - 60
            # for p in range(1,11):
            params = p

            if index % 2 == 0:  # index== 0
                input_images = range(200, 250)
            else:
                input_images = range(250, 300)

            # input_images = range(200, 300)

            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(data_path, file_list[j]))
                seed_image = image_blur(seed_image, params)

                ndict, covered_neurons, total_neurons, pred_bbox = model.predict_fn(seed_image)

                # 准备打印数据
                tempk = []
                for k in ndict.keys():
                    if ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                # 清空神经元覆盖
                # model.nc_yolo.reset_cov_dict()

                param_name = ""
                if params == 1:
                    param_name = "averaging:3:3"
                if params == 2:
                    param_name = "averaging:4:4"
                if params == 3:
                    param_name = "averaging:5:5"
                if params == 4:
                    param_name = "GaussianBlur:3:3"
                if params == 5:
                    param_name = "GaussianBlur:5:5"
                if params == 6:
                    param_name = "GaussianBlur:7:7"
                if params == 7:
                    param_name = "medianBlur:3"
                if params == 8:
                    param_name = "medianBlur:5"
                if params == 9:
                    param_name = "averaging:6:6"
                if params == 10:
                    param_name = "bilateralFilter:9:75:75"

                csvrecord.append(j)
                csvrecord.append(str(file_list[j]))
                csvrecord.append('blur')
                csvrecord.append(param_name)
                csvrecord.append('-')
                csvrecord.append(model.nactivated_threshold)
                csvrecord.append(str(round(100.0 * covered_neurons / total_neurons, 2)) + '%')
                csvrecord.append(covered_neurons)
                csvrecord.append(total_neurons)
                # csvrecord.append(covered_detail)

                print(csvrecord[:8])
                writer.writerow(csvrecord)

        print("all done")


if __name__ == '__main__':
    # 单张图片检测和神经元覆盖率计算
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--img_dir', type=str, default='../data/voc/train/VOCdevkit/VOC2012/JPEGImages/2007_000423.jpg',
    #                     help='detect image dir')
    # parser.add_argument('--class_name_dir', type=str, default='../data/voc/train/VOCdevkit/VOC2012/voc2012.names',
    #                     help='classes name dir')
    # parser.add_argument('--model_dir', type=str, default='../weights/yolov5', help='saved pb model dir')
    # parser.add_argument('--img_size', type=int, default=640, help='image target size')
    # parser.add_argument('--conf_threshold', type=float, default=0.4, help='filter confidence threshold')
    # parser.add_argument('--iou_threshold', type=float, default=0.3, help='nms iou threshold')
    # parser.add_argument('--nactivated_threshold', type=float, default=0.3, help='neuron activated threshold')
    # opt = parser.parse_args()
    #
    # yolo_model = YoloV5Model(opt.model_dir, "../weights", opt.class_name_dir, opt.conf_threshold, opt.iou_threshold,
    #                          opt.nactivated_threshold)
    # img = cv2.imread(opt.img_dir)
    #
    # yolo_model.only_predict(img, opt.img_size)
    # ndict, covered_neurons, total_neurons, pred_bbox = yolo_model.predict_fn(img, opt.img_size)
    # print("covered_neurons:", covered_neurons)
    # print("total_neurons:", total_neurons)

    # 实验
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../data/voc/train/VOCdevkit/VOC2012/JPEGImages',
                        help='path for dataset')
    parser.add_argument('--index', type=int, default=110,
                        help='different indice mapped to different transformations and params')
    args = parser.parse_args()

    yolo_coverage(args.index, args.dataset)
