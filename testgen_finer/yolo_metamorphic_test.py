from yolo_coverage import YoloV5Model
import csv
import os
import argparse
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

def yolo_metamorphic(index, data_path):
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