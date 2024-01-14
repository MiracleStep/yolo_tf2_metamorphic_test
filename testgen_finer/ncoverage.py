from __future__ import print_function
import numpy as np
from keras.models import Model
from model import Yolo
import tensorflow as tf
# TODO：未完成神经元覆盖
class NCoverage():

    def __init__(self, model, threshold=0.1, exclude_layer=["input", 'pool', 'fc', 'flatten', "detect"], only_layer=""):
        '''
        Initialize the model to be tested
        :param threshold: threshold to determine if the neuron is activated
        :param model_name: ImageNet Model name, can have ('vgg16','vgg19','resnet50')
        :param neuron_layer: Only these layers are considered for neuron coverage
        '''
        self.threshold = float(threshold)

        self.model = model
        # tf.keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
        print('models loaded')
        # the layers that are considered in neuron coverage computation
        self.layer_to_compute = [] # 计算神经元覆盖的layers
        for layer in self.model.layers:
            if all(ex not in layer.name for ex in exclude_layer):
                self.layer_to_compute.append(layer.name)

        if only_layer != "":
            self.layer_to_compute = [only_layer]

        # init coverage table
        self.cov_dict = {} # 统计神经元是否覆盖字典，键为tuple形式 (所有参数神经元覆盖计算的layer名, layer输出的大小 从0计算)

        for layer_name in self.layer_to_compute:
            cur_layer = self.model.get_layer(layer_name)
            for index in range(cur_layer.output_shape[-1]):
                self.cov_dict[(layer_name, index)] = False

    def scale(self, layer_outputs, rmax=1, rmin=0):
        '''
        scale the intermediate layer's output between 0 and 1
        :param layer_outputs: the layer's output tensor
        :param rmax: the upper bound of scale
        :param rmin: the lower bound of scale
        :return:
        '''
        divider = (layer_outputs.max() - layer_outputs.min())
        if divider == 0:
            return np.zeros(shape=layer_outputs.shape)
        X_std = (layer_outputs - layer_outputs.min()) / divider
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled


    def set_covdict(self, covdict):
        self.cov_dict = dict(covdict)


    def update_coverage(self, input_data):
        '''
        Given the input, update the neuron covered in the model by this input.
            This includes mark the neurons covered by this input as "covered"
        :param input_data: the input image
        :return: the neurons that can be covered by the input
        '''
        for layer_name in self.layer_to_compute:
            layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)

            layer_outputs = layer_model.predict(input_data)

            for layer_output in layer_outputs:
                scaled = self.scale(layer_output)
                #print(scaled.shape)
                for neuron_idx in range(scaled.shape[-1]):
                    if np.mean(scaled[..., neuron_idx]) > self.threshold:
                        self.cov_dict[(layer_name, neuron_idx)] = True
            del layer_outputs
            del layer_model


        return self.cov_dict


    def curr_neuron_cov(self):
        '''
        Get current coverage information of MUT
        :return: number of covered neurons,
            number of total neurons,
            number of neuron coverage rate
        '''
        covered_neurons = len([v for v in self.cov_dict.values() if v])
        total_neurons = len(self.cov_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


    def activated(self, layer_name, idx, input_data):
        '''
        Test if the neuron is activated by this input。
        :param layer_name: the layer name
        :param idx: the neuron index in the layer
        :param input_data: the input
        :return: True/False
        '''
        layer_model = Model(inputs=self.model.input,
                            outputs=self.model.get_layer(layer_name).output)
        layer_output = layer_model.predict(input_data)[0]
        scaled = self.scale(layer_output)
        if np.mean(scaled[..., idx]) > self.threshold:
            return True
        return False

    def get_neuron_coverage(self, input_data):
        '''
        Given the input, return the covered neurons by the input data without marking the covered neurons.
        '''

        covered_neurons = []
        for layer_name in self.layer_to_compute:
            layer_model = Model(input=self.model.input,
                                output=self.model.get_layer(layer_name).output)
            layer_outputs = layer_model.predict(input_data)
            for layer_output in layer_outputs:
                scaled = self.scale(layer_output)
                for neuron_idx in range(scaled.shape[-1]):
                    if np.mean(scaled[..., neuron_idx]) > self.threshold:
                        if (layer_name, neuron_idx) not in covered_neurons:
                            covered_neurons.append((layer_name, neuron_idx))
        return covered_neurons # 覆盖的神经元的 名称+idx


    def is_testcase_increase_coverage(self, input_data):
        '''
        Given the input, check if new neuron is covered without marking the covered neurons.
        If a previously not covered neuron is covered by the input, return True.
        Otherwise return False.
        '''
        for layer_name in self.layer_to_compute:
            layer_model = Model(inputs=self.model.inputs,
                                outputs=self.model.get_layer(layer_name).output)

            layer_outputs = layer_model.predict(input_data)

            for layer_output in layer_outputs:
                scaled = self.scale(layer_output)
                #print(scaled.shape)
                for neuron_idx in range(scaled.shape[-1]):
                    if np.mean(scaled[..., neuron_idx]) > self.threshold:
                        if not self.cov_dict[(layer_name, neuron_idx)]:
                            return True
        return False

    def reset_cov_dict(self):
        '''
        Reset the coverage table
        :return:
        '''
        for layer_name in self.layer_to_compute:
            for index in range(self.model.get_layer(layer_name).output_shape[-1]):
                self.cov_dict[(layer_name, index)] = False