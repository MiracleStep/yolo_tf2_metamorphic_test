import tensorflow as tf
if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    print(tf.__version__)
    print(tf.test.is_gpu_available())