import argparse

from testgen_finer.yolo_coverage import yolo_coverage

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='VOCdevkit/VOC2007/JPEGImages',
                        help='path for dataset')
    parser.add_argument('--index', type=int, default=110,
                        help='different indice mapped to different transformations and params')
    args = parser.parse_args()

    yolo_coverage(args.index, args.dataset)