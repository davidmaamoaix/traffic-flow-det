import cv2 as cv
import numpy as np

__all__ = [
	'load_yolo'
]


class YoloWrapper:
    '''A wrapper class for the yolo model (bundled with NMS)'''

    def __init__(self, network, config):
        '''
        network, the yolo net loaded from Darknet
        config: dict, see 'load_yolo' method
        '''
        self.network = network

        self.img_size = config.get('image_size', 416)
        self.confidence_thres = config.get('confidence_thres', 0.75)
        self.out_names = config.get(
            'out_layer_names',
            ['yolo_82', 'yolo_94', 'yolo_106']
        )

    def forward(self, image):
        '''Forward pass of YOLO + NMS'''

        blob = cv.dnn.blobFromImage(
            image=image,
            scalefactor=1/255, # int: [0, 255] to float: [0, 1]
            size=(self.img_size, self.img_size),
            swapRB=True # swap R and B channels since OpenCV uses BGR
        )
        self.network.setInput(blob)

        layered_output = self.network.forward(self.out_names)
        print(layered_output)
        output = np.vstack(layered_output)


def load_yolo(weights_path, cfg_path, names_path, config):
    '''
    *_path: str, the path to the file

    config: dict, containing the following keys (with defaults):
        image_size (416): the size of the model's input
        confidence_thres (0.75): the minimal threshold for a valid prediction
        out_layer_names ([...]): name of output layers
    '''

    # read object classes
    with open(names_path, 'r') as f:
        classes = [i.strip() for i in f.readlines()]

    # network setup
    network = cv.dnn.readNetFromDarknet(cfg_path, weights_path)

    return YoloWrapper(network, {**config, classes: classes})