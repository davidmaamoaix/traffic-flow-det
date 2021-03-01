import os
import datetime
import cv2 as cv
import numpy as np


__all__ = [
	'load_yolo',
    'output_stream'
]


class YoloWrapper:
    '''A wrapper class for the yolo model (bundled with NMS)'''

    def __init__(self, network, config):
        '''
        network, the yolo net loaded from Darknet
        config: dict, see 'load_yolo' method
        '''
        self.network = network
        self.classes = config['classes']

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
        output = np.vstack(layered_output)
        predictions = np.argmax(output[:, 5:], axis=1)

        height, width, _ = image.shape
        boxes = np.zeros((predictions.shape[0], 4), dtype=np.int)

        # since all points are in percentage of the image size,
        # scaling by the image size is done to obtain the absolute value

        # x and y
        boxes[:, 0] = (output[:, 0] - output[:, 2] / 2) * width
        boxes[:, 1] = (output[:, 1] - output[:, 3] / 2) * height

        # width and height
        boxes[:, 2] = output[:, 2] * width
        boxes[:, 3] = output[:, 3] * height

        # filtered is an array of indices of the valid boxes
        filtered = cv.dnn.NMSBoxes(
            boxes.tolist(),
            output[:, 4].tolist(),
            self.confidence_thres,
            0.2 # non max suppresion threshold
        )

        indices = np.array(filtered).flatten()
        valid_boxes = boxes[indices]
        output_classes = self.classes[predictions[indices]]

        return valid_boxes, output_classes


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
        # for ease of indexing
        classes = np.array([i.strip() for i in f.readlines()])

    # network setup
    network = cv.dnn.readNetFromDarknet(cfg_path, weights_path)

    return YoloWrapper(network, {**config, 'classes': classes})


def output_stream(reader):
    '''returns a VideoWriter for easier writing'''

    fps = reader.get(cv.CAP_PROP_FPS)
    width = int(reader.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv.CAP_PROP_FRAME_HEIGHT))

    output_name = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S.avi")

    return cv.VideoWriter(
        os.path.join('output/', output_name),
        cv.VideoWriter_fourcc(*'MJPG'),
        fps,
        (width, height)
    )