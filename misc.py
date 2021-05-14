import os
import cv2
import datetime
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

        blob = cv2.dnn.blobFromImage(
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
        filtered = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            output[:, 4].tolist(),
            self.confidence_thres,
            0.2 # non max suppresion threshold
        )

        indices = np.array(filtered).flatten()

        if indices.size == 0:
            return np.array([]), np.array([])

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
    network = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

    return YoloWrapper(network, {**config, 'classes': classes})


def output_stream(reader):
    '''returns a VideoWriter for easier writing'''

    fps = reader.get(cv2.CAP_PROP_FPS)
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_name = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S.avi")

    return cv2.VideoWriter(
        os.path.join('output/', output_name),
        cv2.VideoWriter_fourcc(*'MJPG'),
        fps,
        (width, height)
    )


def distance(a, b):
    delta_x, delta_y = abs(b[0] - a[0]), abs(b[1] - a[1])
    return (delta_x ** 2 + delta_y ** 2) ** 0.5


def reverse_projection(x_coord, y_coord, marker):
    w = 3600

    xA, yA = marker[0]
    xB, yB = marker[1]
    xC, yC = marker[2]
    xD, yD = marker[3]

    aAB = xB - xA
    bAB = yB - yA
    xAB = xA * yB - xB * yA

    aAC = xC - xA
    bAC = yC - yA
    xAC = xA * yC - xC * yA

    aBD = xD - xB
    bBD = yD - yB
    xBD = xB * yD - xD * yB

    aCD = xD - xC
    bCD = yD - yC
    xCD = xC * yD - xD * yC

    tan_s_molecule = -bAB * bAC * xBD * aCD + bAC * aBD * bAB * xCD + \
                     bCD * xAB * bBD * aAC - bAB * xCD * bBD * aAC - \
                     bCD * bBD * xAC * aAB - bAC * xAB * aBD * bCD + \
                     bAB * xAC * bBD * aCD + bCD * bAC * xBD * aAB

    tan_s_denominator = -bAB * xAC * aBD * aCD + bAC * xAB * aBD * aCD - \
                        bAC * aBD * aAB * xCD - aAC * xBD * bCD * aAB - \
                        aCD * xAB * bBD * aAC + bAB * aAC * xBD * aCD + \
                        aAB * xCD * bBD * aAC + aBD * xAC * bCD * aAB

    tan_s = tan_s_molecule / tan_s_denominator
    sin_s = tan_s * ((1 / (tan_s ** 2 + 1)) ** 0.5)
    cos_s = (1 / (tan_s ** 2 + 1)) ** 0.5

    sin_t_molecule = ((aBD * xAC - aAC * xBD) * sin_s + \
                     (bBD * xAC - bAC * xBD) * cos_s) * \
                     ((aCD * xAB - aAB * xCD) * sin_s + \
                     (bCD * xAB - bAB * xCD) * cos_s)

    sin_t_denominator = ((aCD * xAB - aAB * xCD) * cos_s + \
                        (bAB * xCD - bCD * xAB) * sin_s) * \
                        ((bBD * xAC - bAC * xBD) * sin_s + \
                        (aAC * xBD - aBD * xAC) * cos_s)

    sin_t = -((sin_t_molecule / sin_t_denominator) ** 0.5)

    cos_t = (1 - sin_t ** 2) ** 0.5
    tan_p_molecule = sin_t * ((bBD * xAC - bAC * xBD) * sin_s + \
                     (aAC * xBD - aBD * xAC) * cos_s)

    tan_p_denominator = (aBD * xAC - aAC * xBD) * sin_s + \
                        (bBD * xAC - bAC * xBD) * cos_s

    tan_p = tan_p_molecule / tan_p_denominator

    sin_p = tan_p * ((1 / (tan_p ** 2 + 1)) ** 0.5)
    cos_p = ((1 / (tan_p ** 2 + 1)) ** 0.5)

    f_molecule = xBD * cos_p * cos_t
    f_denominator = bBD * sin_p * cos_s - bBD * cos_p * sin_t * sin_s + \
                    aBD * sin_p * sin_s + aBD * cos_p * sin_t * cos_s

    f = abs(f_molecule / f_denominator)

    l_molecule = w * (f * sin_t + xA * cos_t * sin_s + yA * cos_t * cos_s) * \
                 (f * sin_t + xC * cos_t * sin_s + yC * cos_t * cos_s)

    l_denominator = -(f * sin_t + xA * cos_t * sin_s + yA * cos_t * cos_s) * \
                    (xC * cos_p * sin_s - xC * sin_p * sin_t * cos_s + \
                    yC * cos_p * cos_s + yC * sin_p * sin_t * sin_s) + \
                    (f * sin_t + xC * cos_t * sin_s + yC * cos_t * cos_s) * \
                    (xA * cos_p * sin_s - xA * sin_p * sin_t * cos_s + \
                    yA * cos_p * cos_s + yA * sin_p * sin_t * sin_s)

    l = abs(l_molecule / l_denominator)

    xQ_molecule = l * sin_p * (x_coord * sin_s + y_coord * cos_s) + \
                  l * sin_t * cos_p * \
                  (x_coord * cos_s - y_coord * sin_s)

    xQ_denominator = x_coord * cos_t * sin_s + \
                     y_coord * cos_t * cos_s + f * sin_t

    xQ = xQ_molecule / xQ_denominator

    yQ_molecule = -l * cos_p * (x_coord * sin_s + y_coord * cos_s) \
                  + l * sin_p * sin_t * (x_coord * cos_s - y_coord * sin_s)

    yQ_denominator = x_coord * cos_t * sin_s + \
                     y_coord * cos_t * cos_s + f * sin_t

    yQ = yQ_molecule / yQ_denominator

    return xQ, yQ