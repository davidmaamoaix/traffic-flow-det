import os
import cv2
import numpy as np

import config
from misc import output_stream, load_yolo


def run(video_path):

    reader = cv2.VideoCapture(video_path)
    writer = output_stream(reader)

    if not os.path.exists('output'):
        os.mkdir('output')

    model = load_yolo(
        config.YOLO_WEIGHTS,
        config.YOLO_CONFIG,
        config.YOLO_CLASSES,
        {}
    )

    session = Session(model)

    while True:
        ret, frame = reader.read()

        if not ret:
            break

        process_frame(frame, session)

    reader.release()
    writer.release()


def process_frame(frame, session):
    pass


class Session:

    def __init__(self, model):
        self.model = model