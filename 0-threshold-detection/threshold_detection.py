import os
import cv2 as cv
import numpy as np
import datetime

from utils import load_yolo


def process_video(video):

    reader = cv.VideoCapture(video)
    fps = reader.get(cv.CAP_PROP_FPS)
    width = int(reader.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv.CAP_PROP_FRAME_HEIGHT))

    output_name = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S.avi")
    writer = cv.VideoWriter(
        os.path.join('output/', output_name),
        cv.VideoWriter_fourcc(*'MJPG'),
        fps,
        (width, height)
    )

    model = load_yolo(
        'models/yolov3.weights',
        'models/yolov3_coco.cfg',
        'models/coco.names',
        {} # default is good
    )

    while True:

        ret, frame = reader.read()

        if not ret:
            break

        writer.write(frame)

    reader.release()
    writer.release()