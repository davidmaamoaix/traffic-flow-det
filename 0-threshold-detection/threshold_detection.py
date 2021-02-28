import os
import cv2 as cv
import numpy as np

from utils import load_yolo, output_stream


def process_video(video):

    reader = cv.VideoCapture(video)
    writer = output_stream(reader)

    if not os.path.exists('output'):
        os.mkdir('output')

    model = load_yolo(
        'models/yolov3.weights',
        'models/yolov3_coco.cfg',
        'models/coco.names',
        {}
    )

    while True:
        ret, frame = reader.read()

        if not ret:
            break

        boxes, labels = model.forward(frame)

        for box, label in zip(boxes, labels):
            x, y, w, h = box

            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

        writer.write(frame)

    reader.release()
    writer.release()