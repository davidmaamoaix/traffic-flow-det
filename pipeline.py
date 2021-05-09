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

    session = Session(model, writer, True)

    while True:
        ret, frame = reader.read()

        if not ret:
            break

        process_frame(frame, session)

        if session.show_img and cv2.waitKey(1) & 0xFF == 27:
            break

    reader.release()
    writer.release()


def process_frame(frame, session):
    boxes, labels = session.model.forward(frame)

    for box, label in zip(boxes, labels):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

    if session.show_img:
        cv2.imshow('img', frame)

    session.writer.write(frame)


class Session:

    def __init__(self, model, writer, show_img=False):
        self.model = model
        self.writer = writer
        self.show_img = show_img