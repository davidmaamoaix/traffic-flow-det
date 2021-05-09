import os
import cv2
import numpy as np

import config
from misc import output_stream, load_yolo
from tools import generate_detections
from deep_sort import nn_matching, tracker, detection


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

    metric = nn_matching.NearestNeighborDistanceMetric('cosine', 0.3)
    tracking = tracker.Tracker(metric)
    encoder = generate_detections.create_box_encoder(
        config.ENCODER_PATH,
        batch_size=1
    )

    session = Session(model, encoder, tracking, writer, True)

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

    extracts = session.encoder(frame, boxes)

    # TODO: retain confidence from yolo
    session.tracking.predict()
    session.tracking.update([
        detection.Detection(box, 0.75, encoded)
        for box, encoded in zip(boxes, extracts)
    ])

    if session.show_img:
        cv2.imshow('img', frame)

    session.writer.write(frame)


class Session:

    def __init__(self, model, encoder, tracking, writer, show_img=False):
        self.model = model
        self.encoder = encoder
        self.tracking = tracking
        self.writer = writer
        self.show_img = show_img