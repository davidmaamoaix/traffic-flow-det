import os
import cv2
import numpy as np

import config
from misc import output_stream, load_yolo, reverse_projection
from tools import generate_detections
from deep_sort import nn_matching, tracker, detection
from PIL import Image

MARKERS = (
    (689, 412),
    (766, 408),
    (713, 517),
    (836, 512)
)


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

    cv2.imwrite("frame1.jpg", reader.read()[1])

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

    #for box, label in zip(boxes, labels):
    #    x, y, w, h = box
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

    extracts = session.encoder(frame, boxes)

    # TODO: retain confidence from yolo
    session.tracking.predict()
    session.tracking.update([
        detection.Detection(box, 0.75, encoded)
        for box, encoded in zip(boxes, extracts)
    ])

    color = (255, 255, 255)
    for tracked in session.tracking.tracks:
        x, y, x_end, y_end = tuple(map(int, tracked.to_tlbr()))
        cv2.rectangle(frame, (x, y), (x_end, y_end), color, 1)
        cv2.putText(frame, str(tracked.track_id), (x, y), 0, 0.5, color, 1)

        x_world, y_world = reverse_projection(
            (x + x_end) / 2,
            (y + y_end) / 2,
            MARKERS
        )

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