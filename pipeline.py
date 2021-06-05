import os
import cv2
import numpy as np
from PIL import Image

import config
from misc import output_stream, load_yolo, reverse_projection, distance
from tools import generate_detections
from deep_sort import nn_matching, tracker, detection
from filter import FilterPID

MARKERS = (
    (689, 412),
    (766, 408),
    (713, 517),
    (836, 512)
)

ROAD_WIDTH = distance(MARKERS[0], MARKERS[1])
print(reverse_projection(10, 10, MARKERS))


def run(video_path):

    reader = cv2.VideoCapture(video_path)
    writer = output_stream(reader)
    fps = reader.get(cv2.CAP_PROP_FPS)

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

    conf = {
        'show_img': True,
        'fps': fps
    }
    session = Session(model, encoder, tracking, writer, conf)

    cv2.imwrite("frame1.jpg", reader.read()[1])

    while True:
        ret, frame = reader.read()

        if not ret:
            break

        process_frame(frame, session)

        if session.conf.get('show_img', False) and cv2.waitKey(1) & 0xFF == 27:
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

        if tracked.track_id not in session.speed:
            controller = FilterPID(0.52, 0.103, 50)
            session.speed[tracked.track_id] = (x_world, y_world, controller)

        x_prev, y_prev, pid_controller = session.speed[tracked.track_id]

        session.speed[tracked.track_id] = x_world, y_world, pid_controller

        dist = distance((x_prev, y_prev), (x_world, y_world))
        prev_frame = session.last_frame.get(
            tracked.track_id,
            session.counter - 1
        )
        delta_time = session.counter - prev_frame

        color = (255, 0, 0)

        speed = 3.6 * dist * session.conf.get('fps', 30) / delta_time / 500
        speed = pid_controller.update(speed)

        cv2.putText(frame, str(speed), (x + 30, y), 0, 0.5, color, 1)

    if session.conf.get('show_img', False):
        cv2.imshow('img', frame)

    session.counter += 1
    session.writer.write(frame)


class Session:

    def __init__(self, model, encoder, tracking, writer, conf):
        self.model = model
        self.encoder = encoder
        self.tracking = tracking
        self.speed = {}
        self.writer = writer
        self.conf = conf
        self.last_frame = {}
        self.filter = {} # used for PID
        self.counter = 0