import os
import pathlib
import argparse

import pipeline


if __name__ == '__main__':

    # some simple cmd-line interface
    parser = argparse.ArgumentParser(
        description='Pipelines for traffic flow detection'
    )
    parser.add_argument('video', type=str, help='Path of the input video')
    parser.add_argument('pipeline', type=int, help='ID of the pipeline to use')
    args = parser.parse_args()

    pipeline = args.pipeline
    pipeline.run(args.video)