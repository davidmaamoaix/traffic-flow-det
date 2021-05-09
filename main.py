import os
import pathlib
import argparse

import pipeline


if __name__ == '__main__':

    # some simple cmd-line interface
    parser = argparse.ArgumentParser(
        description='Pipelines for traffic flow detection'
    )
    parser.add_argument('video', type=str, help='path of the input video')
    args = parser.parse_args()

    pipeline.run(args.video)