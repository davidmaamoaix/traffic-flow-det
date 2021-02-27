import os
import argparse


PIPELINES = [
    '0-threshold-detection'
]


if __name__ == '__main__':

    # some simple cmd-line interface
    parser = argparse.ArgumentParser(
        description='Pipelines for traffic flow detection'
    )
    parser.add_argument('pipeline', type=int, help='ID of the pipeline to use')
    args = parser.parse_args()

    pipeline = args.pipeline
