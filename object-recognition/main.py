import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model-path',
        dest='model_path',
        type=str,
        help='Path to the model image that you are trying to find in the target image.'
    )
    parser.add_argument(
        '-t',
        '--target-path',
        dest='target_path',
        type=str,
        help='Path of the target image in which you are trying to find an object.'
    )