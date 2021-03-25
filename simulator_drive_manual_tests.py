import argparse
import os
from time import sleep

import pgdrive  # Import this package to register the environment!
import random
import cv2
import numpy as np
from keyboard import is_pressed
from tensorflow import keras

### Fix "Could not create cudnn handle: CUDNN_STATUS_NOT_INITIALIZED" if something else is also using GPU memory ###
import tensorflow as tf

from utils import get_model, model_predict

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
######


def test_drive(args):
    ### Build environment ###
    MAP_PIECES = 7
    env = pgdrive.PGDriveEnv(config=dict(
        environment_num=20,
        load_map_from_json=False,
        map=MAP_PIECES,  # Define custom map look: https://pgdrive.readthedocs.io/en/latest/env_config.html#map-config
        # S = rovne, r = najezd, c = mala zatacka, R = sjezd
        start_seed=random.randint(0, 1000000),
        use_render=True,
        use_image=True,

        traffic_density=0.0,
        traffic_mode="reborn",
        random_traffic=True,

        use_chase_camera=True,
        rgb_clip=False,
    ))

    ### load model ###
    model_name = os.path.basename(args.model_path).replace(".h5", "")
    model_folder = os.path.dirname(args.model_path)
    model = get_model(model_name, model_folder)


    def encodeKeys():
        return is_pressed("q"), is_pressed("e"), is_pressed("x")
        # keys = key_check()
        # return "Q" in keys, "E" in keys, "X" in keys

    ### Drive ###
    env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())  # Use random policy to get first image
    # obs = {image: (camera_data), state: (lidar_data)}
    while True:
        image = obs["image"]  # 160x120px

        # Collect for keyboard
        should_turn_left, should_turn_right, should_exit = encodeKeys()
        if should_exit:
            print("[INFO] Exiting")
            break

        if args.debug:
            if should_turn_left:
                print("Should turn Left")
            if should_turn_right:
                print("Should turn Right")

        pred = model_predict(model, image, should_turn_left, should_turn_right)
        if args.debug: print("pred: ", pred)

        obs, reward, done, info = env.step(np.array([pred, 1]))  # Use prediction

        env.render()
        if done:
            env.reset()
            model.reset_states()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simulator drive manual tests for trained models.')

    # --- #
    parser.add_argument('-m', dest='model_path', type=str, required=True,
                              help='Path to the model')
    parser.add_argument('-v', dest='debug', action='store_true',
                              help='Run drive test in verbose mode. -> Printing a lot of data.')
    # --- #

    args = parser.parse_args()
    assert os.path.isfile(args.model_path), "[ERROR] Model file does not exist!"

    print("\n[INFO] To exit press X, so the dataset file gets saved correctly.")
    print("[INFO] To go left press Q, to go right press E.\n")
    sleep(5)
    test_drive(args)
    print("[DONE]")


if __name__ == '__main__':
    main()
