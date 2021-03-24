import argparse
import os
from time import sleep
import matplotlib.pyplot as plt

import pgdrive
import random
import h5py
import cv2
import numpy as np
from keyboard import is_pressed


def collect(args):
    is_auto = args.collection_method == "a"

    ### Create environment to drive in ###
    MAP_PIECES = 7
    env = pgdrive.PGDriveEnv(config=dict(
        environment_num=2000,
        load_map_from_json=False,
        map=MAP_PIECES,
        # Define predefine map look: https://pgdrive.readthedocs.io/en/latest/env_config.html#map-config
        # These are the most important ones. 'c' is new in used fork of PGDrive
        # S = rovne, r = najezd, c = mala zatacka, R = sjezd
        start_seed=random.randint(0, 1000000),
        use_render=True,
        use_image=True,

        traffic_density=0.0,
        traffic_mode="reborn",
        random_traffic=True,

        controller="keyboard" if args.collection_method == "k" else "joystick",
        manual_control=args.collection_method != "a",
        use_chase_camera=True,

        rgb_clip=False,
    ))

    def encodeKeys():
        return is_pressed("q"), is_pressed("e"), is_pressed("x")

    def close_figure():
        plt.close()

    ### Open dataset file for writing collected data ###
    os.makedirs(os.path.dirname(args.data_file), exist_ok=True)
    with h5py.File(args.data_file, "a") as h5:
        ### Create images and action datasets in .h5 file ###
        img_size = 0
        out_size = 0
        if "images" not in h5.keys():
            # rides, steps, 160x56px RGB
            # - steps=7*80=560
            img_dataset = h5.create_dataset("images",
                                            shape=(0, MAP_PIECES*80, 84, 160, 3),
                                            maxshape=(None, MAP_PIECES*80, 84, 160, 3), dtype="uint8")
        else:
            img_dataset = h5["images"]
            img_size = img_dataset.shape[0]

        if "action" not in h5.keys():
            # rides, steps, (velocity, steering, accel, should_turn_left, should_turn_right, stripped_line, contains_data)
            # Collecting as much data as we can as it may be needed in future.
            # - contains_data -> is 0, this is just a filler data
            # - steps=7*80=560
            out_dataset = h5.create_dataset("action",
                                            shape=(0, MAP_PIECES*80, 7),
                                            maxshape=(None, MAP_PIECES*80, 7), dtype="float64")
        else:
            out_dataset = h5["action"]
            out_size = out_dataset.shape[0]

        assert img_size == out_size, "Can't append to dataset! It has mismatched count of images and rides data."

        current_step_collection = img_dataset.shape[0]
        print(f"Already collected rides: {current_step_collection}/{args.rides_count}")
        if current_step_collection >= args.rides_count:
            print("Already collected enough rides.")
            return

        ### Start data collection ###
        env.reset()
        images = []
        outputs = []
        obs, reward, done, info = env.step([0, 1])  # No steer, speed 100%
        # obs = {image: (camera_data), state: (lidar_data)}
        if is_auto:
            current_line = 1  # 0=left, 1=middle, 2=right
            latest_position = -0.5  # Left line
            turn_key_down = False
            last_lane_change = 50

        ride_step = 0
        while True:
            ride_step += 1

            image = obs["image"]  # 160x120px
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image = cv2.flip(image, 1)
            image = image[36:, :]  # 160x84px

            velocity = info["velocity"]  # m/s ?
            steering = info["steering"]  # -60deg to 60deg -> -1 to 1
            acceleration = info["acceleration"]  # m/s^2 ?
            stripped_line = info["stripped_line"]

            if args.debug_camera:
                plt.imshow(image)
                plt.show()

            lateral_to_left = obs["state"][0]  # distance to left: 0 to 1
            lateral_to_right = obs["state"][1]  # distance to left: 0 to 1
            position_from_center_line = lateral_to_left - lateral_to_right  # Distance to center: -1 to 1
            if args.debug:
                position_from_left_line = position_from_center_line + 0.5
                position_from_right_line = position_from_center_line - 0.5
                print(f"Position from center: {position_from_center_line}")
                print(f"Position from left: {position_from_left_line}")
                print(f"Position from right: {position_from_right_line}")

            # Get keyboard input
            should_turn_left, should_turn_right = False, False
            if is_auto:
                _, _, should_exit = encodeKeys()
            else:  # is_manual
                should_turn_left, should_turn_right, should_exit = encodeKeys()

            if should_exit:
                print("[INFO] Collecting exited by user (pressing X)")
                break  # On pressing "x", exit this so the .h5 file will get closed correctly

            if is_auto:
                # Automatic switching lines
                if ride_step > last_lane_change + random.randint(50, 200):  # 5 to 20 seconds before line changes
                    if current_line == 0:
                        should_turn_right = True
                    elif current_line == 2:
                        should_turn_left = True

                    # If on center line decide randomly
                    elif random.random() > 0.5:
                        should_turn_right = True
                    else:
                        should_turn_left = True

                    last_lane_change = ride_step

                # Switching lines by should_turn_left, should_turn_right
                if not turn_key_down:
                    if should_turn_left and current_line > 0:
                        current_line -= 1
                        turn_key_down = True
                    if should_turn_right and current_line < 2:
                        current_line += 1
                        turn_key_down = True
                if not should_turn_left and not should_turn_right:
                    turn_key_down = False

            if args.debug:
                if should_turn_left:
                    print("Turning left!")
                if should_turn_right:
                    print("Turning right!")

            images.append(image)
            outputs.append([velocity, steering, acceleration, should_turn_left, should_turn_right, stripped_line, 1])

            env.render()

            ### Process drive
            if done:
                ### Drive completed
                if info["arrive_dest"]:
                    # Drive successful -> save
                    images = images[50:-50]    # First and last 5 seconds are not good training representation
                    outputs = outputs[50:-50]  # First and last 5 seconds are not good training representation
                    print(f"Collected {len(images)} images. Keeping {MAP_PIECES*80}.")
                    if len(images) < MAP_PIECES*80:
                        print("Not enough images! Filling with blank space! Dont worry, this is normal.")
                        for i in range(MAP_PIECES*80 - len(images)):
                            images.append(np.zeros((84, 160, 3)))
                            outputs.append(np.zeros((7,)))

                    # Save to H5
                    imgs_len = len(images)
                    imgs = images[: MAP_PIECES*80]
                    outs = outputs[: MAP_PIECES*80]

                    img_dataset.resize(img_dataset.shape[0]+1, axis=0)
                    img_dataset[img_dataset.shape[0]-1] = np.array(imgs)

                    out_dataset.resize(out_dataset.shape[0]+1, axis=0)
                    out_dataset[out_dataset.shape[0]-1] = np.array(outs)

                    print(f"Saved collection ({img_dataset.shape[0]}).")

                    current_step_collection = img_dataset.shape[0]
                    print(f"Rides collected: {current_step_collection}/{args.rides_count}")
                    if current_step_collection >= args.rides_count:
                        print("Collected enough rides, exiting.")
                        break

                else:
                    # Car crashed!
                    print("Crash or Out of road!")

                # Reset env
                env.reset()
                images = []
                outputs = []
                ride_step = 0
                obs, reward, done, info = env.step([0, 1])  # No steer, speed 100%

                if is_auto:
                    current_line = 1  # Middle line
                    latest_position = 0  # Middle line
                    turn_key_down = False
                    last_lane_change = 50
                print()

            else:
                ### RIDE
                if is_auto:  # Ride auto
                    offset = position_from_center_line   # Center line -> no change
                    if current_line == 0: offset += 0.5  #   Left line -> +0.5
                    if current_line == 2: offset -= 0.5  #  Right line -> -0.5
                    if args.debug: print(f"Offset: {offset}")

                    current_steer_speed = position_from_center_line - latest_position
                    if args.debug: print(f"Steer Speed: {current_steer_speed}")

                    target_steer_speed = -(offset / 15)
                    if abs(target_steer_speed) < 0.02:
                        target_steer_speed *= 2
                    if args.debug: print(f"target_steer_speed: {target_steer_speed}")

                    steer = -(target_steer_speed - current_steer_speed)
                    steer += random.randrange(-100, 100, 1) / 100 * 0.01
                    if args.debug: print(f"Steer: {steer}")

                    obs, reward, done, info = env.step([steer, 1])  # No steer, speed 100%
                    latest_position = position_from_center_line

                    if args.debug: print()

                else:  # Ride manual
                    obs, reward, done, info = env.step([0, 1])  # Can be random, as real input is from user


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Data collection for Sofia experiment. To exit when running press X.')

    # --- #
    parser.add_argument('-m', dest='collection_method', type=str, default='a',
                              help='Method of data collection'
                                   ' (a=automatic, j=manual_using_ps4_controller, k=manual_using_keyboard)')
    parser.add_argument('-d', dest='data_file', type=str, default='datasets/collected_data.h5',
                              help='Where to save collected data. Has to be .h5 file.'
                                   ' Can be already existing dataset, as it will append to it.')
    parser.add_argument('-c', dest='rides_count', type=int, default=250,
                              help='How many rides to drive and store in the database before done.')
    parser.add_argument('-v', dest='debug', action='store_true',
                              help='Run data collection in verbose mode. -> Printing a lot of data.')
    parser.add_argument('-vc', dest='debug_camera', action='store_true',
                               help='Run data collection in camera verbose mode. -> Show each camera image captured.')
    # --- #

    args = parser.parse_args()
    print("\n[INFO] To exit press X, so the dataset file gets saved correctly.\n")
    sleep(3)
    collect(args)
    print("[DONE]")


if __name__ == '__main__':
    main()
