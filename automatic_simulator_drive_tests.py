import argparse
import os
import time
import pgdrive
import random
import cv2
import numpy as np
from tensorflow.python.keras import Model
from utils import get_model


### Fix "Could not create cudnn handle: CUDNN_STATUS_NOT_INITIALIZED" if something else is also using GPU memory ###
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
######


estimate_start = 0
def get_estimated_time(start_done: float, end_done: float, total: float) -> float:
    global estimate_start
    time_used_this_batch = time.time() - estimate_start
    done_this_batch = end_done - start_done
    if done_this_batch > 0:
        time_to_complete_one = time_used_this_batch / done_this_batch

        remaining_amount = total - end_done
        remaining_time = time_to_complete_one * remaining_amount
    else:
        remaining_time = 999999999

    estimate_start = time.time()
    return remaining_time

out_of_road = 0
wrong_line = 0
Lines = {"LEFT": 0, "CENTER": 1, "RIGHT": 2}
already_done = 0
def test_model(args):
    global out_of_road
    global wrong_line

    model_name = os.path.basename(args.model_path).replace(".h5", "")
    model_folder = os.path.dirname(args.model_path)
    # rnd = random.Random(1)   # We want random numbers identical for all models
    model = get_model(model_name, model_folder)

    ### Create environment ###
    env = pgdrive.PGDriveEnv(config=dict(
        environment_num=2000,
        load_map_from_json=False,
        map=6,
        start_seed=1,  # We want all the maps to be the same for all models
        use_render=True,
        use_image=True,
        traffic_density=0.0,
        traffic_mode="reborn",
        random_traffic=True,
        use_chase_camera=True,
        rgb_clip=False,
    ))

    ### Drive ###
    rides_per_test = args.rides_count // 15
    if args.debug: print(f"[DEBUG] rides_per_test: {rides_per_test}")

    def do_test(start_line: int, target_line: int, should_turn_left_for: int, should_turn_right_for: int):
        global out_of_road
        global wrong_line
        crashes, bad_lines = test_drive(env, model, start_line, target_line,
                                        should_turn_left_for, should_turn_right_for, rides_per_test, args.debug)
        out_of_road += crashes
        wrong_line += bad_lines


    # Turning - 1 step
    if args.debug: print(f"[DEBUG] Testing turning - 1 step")
    out_of_road, wrong_line = 0, 0
    do_test(Lines["RIGHT"], Lines["CENTER"], 1, 0)
    do_test(Lines["CENTER"], Lines["LEFT"], 1, 0)
    do_test(Lines["CENTER"], Lines["RIGHT"], 0, 1)
    do_test(Lines["LEFT"], Lines["CENTER"], 0, 1)
    one_step_order = wrong_line

    # Turning - 3 step
    if args.debug: print(f"[DEBUG] Testing turning - 3 step")
    out_of_road, wrong_line = 0, 0
    do_test(Lines["RIGHT"], Lines["CENTER"], 3, 0)
    do_test(Lines["CENTER"], Lines["LEFT"], 3, 0)
    do_test(Lines["CENTER"], Lines["RIGHT"], 0, 3)
    do_test(Lines["LEFT"], Lines["CENTER"], 0, 3)
    three_step_order = wrong_line

    # Turning - 5 step
    if args.debug: print(f"[DEBUG] Testing turning - 5 step")
    wrong_line = 0
    do_test(Lines["RIGHT"], Lines["CENTER"], 5, 0)
    do_test(Lines["CENTER"], Lines["LEFT"], 5, 0)
    do_test(Lines["CENTER"], Lines["RIGHT"], 0, 5)
    do_test(Lines["LEFT"], Lines["CENTER"], 0, 5)
    five_step_order = wrong_line

    # Straight
    if args.debug: print(f"[DEBUG] Testing straight drive")
    wrong_line = 0
    do_test(Lines["RIGHT"], Lines["RIGHT"], 0, 0)
    do_test(Lines["CENTER"], Lines["CENTER"], 0, 0)
    do_test(Lines["LEFT"], Lines["LEFT"], 0, 0)
    line_changes_without_order = wrong_line


    ### Print and save results ###
    results = f"{out_of_road} / {line_changes_without_order} / {one_step_order} / {three_step_order} / {five_step_order}"
    print(f"{model_name} - {results}")

    with open(f"{model_folder}/{model_name}.results.txt", "w+") as f:
        f.write(f"  Model: {model_name}")
        f.write(f"")
        f.write(f"Results: ")
        f.write(f"     out_of_road / line_changes_without_order / one_step_order / three_step_order / five_step_order")
        f.write(f"            : {results}")
        f.write(f"out of total: {args.rides_count} / {rides_per_test*3} / {rides_per_test*4} / {rides_per_test*4} / {rides_per_test*4}")


    print(f'[INFO] Results saved to "{model_folder}/{model_name}.results.txt"')
    env.close()


# Start line -  0=left, 1=center, 2=right
def test_drive(env, model,
               start_line: int, target_line: int,
               should_turn_left_for: int, should_turn_right_for: int,
               steps: int, debug: bool = False):
    global already_done
    print(f"Testing, start: {start_line}, end: {target_line}, left: {should_turn_left_for}, right: {should_turn_right_for}, steps: {steps}")
    crashes = 0
    bad_lines = 0

    for i in range(steps):
        step_size = 1 / 15 / steps
        print(f"Estimated time to finish: {get_estimated_time(already_done, already_done+step_size, 15)} seconds")
        already_done += step_size

        env.reset()
        model.reset_states()

        drive_straight(env, model, start_line, 150)
        turn_lefts = should_turn_left_for
        turn_rights = should_turn_right_for

        # Let model drive for 200 steps
        crash = False
        obs, reward, done, info = env.step([0, 1])  # No steer, speed 100% to get
        if debug: print("Model driving")
        for step in range(200):
            left = False
            right = False
            if step > 30:
                left = turn_lefts > 0
                right = turn_rights > 0
                turn_lefts -= 1
                turn_rights -= 1

            if debug:
                if left: print("Turn left signal")
                if right: print("Turn right signal")

            pred = model_predict(model, obs["image"], left, right)
            if debug: print(f"[DEBUG] Predicted steer: {pred}")
            obs, reward, done, info = env.step([pred, 1])  # Predicted steer, speed 100%
            env.render()

            # It is not possible to finish in 100 steps so it must be crash if done
            if done:
                line_position = get_position_from_line(obs, Lines["CENTER"])
                if (info["crash"] or info["out_of_road"]) and abs(line_position) > 0.7:
                    if debug: print(f"Crash, linepos: {line_position}")
                    crash = True
                    crashes += 1
                break

        # Check if in correct line
        line_position = get_position_from_line(obs, target_line)
        if abs(line_position) > 0.2 and not crash:
            if debug: print(f"Bad line by: {line_position:.2f}")
            bad_lines += 1


    return crashes, bad_lines


def model_predict(model: Model, image: np.ndarray, should_turn_left: bool, should_turn_right: bool):
    # image = obs["image"]  # 160x120px
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = cv2.flip(image, 1)
    image = image[36:, :]  # 160x84px

    turns_input = model.input_shape[1][0][2]
    if turns_input == 2:
        should_turns = [1 if should_turn_left else 0, 1 if should_turn_right else 0]
    else:
        should_turns = -1 if should_turn_left else 1 if should_turn_right else 0
    to_pred = [[np.array([[image]]), np.array([[should_turns]])]]

    pred = model.predict(to_pred)[0] / 20
    return pred


latest_position = -0.5
# Target line 0=left, 1=center, 2=right
def drive_straight(env, model, target_line: int, steps: int):
    global latest_position
    obs, reward, done, info = env.step([0, 1])  # No steer, speed 100%
    env.render()
    position_from_center_line = get_position_from_line(obs, 1)
    latest_position = position_from_center_line

    for i in range(steps):
        position_from_center_line = get_position_from_line(obs, 1)
        steer = _calculate_target_steer(position_from_center_line, target_line, latest_position)
        latest_position = position_from_center_line

        if i > steps - 30:
            model_predict(model, obs["image"], False, False)
        obs, reward, done, info = env.step([steer, 1])  # steer, speed 100%
        env.render()

def get_position_from_line(obs, line: int):
    lateral_to_left = obs["state"][0]  # distance to left: 0 to 1
    lateral_to_right = obs["state"][1]  # distance to left: 0 to 1
    position_from_center_line = lateral_to_left - lateral_to_right  # Distance to center: -1 to 1
    if line == 0: return position_from_center_line + 0.5
    if line == 1: return position_from_center_line
    if line == 2: return position_from_center_line - 0.5
    raise ValueError("line must be either 0(left), 1(center) or 2(right)")

# Target line 0=left, 1=center, 2=right
def _calculate_target_steer(position_from_center_line, target_line: int, latest_pos: int) -> (int, int):
    offset = position_from_center_line  # Center line -> no change
    if target_line == 0: offset += 0.5  # Left line -> +0.5
    if target_line == 2: offset -= 0.5  # Right line -> -0.5

    current_steer_speed = position_from_center_line - latest_pos

    target_steer_speed = -(offset / 15)
    if abs(target_steer_speed) < 0.02:
        target_steer_speed *= 2

    steer = -(target_steer_speed - current_steer_speed)
    steer += random.randrange(-100, 100, 1) / 100 * 0.01  # Add some randomness to better prevent overfitting

    return steer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simulator drive tests for trained models.')

    # --- #
    parser.add_argument('-m', dest='model_path', type=str, required=True,
                              help='Path to the model')

    parser.add_argument('-r', dest='rides_count', type=int, default=150,
                              help='Total test rides. Has to by multiply of 15.')
    parser.add_argument('-v', dest='debug', action='store_true',
                              help='Run drive test in verbose mode. -> Printing a lot of data.')
    # --- #

    args = parser.parse_args()
    assert os.path.isfile(args.model_path), "[ERROR] Model file does not exist!"
    assert args.rides_count % 15 == 0, "[ERROR] RIDES_COUNT must be a multiply of 15!"
    assert args.rides_count > 0, "[ERROR] RIDES_COUNT must be a positive integer."

    print(f"\n[INFO] Running drive tests in simulator for {args.rides_count} rides.\n")
    test_model(args)
    print("[DONE]")


if __name__ == '__main__':
    main()
