import argparse
from time import sleep

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import moving_average


def examine(args):
    with h5py.File(args.data_file, "r") as h5:
        images = h5["images"]   #  images.shape = (None, 560, 84, 160, 3)
        actions = h5["action"]  # actions.shape = (None, 560, 7)
        # action (7) = (velocity, steering, accel, should_turn_left, should_turn_right, stripped_line, contains_data)
        assert images.shape[0] == actions.shape[0], f"Images and Action must match in length! \n" \
                                                    f"  images length: {images.shape[0]}, " \
                                                    f" actions length: {actions.shape[0]}"

        datasets_count = images.shape[0]
        print(f"--- There are {datasets_count} datasets in the file ---")

        #### Look at some data from dataset #####

        # region ### Plot steering for 1st ride ###
        print("--- Steering for 1st fide ---")
        ride = actions[0, :, 1]  # * 20

        fig, axs = plt.subplots(2, 2, figsize=(12, 6))
        h = {1: "a", 2: "b", 4: "c", 5: "d"}
        for i in range(2):
            for j in range(2):
                smoothness = 1 + (i * 3) + j

                x = moving_average(ride, smoothness)

                ax = axs[i][j]
                ax.plot(ride, color="lightblue")
                ax.plot(x, color="red")
                # ax.plot(smooth(ride, smoothness))
                # ax.set_title(f"Steering - smoothness: {smoothness}")
                ax.set_title(f"{h[smoothness]})")
        fig.suptitle("Steering")
        plt.show()
        # endregion

        # region ### (should_turn_left, should_turn_right, stripped_line) ###
        print("--- More interesting data: ---")

        total_steps = 0
        should_turn_lefts = 0
        should_turn_rights = 0
        stripped_lines = 0
        for series in actions:
            for action in series:
                if action[6]:
                    total_steps += 1
                    if action[3]: should_turn_lefts += 1
                    if action[4]: should_turn_rights += 1
                    if action[5]: stripped_lines += 1
                else:
                    break

        print("       total_steps: ", total_steps)
        print(" should_turn_lefts: ", should_turn_lefts)
        print("should_turn_rights: ", should_turn_rights)
        # endregion

        # region ### Show some images from ride ###
        print("--- Some images from ride ---")

        fig, axs = plt.subplots(3, 3)
        for i in range(3):
            for j in range(3):
                m = images[0][i * 50 + j * 100]
                ax = axs[i][j]
                ax.imshow(m, cmap="bone")
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle("Some images from ride")
        plt.show()
        # endregion

        # region ### Plot steering distribution ###
        print("--- Steering distribution ---")

        out_steers = []
        for series in actions:
            for action in series:
                if action[6]:
                    out_steers.append(action[1])
                else:
                    break

        out_steers = np.array(out_steers)# * 20   # Steering is really small, so scale it for easier learning
        out_steers = np.clip(out_steers, -1, 1)  # Clip to be sure it is really in -1 to 1

        plt.hist(out_steers, label="Steer")
        plt.show()
        # endregion

        # region ### Play video of 1th ride ###
        print("--- Video of 1st ride ---")

        fig = plt.figure()
        im = plt.imshow(images[0][0])

        def init():
            im.set_data(images[0][0])

        def animate(frame):
            img = images[0][frame]
            im.set_data(img)
            return im

        ani = FuncAnimation(fig, animate, init_func=init, frames=len([1 for point in actions[0] if point[6]]), interval=50)
        plt.show()
        # endregion

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Examine collected data')

    # --- #
    parser.add_argument('-d', dest='data_file', type=str, default='datasets/collected_data.h5',
                              help='Where to save collected data. Has to be .h5 file.'
                                   ' Can be already existing dataset, as it will append to it.')
    # --- #

    args = parser.parse_args()
    examine(args)
    print("[DONE]")


if __name__ == '__main__':
    main()
