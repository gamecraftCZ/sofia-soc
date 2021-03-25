import tempfile
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import Model

from build_ncp_model import build_ncp_model


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def baseline_mean_squared_error_from_rides(rides_flat):
    output_steer = rides_flat[:, 1]

    steer_mean = np.mean(output_steer)

    error = (output_steer - steer_mean)
    error_squared = error ** 2
    mean_squared_error = np.mean(error_squared)

    return mean_squared_error


def get_model(model_name: str, model_folder: str):
    with tempfile.TemporaryDirectory() as temp_folder:
        # Load model
        trained_model = keras.models.load_model(f"{model_folder}/{model_name}.h5")
        trained_model.save_weights(f"{temp_folder}/temp.weights")
        trained_model.summary(line_length=140)
        print("---xxx---")

        # Create new model
        spl = model_name.split("-")
        model_config = dict(
            convolution_out_size=int(spl[2]),
            inter_neurons=int(spl[3]),
            inter_fanout=int(spl[4]),
            recurrent_command_synapses=int(spl[5]),
            command_input_size=int(spl[6]),
            use_dense_size=None if spl[7] == "None" else int(spl[7]),
        )
        model = build_ncp_model((84, 160, 3), model_config, stateful=True)  # 10 frames = 1s
        print(f"Testing model: {model_name}")
        trained_model.summary(line_length=140)

        # Load weights
        model.load_weights(f"{temp_folder}/temp.weights")
        return model


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
