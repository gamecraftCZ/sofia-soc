from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense, Lambda, Conv2D, Flatten, \
    TimeDistributed, RNN, Input, Activation, concatenate
from tensorflow.keras import Model
from kerasncp import wirings
from kerasncp.tf import LTCCell
import matplotlib.pyplot as plt
import seaborn as sns

# batch_size must to be defined when statefull=True
# config - inter_neurons, inter_fanout, recurrent_command_synapses, \
#           convolution_out_size, command_input_size, use_dense_size
def build_ncp_model(input_shape, config, stateful=False, batch_size=1, draw_ncp_internals=False):
    ## Define NCP
    ncp_wiring = wirings.NCP(
        inter_neurons=config["inter_neurons"],  # Number of inter neurons
        command_neurons=12,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=6,  # How many outgoing synapses has each sensory neuron
        inter_fanout=config["inter_fanout"],  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=config["recurrent_command_synapses"],  # Now many recurrent synapses are in the command neuron layer
        motor_fanin=8,  # How many incomming syanpses has each motor neuron
    )
    ncp_cell = LTCCell(ncp_wiring)


    ## Build model ##
    # INPUT_SHAPE = (timesteps, *input_shape)
    if not stateful:
        batch_size = None

    IMAGE_INPUT_SHAPE = (batch_size, None, *input_shape)
    STEER_INPUT_SHAPE = (batch_size, None, config["command_input_size"])

    # Create CNN part of the network based on Nvidia model
    model_cnn = Sequential()
    model_cnn.add(Lambda(lambda x: x / 127.5 - 1.0, batch_input_shape=IMAGE_INPUT_SHAPE, name="image_normalization"))
    model_cnn.add(TimeDistributed(Conv2D(24, 5, activation='elu', strides=(2, 2)), name="convolution_1"))
    model_cnn.add(TimeDistributed(Conv2D(36, 5, activation='elu', strides=(2, 2)), name="convolution_2"))
    model_cnn.add(TimeDistributed(Conv2D(48, 5, activation='elu', strides=(2, 2)), name="convolution_3"))
    model_cnn.add(TimeDistributed(Conv2D(64, 3, activation='elu'), name="convolution_4"))
    model_cnn.add(TimeDistributed(Conv2D(16, 3, activation='elu'), name="convolution_5"))
    model_cnn.add(TimeDistributed(Dropout(0.3), name="convolution_dropout"))
    model_cnn.add(TimeDistributed(Flatten(), name="convolution_flatten"))
    model_cnn.add(TimeDistributed(Dense(100, activation='elu'), name="convolution_inner_dense"))
    model_cnn.add(TimeDistributed(Dense(config["convolution_out_size"], activation='elu'), name="convolution_output_dense"))

    # Create input for Steering data
    steer_input = Input(batch_input_shape=STEER_INPUT_SHAPE, name="command_input")
    model_steer = steer_input
    if config["use_dense_size"]:
        model_steer = TimeDistributed(Dense(config["use_dense_size"], activation="elu"), name="command_dense")(steer_input)

    model_steer = TimeDistributed(Flatten(), name="command_output_flatten")(model_steer)
    model_steer = Model(inputs=steer_input, outputs=model_steer)

    # Combine CNN part with Steering input
    combined = concatenate([model_cnn.output, model_steer.output], name="combine_cnn_command")

    # Add Recurrent NCP layer to the model
    z = RNN(ncp_cell, return_sequences=False, stateful=stateful, name="NCP_RNN")(combined)
    z = Activation("tanh", name="tanh_output_activation")(z)
    # model.add(TimeDistributed(Dense(1)))

    # Finish model
    model = Model(inputs=[model_cnn.inputs, model_steer.inputs], outputs=z)

    # Visualize NCP connections - must be at the end after model is created
    if draw_ncp_internals:
        sns.set_style("white")
        plt.figure(figsize=(12, 8))
        legend_handles = ncp_cell.draw_graph(neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1.1, 1.1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()

    return model
