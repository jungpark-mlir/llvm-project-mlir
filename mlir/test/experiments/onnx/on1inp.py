import numpy as np
import onnx


def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor


def main() -> None:

    # Create a dummy convolutional neural network.

    # IO tensors (ValueInfoProto).
    model_input_name0 = "X0"
    X0 = onnx.helper.make_tensor_value_info(model_input_name0, onnx.TensorProto.FLOAT, [None, 768, 1, 256])

    w_name0 = "W0"
    W0 = onnx.helper.make_tensor_value_info(w_name0, onnx.TensorProto.FLOAT, [768, 768, 1, 1])
    w_name1 = "W1"
    W1 = onnx.helper.make_tensor_value_info(w_name1, onnx.TensorProto.FLOAT, [768, 768, 1, 1])
    w_name2 = "W2"
    W2 = onnx.helper.make_tensor_value_info(w_name2, onnx.TensorProto.FLOAT, [768, 768, 1, 1])


    model_output_name0 = "Y0"
    model_output_channels = 768
    Y0 = onnx.helper.make_tensor_value_info(model_output_name0,
                                           onnx.TensorProto.FLOAT,
                                           [None, model_output_channels, 1, 256])
    model_output_name1 = "Y1"
    Y1 = onnx.helper.make_tensor_value_info(model_output_name1,
                                           onnx.TensorProto.FLOAT,
                                           [None, model_output_channels, 1, 256])
    model_output_name2 = "Y2"
    Y2 = onnx.helper.make_tensor_value_info(model_output_name2,
                                           onnx.TensorProto.FLOAT,
                                           [None, model_output_channels, 1, 256])


    # Create a Conv node (NodeProto).
    # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#conv
    conv0_output_node_name = "Conv0_Y"
    # Dummy weights for conv.
    conv0_in_channels = 768
    conv0_out_channels = 768
    conv0_kernel_shape = (1, 1)
    conv0_pads = (0, 0, 0, 0)
    conv0_B = np.ones(shape=(conv0_out_channels)).astype(np.float32)
    # Create the initializer tensor for the weights.
    conv0_B_initializer_tensor_name = "Conv0_B"
    conv0_B_initializer_tensor = create_initializer_tensor(
        name=conv0_B_initializer_tensor_name,
        tensor_array=conv0_B,
        data_type=onnx.TensorProto.FLOAT)

    conv0_node = onnx.helper.make_node(
        name="Conv0",  # Name is optional.
        op_type="Conv",
        # Must follow the order of input and output definitions.
        # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
        inputs=[
            model_input_name0, w_name0,
            conv0_B_initializer_tensor_name
        ],
        outputs=[conv0_output_node_name],
        # The following arguments are attributes.
        kernel_shape=conv0_kernel_shape,
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=conv0_pads,
    )

    conv1_output_node_name = "Conv1_Y"
    # Dummy weights for conv.
    conv1_in_channels = 768
    conv1_out_channels = 768
    conv1_kernel_shape = (1, 1)
    conv1_pads = (0, 0, 0, 0)
    conv1_B = np.ones(shape=(conv1_out_channels)).astype(np.float32)
    # Create the initializer tensor for the weights.
    conv1_B_initializer_tensor_name = "Conv1_B"
    conv1_B_initializer_tensor = create_initializer_tensor(
        name=conv1_B_initializer_tensor_name,
        tensor_array=conv1_B,
        data_type=onnx.TensorProto.FLOAT)

    conv1_node = onnx.helper.make_node(
        name="Conv1",  # Name is optional.
        op_type="Conv",
        # Must follow the order of input and output definitions.
        # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
        inputs=[
            model_input_name0, w_name1,
            conv1_B_initializer_tensor_name
        ],
        outputs=[conv1_output_node_name],
        # The following arguments are attributes.
        kernel_shape=conv1_kernel_shape,
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=conv1_pads,
    )

    conv2_output_node_name = "Conv2_Y"
    # Dummy weights for conv.
    conv2_in_channels = 768
    conv2_out_channels = 768
    conv2_kernel_shape = (1, 1)
    conv2_pads = (0, 0, 0, 0)
    conv2_B = np.ones(shape=(conv2_out_channels)).astype(np.float32)
    # Create the initializer tensor for the weights.
    conv2_B_initializer_tensor_name = "Conv2_B"
    conv2_B_initializer_tensor = create_initializer_tensor(
        name=conv2_B_initializer_tensor_name,
        tensor_array=conv2_B,
        data_type=onnx.TensorProto.FLOAT)

    conv2_node = onnx.helper.make_node(
        name="Conv2",  # Name is optional.
        op_type="Conv",
        # Must follow the order of input and output definitions.
        # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
        inputs=[
            model_input_name0, w_name2,
            conv2_B_initializer_tensor_name
        ],
        outputs=[conv2_output_node_name],
        # The following arguments are attributes.
        kernel_shape=conv2_kernel_shape,
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=conv2_pads,
    )

    # Create a ReLU node (NodeProto).
    relu0_output_node_name = "ReLU0_Y"
    relu0_node = onnx.helper.make_node(
        name="ReLU0",  # Name is optional.
        op_type="Relu",
        inputs=[conv0_output_node_name],
        outputs=[model_output_name0],
    )
    relu1_output_node_name = "ReLU1_Y"
    relu1_node = onnx.helper.make_node(
        name="ReLU1",  # Name is optional.
        op_type="Relu",
        inputs=[conv1_output_node_name],
        outputs=[model_output_name1],
    )
    relu2_output_node_name = "ReLU2_Y"
    relu2_node = onnx.helper.make_node(
        name="ReLU2",  # Name is optional.
        op_type="Relu",
        inputs=[conv2_output_node_name],
        outputs=[model_output_name2],
    )

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=[conv0_node, conv1_node, conv2_node, relu0_node, relu1_node, relu2_node],
        name="ConvNet",
        inputs=[X0, W0, W1, W2],  # Graph input
        outputs=[Y0, Y1, Y2],  # Graph output
        initializer=[
            conv0_B_initializer_tensor,
            conv1_B_initializer_tensor,
            conv2_B_initializer_tensor
        ],
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = 13

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    onnx.save(model_def, "convshare.onnx")


if __name__ == "__main__":

    main()
