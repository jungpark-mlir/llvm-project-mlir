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
    model_input_name1 = "X1"
    model_input_name2 = "X2"
    model_input_name3 = "X3"
    
    X0 = onnx.helper.make_tensor_value_info(model_input_name0, onnx.TensorProto.FLOAT, [None, 256, 1, 256])
    X1 = onnx.helper.make_tensor_value_info(model_input_name1, onnx.TensorProto.FLOAT, [None, 256, 1, 256])
    X2 = onnx.helper.make_tensor_value_info(model_input_name2, onnx.TensorProto.FLOAT, [None, 256, 1, 256])
    X3 = onnx.helper.make_tensor_value_info(model_input_name3, onnx.TensorProto.FLOAT, [None, 256, 1, 256])

    model_input_name4 = "X4"
    model_input_name5 = "X5"
    model_input_name6 = "X6"
    model_input_name7 = "X7"
    
    X4 = onnx.helper.make_tensor_value_info(model_input_name4, onnx.TensorProto.FLOAT, [256, 256, 1, 1])
    X5 = onnx.helper.make_tensor_value_info(model_input_name5, onnx.TensorProto.FLOAT, [256, 256, 1, 1])
    X6 = onnx.helper.make_tensor_value_info(model_input_name6, onnx.TensorProto.FLOAT, [256, 256, 1, 1])
    X7 = onnx.helper.make_tensor_value_info(model_input_name7, onnx.TensorProto.FLOAT, [256, 256, 1, 1])



    model_output_name = "Y"
    model_output_channels = 256
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           [None, model_output_channels, 1, 256])

    # Create a Conv node (NodeProto).
    # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#conv
    conv0_output_node_name = "Conv0_Y"
    # Dummy weights for conv.
    conv0_in_channels = 256
    conv0_out_channels = 256
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
            model_input_name0, model_input_name4,
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
    conv1_in_channels = 256
    conv1_out_channels = 256
    conv1_kernel_shape = (1, 1)
    conv1_pads = (0, 0, 0, 0)
    conv1_B = np.ones(shape=(conv1_out_channels)).astype(np.float32)
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
            model_input_name1, model_input_name5,
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
    conv2_in_channels = 256
    conv2_out_channels = 256
    conv2_kernel_shape = (1, 1)
    conv2_pads = (0, 0, 0, 0)
    conv2_B = np.ones(shape=(conv2_out_channels)).astype(np.float32)
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
            model_input_name2, model_input_name6,
            conv2_B_initializer_tensor_name
        ],
        outputs=[conv2_output_node_name],
        # The following arguments are attributes.
        kernel_shape=conv2_kernel_shape,
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=conv2_pads,
    )

    conv3_output_node_name = "Conv3_Y"
    # Dummy weights for conv.
    conv3_in_channels = 256
    conv3_out_channels = 256
    conv3_kernel_shape = (1, 1)
    conv3_pads = (0, 0, 0, 0)
    conv3_B = np.ones(shape=(conv3_out_channels)).astype(np.float32)
    conv3_B_initializer_tensor_name = "Conv3_B"
    conv3_B_initializer_tensor = create_initializer_tensor(
        name=conv3_B_initializer_tensor_name,
        tensor_array=conv3_B,
        data_type=onnx.TensorProto.FLOAT)

    conv3_node = onnx.helper.make_node(
        name="Conv3",  # Name is optional.
        op_type="Conv",
        # Must follow the order of input and output definitions.
        # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
        inputs=[
            model_input_name3, model_input_name7,
            conv3_B_initializer_tensor_name
        ],
        outputs=[conv3_output_node_name],
        # The following arguments are attributes.
        kernel_shape=conv3_kernel_shape,
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=conv3_pads,
    )


    # Create a ReLU node (NodeProto).
    relu0_output_node_name = "ReLU0_Y"
    relu0_node = onnx.helper.make_node(
        name="ReLU0",  # Name is optional.
        op_type="Relu",
        inputs=[conv0_output_node_name],
        outputs=[relu0_output_node_name],
    )
    relu1_output_node_name = "ReLU1_Y"
    relu1_node = onnx.helper.make_node(
        name="ReLU1",  # Name is optional.
        op_type="Relu",
        inputs=[conv1_output_node_name],
        outputs=[relu1_output_node_name],
    )
    relu2_output_node_name = "ReLU2_Y"
    relu2_node = onnx.helper.make_node(
        name="ReLU2",  # Name is optional.
        op_type="Relu",
        inputs=[conv2_output_node_name],
        outputs=[relu2_output_node_name],
    )
    relu3_output_node_name = "ReLU3_Y"
    relu3_node = onnx.helper.make_node(
        name="ReLU3",  # Name is optional.
        op_type="Relu",
        inputs=[conv3_output_node_name],
        outputs=[relu3_output_node_name],
    )

    # Create a Add node (NodeProto).
    add1_output_node_name = "add1_Y"
    add1_node = onnx.helper.make_node(
        name="add1",  # Name is optional.
        op_type="Add",
        inputs=[relu0_output_node_name, relu1_output_node_name],
        outputs=[add1_output_node_name],
    )

    # Create a Add node (NodeProto).
    add2_output_node_name = "add2_Y"
    add2_node = onnx.helper.make_node(
        name="add2",  # Name is optional.
        op_type="Add",
        inputs=[relu2_output_node_name, relu3_output_node_name],
        outputs=[add2_output_node_name],
    )

    # Create a Add node (NodeProto).
    add3_output_node_name = "add3_Y"
    add3_node = onnx.helper.make_node(
        name="add3",  # Name is optional.
        op_type="Add",
        inputs=[add1_output_node_name, add2_output_node_name],
        outputs=[model_output_name],
    )


    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=[conv0_node, conv1_node, conv2_node, conv3_node, relu0_node, relu1_node, relu2_node, relu3_node, add1_node, add2_node, add3_node],
        name="ConvNet",
        inputs=[X0, X1, X2, X3, X4, X5, X6, X7],  # Graph input
        outputs=[Y],  # Graph output
        initializer=[
            conv0_B_initializer_tensor,
            conv1_B_initializer_tensor,
            conv2_B_initializer_tensor,
            conv3_B_initializer_tensor
        ],
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = 13

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    onnx.save(model_def, "tiled.onnx")


if __name__ == "__main__":

    main()
