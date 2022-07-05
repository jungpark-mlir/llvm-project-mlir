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
    bsize = 256
    model_input_name0 = "X0"
    X0 = onnx.helper.make_tensor_value_info(model_input_name0, onnx.TensorProto.FLOAT, [bsize, 256, 14, 14])

    model_output_name = "Y"
    model_output_channels = 256
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           [bsize, model_output_channels, 14, 14])

    # Create a Conv node (NodeProto).
    # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#conv
    conv0_output_node_name = "Conv0_Y"
    # Dummy weights for conv.
    conv0_in_channels = 256
    conv0_out_channels = 256
    conv0_kernel_shape = (3, 3)
    conv0_pads = (1, 1, 1, 1)
    conv0_W = np.ones(shape=(conv0_out_channels, conv0_in_channels,
                             *conv0_kernel_shape)).astype(np.float32)
    conv0_B = np.ones(shape=(conv0_out_channels)).astype(np.float32)
    # Create the initializer tensor for the weights.
    conv0_W_initializer_tensor_name = "Conv0_W"
    conv0_W_initializer_tensor = create_initializer_tensor(
        name=conv0_W_initializer_tensor_name,
        tensor_array=conv0_W,
        data_type=onnx.TensorProto.FLOAT)
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
            model_input_name0, conv0_W_initializer_tensor_name,
            conv0_B_initializer_tensor_name
        ],
        outputs=[conv0_output_node_name],
        # The following arguments are attributes.
        kernel_shape=conv0_kernel_shape,
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=conv0_pads,
    )

    # Create a ReLU node (NodeProto).
    relu0_output_node_name = "ReLU0_Y"
    relu0_node = onnx.helper.make_node(
        name="ReLU0",  # Name is optional.
        op_type="Relu",
        inputs=[conv0_output_node_name],
        outputs=[model_output_name],
    )

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=[conv0_node, relu0_node],
        name="ConvNet",
        inputs=[X0],  # Graph input
        outputs=[Y],  # Graph output
        initializer=[
            conv0_W_initializer_tensor, conv0_B_initializer_tensor
        ],
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = 13

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    onnx.save(model_def, "rconv9.onnx")


if __name__ == "__main__":

    main()
