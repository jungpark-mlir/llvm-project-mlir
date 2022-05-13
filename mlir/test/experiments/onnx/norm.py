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
    X0 = onnx.helper.make_tensor_value_info(model_input_name0, onnx.TensorProto.FLOAT, [256, 768])

    model_input_name1 = "X1"
    X1 = onnx.helper.make_tensor_value_info(model_input_name1, onnx.TensorProto.FLOAT, [256, 768])

    model_output_name = "Y"
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           [256, 1])

    add0_output_node_name = "Add0_Y"
    add0_node = onnx.helper.make_node(
        name="Add0",  # Name is optional.
        op_type="Add",
        inputs=[model_input_name0, model_input_name1],
        outputs=[add0_output_node_name]
    )

    rmean0_output_node_name = "RMean0_Y"
    rmean0_node = onnx.helper.make_node(
        name="RMean0",  # Name is optional.
        op_type="ReduceMean",
        inputs=[add0_output_node_name],
        outputs=[rmean0_output_node_name],
        axes=[1]
    )

    sub0_output_node_name = "Sub0_Y"
    sub0_node = onnx.helper.make_node(
        name="Sub0",  # Name is optional.
        op_type="Sub",
        inputs=[add0_output_node_name, rmean0_output_node_name],
        outputs=[sub0_output_node_name]
    )

    mul0_output_node_name = "Mul0_Y"
    mul0_node = onnx.helper.make_node(
        name="Mul0",  # Name is optional.
        op_type="Mul",
        inputs=[sub0_output_node_name, sub0_output_node_name],
        outputs=[mul0_output_node_name]
    )

    rmean1_output_node_name = "RMean1_Y"
    rmean1_node = onnx.helper.make_node(
        name="RMean1",  # Name is optional.
        op_type="ReduceMean",
        inputs=[mul0_output_node_name],
        outputs=[rmean1_output_node_name],
        axes=[1]
    )

    add1_output_node_name = "Add0_Y"
    add1_node = onnx.helper.make_node(
        name="Add1",  # Name is optional.
        op_type="Add",
        inputs=[rmean0_output_node_name, rmean1_output_node_name],
        outputs=[model_output_name]
    )

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=[add0_node, rmean0_node, sub0_node, mul0_node, rmean1_node, add1_node],
        name="norm",
        inputs=[X0, X1],  # Graph input
        outputs=[Y],  # Graph output
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = 13

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    onnx.save(model_def, "norm.onnx")


if __name__ == "__main__":

    main()
