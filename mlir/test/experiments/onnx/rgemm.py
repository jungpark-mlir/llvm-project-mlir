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
    X0 = onnx.helper.make_tensor_value_info(model_input_name0, onnx.TensorProto.FLOAT, [256, 768, 768])

    model_output_name = "Y"
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           [256, 768, 1])

    rmean0_output_node_name = "RMean0_Y"
    rmean0_node = onnx.helper.make_node(
        name="RMean0",  # Name is optional.
        op_type="ReduceMean",
        inputs=[model_input_name0],
        outputs=[model_output_name],
        axes=[2]
    )

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=[rmean0_node],
        name="norm",
        inputs=[X0],  # Graph input
        outputs=[Y],  # Graph output
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = 13

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    onnx.save(model_def, "rgemm.onnx")


if __name__ == "__main__":

    main()
