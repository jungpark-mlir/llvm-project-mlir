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
                                           [256, 768])

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

    const0 = np.ones(shape=(1)).astype(np.float32)
    const0_tensor_name = "Const0"
    const0_tensor = create_initializer_tensor(
        name=const0_tensor_name,
        tensor_array=const0,
        data_type=onnx.TensorProto.FLOAT)

    add1_output_node_name = "Add1_Y"
    add1_node = onnx.helper.make_node(
        name="Add1",  # Name is optional.
        op_type="Add",
        inputs=[rmean1_output_node_name, const0_tensor_name],
        outputs=[add1_output_node_name]
    )

    sqrt0_output_node_name = "SQRT0_Y"
    sqrt0_node = onnx.helper.make_node(
        name="SQRT0",  # Name is optional.
        op_type="Sqrt",
        inputs=[add1_output_node_name],
        outputs=[sqrt0_output_node_name],
    )

    recip0_output_node_name = "Recip0_Y"
    recip0_node = onnx.helper.make_node(
        name="Recip0",  # Name is optional.
        op_type="Reciprocal",
        inputs=[sqrt0_output_node_name],
        outputs=[recip0_output_node_name],
    )

    const1 = np.ones(shape=(768)).astype(np.float32)
    const1_tensor_name = "Const1"
    const1_tensor = create_initializer_tensor(
        name=const1_tensor_name,
        tensor_array=const1,
        data_type=onnx.TensorProto.FLOAT)

    mul1_output_node_name = "Mul1_Y"
    mul1_node = onnx.helper.make_node(
        name="Mul1",  # Name is optional.
        op_type="Mul",
        inputs=[recip0_output_node_name, const1_tensor_name],
        outputs=[mul1_output_node_name]
    )

    mul2_output_node_name = "Mul2_Y"
    mul2_node = onnx.helper.make_node(
        name="Mul2",  # Name is optional.
        op_type="Mul",
        inputs=[rmean0_output_node_name, mul1_output_node_name],
        outputs=[mul2_output_node_name]
    )

    const2 = np.ones(shape=(768)).astype(np.float32)
    const2_tensor_name = "Const2"
    const2_tensor = create_initializer_tensor(
        name=const2_tensor_name,
        tensor_array=const2,
        data_type=onnx.TensorProto.FLOAT)

    add2_output_node_name = "Add2_Y"
    add2_node = onnx.helper.make_node(
        name="Add2",  # Name is optional.
        op_type="Add",
        inputs=[const2_tensor_name, mul2_output_node_name],
        outputs=[model_output_name]
    )

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=[add0_node, rmean0_node, sub0_node, mul0_node, rmean1_node, add1_node, sqrt0_node, recip0_node, mul1_node, mul2_node, add2_node],
        name="norm",
        inputs=[X0, X1],  # Graph input
        outputs=[Y],  # Graph output
        initializer=[
            const0_tensor, const1_tensor, const2_tensor 
        ],
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = 13

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    onnx.save(model_def, "normblock1.onnx")


if __name__ == "__main__":

    main()
