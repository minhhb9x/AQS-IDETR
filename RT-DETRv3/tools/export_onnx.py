import os
import argparse
import numpy as np
import paddle
import paddle2onnx
import onnxruntime as ort

def export_to_onnx(model_dir, model_filename, params_filename, onnx_save_path, opset=16):
    # tạo folder nếu chưa có
    os.makedirs(os.path.dirname(onnx_save_path), exist_ok=True)

    # dynamic shape
    dynamic_input_shapes = {
        "image": [-1, 3, -1, -1],
        "im_shape": [-1, 2],
        "scale_factor": [-1, 2]
    }

    # xuất ONNX
    dynamic_shapes_str = ";".join([f"{k}:{','.join(map(str, v))}" for k,v in dynamic_input_shapes.items()])
    model_filename = os.path.join(model_dir, model_filename)
    params_filename = os.path.join(model_dir, params_filename)
    paddle2onnx.export(
        model_filename=model_filename,
        params_filename=params_filename,
        opset_version=opset,
        save_file=onnx_save_path,
        enable_onnx_checker=True,
        export_fp16_model=True
    )

    print(f"[INFO] ONNX model saved at: {onnx_save_path}")

def test_onnx(onnx_path):
    # tạo input giả
    batch_size = 1
    H, W = 640, 640
    image = np.random.randn(batch_size, 3, H, W).astype("float32")
    im_shape = np.array([[H, W]]).astype("float32")
    scale_factor = np.array([[H/640, W/640]]).astype("float32")

    ort_session = ort.InferenceSession(onnx_path)
    inputs = {
        "image": image,
        "im_shape": im_shape,
        "scale_factor": scale_factor
    }

    outputs = ort_session.run(None, inputs)
    for i, out in enumerate(outputs):
        print(f"Output {i}: shape={out.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Folder containing model.json + model.pdiparams")
    parser.add_argument("--model_filename", type=str, default="model.json")
    parser.add_argument("--params_filename", type=str, default="model.pdiparams")
    parser.add_argument("--onnx_save_path", type=str, default="./export_models/model.onnx")
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--test", action="store_true", help="Run a quick ONNX test after export")

    args = parser.parse_args()

    export_to_onnx(
        model_dir=args.model_dir,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        onnx_save_path=args.onnx_save_path,
        opset=args.opset
    )

    if args.test:
        print("[INFO] Running test inference on exported ONNX model...")
        test_onnx(args.onnx_save_path)
