import argparse
import tensorrt as trt
import os


def build_engine(onnx_model, engine_file, workspace_size, use_fp16):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    # Config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 20))
    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Optimization profile (cho tất cả input)
    profile = builder.create_optimization_profile()

    # image: cố định 1x3x640x640
    profile.set_shape("image",
                      min=(1, 3, 640, 640),
                      opt=(1, 3, 640, 640),
                      max=(1, 3, 640, 640))

    # im_shape: batch=1, shape=(1,2)
    profile.set_shape("im_shape",
                      min=(1, 2),
                      opt=(1, 2),
                      max=(1, 2))

    # scale_factor: batch=1, shape=(1,2)
    profile.set_shape("scale_factor",
                      min=(1, 2),
                      opt=(1, 2),
                      max=(1, 2))

    config.add_optimization_profile(profile)

    print("🚀 Building TensorRT engine…")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("❌ Engine build failed")

    os.makedirs(os.path.dirname(engine_file), exist_ok=True)
    with open(engine_file, "wb") as f:
        f.write(engine)  # engine đã serialize sẵn
    print(f"✅ Saved engine to {engine_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--engine", type=str, required=True, help="Path to save TensorRT engine")
    parser.add_argument("--workspace", type=int, default=4096, help="Max workspace size in MB")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")

    args = parser.parse_args()

    build_engine(
        onnx_model=args.onnx,
        engine_file=args.engine,
        workspace_size=args.workspace,
        use_fp16=args.fp16,
    )
