import tensorrt as trt

def build_engine(onnx_file_path, engine_file_path="model.engine", fp16=True):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags=network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ Failed to parse ONNX:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # ✅ 设置 FP16
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # ✅ 设置 Optimization Profile（关键）
    input_tensor = network.get_input(0)  # 通常是 "images"
    profile = builder.create_optimization_profile()

    # 为 YOLOv8 模型设置输入尺寸范围：min / opt / max
    profile.set_shape(
        input_tensor.name,
        min=(1, 3, 320, 320),
        opt=(1, 3, 640, 640),
        max=(1, 3, 1280, 1280)
    )
    config.add_optimization_profile(profile)

    # ✅ 构建序列化引擎
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("❌ 引擎构建失败")
        return None

    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    print(f"✅ TensorRT 引擎已保存到: {engine_file_path}")
    return serialized_engine


if __name__ == "__main__":
    build_engine("yolov8n.onnx", "yolov8n.engine")

