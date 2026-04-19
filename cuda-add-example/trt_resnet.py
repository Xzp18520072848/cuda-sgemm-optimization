import torch
import torchvision.models as models
import tensorrt as trt
import time

# 1. 加载 PyTorch 模型（使用本地缓存）
model = models.resnet18(weights=None).cuda().eval()
dummy_input = torch.randn(1, 3, 224, 224).cuda()

# 2. 导出 ONNX
torch.onnx.export(model, dummy_input, "resnet18.onnx", 
                  input_names=["input"], output_names=["output"])

# 3. 构建 TensorRT 引擎
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open("resnet18.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
config.set_flag(trt.BuilderFlag.FP16)  # FP16 加速
engine = builder.build_serialized_network(network, config)

# 4. 创建运行时
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine)
context = engine.create_execution_context()

# 5. 准备 buffer
d_input = torch.empty((1, 3, 224, 224), dtype=torch.float32, device="cuda")
d_output = torch.empty((1, 1000), dtype=torch.float32, device="cuda")
context.set_tensor_address("input", d_input.data_ptr())
context.set_tensor_address("output", d_output.data_ptr())

# 6. 测速：PyTorch
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(dummy_input)
torch.cuda.synchronize()
pytorch_time = (time.time() - start) / 100

# 7. 测速：TensorRT
d_input.copy_(dummy_input)
stream = torch.cuda.Stream()
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    context.execute_async_v3(stream.cuda_stream)
stream.synchronize()
trt_time = (time.time() - start) / 100

print(f"PyTorch 推理耗时: {pytorch_time*1000:.2f} ms")
print(f"TensorRT (FP16) 耗时: {trt_time*1000:.2f} ms")
print(f"加速比: {pytorch_time/trt_time:.2f}x")