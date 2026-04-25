import torch
import torchvision.models as models
import tensorrt as trt
import numpy as np
import time

# 1. 加载模型并导出 ONNX
model = models.resnet18(weights=None).cuda().eval()
dummy_input = torch.randn(1, 3, 224, 224).cuda()

torch.onnx.export(model, dummy_input, "resnet18.onnx",
                  input_names=["input"], output_names=["output"])

# 2. 构建 TensorRT 配置
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open("resnet18.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)

# 3. 修复后的校准器：有限次数返回 batch
class MyCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, num_batches=5):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.num_batches = num_batches
        self.batch_count = 0
        self.cache_file = "calibration.cache"
        # 预生成随机数据放在 GPU 上
        self.device_inputs = [torch.randn(10, 3, 224, 224).cuda() for _ in range(num_batches)]

    def get_batch_size(self):
        return 10

    def get_batch(self, names):
        if self.batch_count < self.num_batches:
            ptr = int(self.device_inputs[self.batch_count].data_ptr())
            self.batch_count += 1
            return [ptr]
        else:
            return []   # 返回空列表告诉 TensorRT 标定结束

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass

calibrator = MyCalibrator(num_batches=5)
config.int8_calibrator = calibrator

# 4. 构建引擎
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Failed to build TensorRT engine.")
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

# 5. 准备 buffer
d_input = torch.empty((1, 3, 224, 224), dtype=torch.float32, device="cuda")
d_output = torch.empty((1, 1000), dtype=torch.float32, device="cuda")
context.set_tensor_address("input", d_input.data_ptr())
context.set_tensor_address("output", d_output.data_ptr())

# 6. PyTorch 测速
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(dummy_input)
torch.cuda.synchronize()
pytorch_time = (time.time() - start) / 100

# 7. TensorRT INT8 测速
d_input.copy_(dummy_input)
stream = torch.cuda.Stream()
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    context.execute_async_v3(stream.cuda_stream)
stream.synchronize()
trt_time = (time.time() - start) / 100

print(f"PyTorch 推理耗时: {pytorch_time*1000:.2f} ms")
print(f"TensorRT (INT8) 耗时: {trt_time*1000:.2f} ms")
print(f"加速比: {pytorch_time/trt_time:.2f}x")