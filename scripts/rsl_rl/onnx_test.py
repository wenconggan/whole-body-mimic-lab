import onnxruntime as ort
import numpy as np
policy_path = ("/home/wenconggan/whole_body_tracking/logs/rsl_rl/x2_flat/2025-09-01_11-17-33/exported/policy.onnx")

# 加载模型
session = ort.InferenceSession(policy_path)

import onnxruntime as ort
import numpy as np

# 加载模型
# session = ort.InferenceSession("model.onnx")

# 获取输入信息
inputs_info = session.get_inputs()

# 构造测试数据
obs = np.random.randn(1, 96).astype(np.float32)
time_step = np.array([[0.0]], dtype=np.float32)  # 这里可以改成实际时间步
for i in session.get_inputs():
    print(f"Input name: {i.name}, shape: {i.shape}, type: {i.type}")
# 构建输入字典
input_feed = {
    inputs_info[0].name: obs,
    inputs_info[1].name: time_step
}

# 推理
outputs = session.run(None, input_feed)

print("模型输出：", outputs)
