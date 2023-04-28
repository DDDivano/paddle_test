import paddle
import numpy as np
from paddle_nets_lib import SimpleNet
# 创建手工张量
x = np.random.randn(1, 3, 28, 28).astype('float32')
x_tensor = paddle.to_tensor(x)

# 调用网络
net = SimpleNet(num_classes=10)
logits = net(x_tensor)
print(logits.numpy())