import exact.cpp_extension.quantization as ext_quantization
from exact.layers import QDropout
import torch
import torch.nn

torch.manual_seed(0)

my_dropout = QDropout(0.5)
torch_dropout = torch.nn.Dropout(0.5)


input = torch.randn(20, 16, dtype=torch.float32, requires_grad=True)
grad_output = torch.randn(20, 16, dtype=torch.float32)

my_output = my_dropout(input)
torch_output = torch_dropout(input)

print(torch.allclose(my_output, torch_output))

my_dropout.backward(grad_output)
torch_output.backward(grad_output)

print(torch.allclose(input.grad, torch_output.grad))

