"""Check the torch.compile implementation's loss and gradients against plain PyTorch, on CPU.

torch_compile is the default implementation on macOS. Linux defaults to the Triton CCE kernels,
which need a CUDA GPU and are not covered here; torch_compile is selected explicitly so the same
check runs on every built platform. TorchInductor compiles C++ kernels for CPU, so the test
environment needs a C++ compiler (see test.requires in meta.yaml).
"""

import torch
import torch.nn.functional as F

from cut_cross_entropy import linear_cross_entropy

torch.manual_seed(0)
embeddings = torch.randn(16, 8, requires_grad=True)
classifier = torch.randn(32, 8, requires_grad=True)
targets = torch.randint(0, 32, (16,))

actual = linear_cross_entropy(embeddings, classifier, targets, impl="torch_compile")
expected = F.cross_entropy(embeddings @ classifier.T, targets)
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

actual_grads = torch.autograd.grad(actual, (embeddings, classifier))
expected_grads = torch.autograd.grad(expected, (embeddings, classifier))
for actual_grad, expected_grad in zip(actual_grads, expected_grads):
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)
