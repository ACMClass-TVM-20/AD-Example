import itertools
import numpy as np
import torch
import torch.nn.functional as F

def grid(*args):
	return itertools.product(*[range(x) for x in args])
# nchw x oikk
# padding = 0
# output shape: N, O, H - (K - 1), W - (K - 1)
# forward
def conv2d(input, weight, output):
	N, C, H, W = input.shape
	O, I, K, _ = weight.shape
	pad_tmp = np.zeros((N, C, H, W))
	for n, c, h, w in grid(N, C, H, W):
		pad_tmp[n, c, h, w] = input[n, c, h, w]
	for n, o, h1, w1, i, k1, k2 in grid(N, O, H - K + 1, W - K + 1, I, K, K):
		if i == 0 and k1 == 0 and k2 == 0:
			output[n, o, h1, w1] = 0
		output[n, o, h1, w1] += pad_tmp[n, i, h1 + k1, w1 + k2] * weight[o, i, k1, k2]


def conv2d_adj(input, weight, output, output_adj, input_adj, weight_adj):
	N, C, H, W = input.shape
	O, I, K, _ = weight.shape
	pad_tmp = np.zeros((N, C, H, W))
	for n, c, h, w in grid(N, C, H, W):
		pad_tmp[n, c, h, w] = input[n, c, h, w]
	for n, o, h1, w1, i, k1, k2 in grid(N, O, H - K + 1, W - K + 1, I, K, K):
		if i == 0 and k1 == 0 and k2 == 0:
			output[n, o, h1, w1] = 0
		output[n, o, h1, w1] += pad_tmp[n, i, h1 + k1, w1 + k2] * weight[o, i, k1, k2]

	pad_tmp_adj = np.zeros(pad_tmp.shape)
	# init pad_tmp_adj and weight_adj
	for n, o, h1, w1, i, k1, k2 in grid(N, O, H - K + 1, W - K + 1, I, K, K):
		# init doesn't influence result
		weight_adj[o, i, k1, k2] += output_adj[n, o, h1, w1] * pad_tmp[n, i, h1 + k1, w1 + k2]
		pad_tmp_adj[n, i, h1 + k1, w1 + k2] += output_adj[n, o, h1, w1] * weight[o, i, k1, k2]
	# init input_adj
	for n, c, h, w in grid(N, C, H, W):
		input_adj[n, c, h, w] += pad_tmp_adj[n, c, h, w]



input = np.random.randint(0, 3, size=(1, 2, 6, 6)).astype(np.float32)
weight = np.random.randint(0, 3, size=(1, 2, 3, 3)).astype(np.float32)
output = np.zeros((1, 1, 4, 4))
output_adj = np.ones(output.shape)
input_adj = np.zeros(input.shape)
weight_adj = np.zeros(weight.shape)
conv2d_adj(input, weight, output, output_adj, input_adj, weight_adj)

input_t = torch.tensor(input, requires_grad = True)
weight_t = torch.tensor(weight, requires_grad = True)
output_t = F.conv2d(input_t, weight_t)
output_t.retain_grad()
loss_t = torch.sum(output_t)
loss_t.backward()
np.testing.assert_equal(output_t.detach().numpy(), output)
np.testing.assert_equal(output_t.grad.detach().numpy(), output_adj)
np.testing.assert_equal(weight_t.grad.detach().numpy(), weight_adj)
np.testing.assert_equal(input_t.grad.detach().numpy(), input_adj)
