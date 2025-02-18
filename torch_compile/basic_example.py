import torch
import torch.nn.functional as F
import time
import sys
import os


# Define a scaled dot-product attention function.
def attention(q, k, v):
    # q, k, v shape: [batch_size, num_heads, seq_length, d_k]
    d_k = q.size(-1)
    # Compute scaled dot-product attention scores.
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    # Multiply the attention weights by the values.
    return torch.matmul(attn, v)

# Compile the attention function.
compiled_attention = torch.compile(attention)

# Create sample tensors.
batch_size = 32
num_heads = 8
seq_length = 128
d_k = 64

q = torch.randn(batch_size, num_heads, seq_length, d_k)
k = torch.randn(batch_size, num_heads, seq_length, d_k)
v = torch.randn(batch_size, num_heads, seq_length, d_k)

# Warm-up the compiled function to avoid including the one-time compilation overhead.
_ = compiled_attention(q, k, v)

iterations = 100

# Benchmark the original attention function.
start_time = time.perf_counter()
for _ in range(iterations):
    _ = attention(q, k, v)
original_duration = time.perf_counter() - start_time

# Benchmark the compiled attention function.
start_time = time.perf_counter()
for _ in range(iterations):
    _ = compiled_attention(q, k, v)
compiled_duration = time.perf_counter() - start_time

print(f"Original attention function time over {iterations} iterations: {original_duration:.6f} seconds")
print(f"Compiled attention function time over {iterations} iterations: {compiled_duration:.6f} seconds")
