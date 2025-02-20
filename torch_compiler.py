# https://pytorch.org/docs/stable/torch.compiler.html
import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import time
import sys
import os

# Define a scaled dot-product attention function. CHANGED: for cuda graph to fusion kernels, we need to make the op more complex 
def attention(q, k, v):
    # q, k, v shape: [batch_size, num_heads, seq_length, d_k]
    d_k = q.size(-1)
    # Compute scaled dot-product attention scores.
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    # Multiply the attention weights by the values.
    mul = torch.matmul(attn, v)

    # Complication from https://pytorch.org/docs/stable/torch.compiler_get_started.html
    a = torch.sin(mul)
    b = torch.cos(a)
    return b

# Create sample tensors.
batch_size = 32
num_heads = 8
seq_length = 128
d_k = 64
device="cuda:5"
CHROME_PROFILE=False

q = torch.randn(batch_size, num_heads, seq_length, d_k, device=device)
k = torch.randn(batch_size, num_heads, seq_length, d_k, device=device)
v = torch.randn(batch_size, num_heads, seq_length, d_k, device=device)

# Sequential attn without CUDA optimizations
def attn_seq(q,k,v):
    """
    Perform number_heads x attn_opperations sequentially, without leveraging parallel compute from GPUs (no CUDA streams)
    """
    torch.cuda.empty_cache()
    iterations = num_heads # Same as number heads for fair comparation
    # warmup
    _ = attention(q, k, v)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        with record_function("attention operation from torch"):
        # Benchmark the original attention function.
            start_time = time.perf_counter()
            for i in range(iterations):
                _ = attention(q, k, v)
            original_duration = time.perf_counter() - start_time

    print(f"Sequential attn_op time over {iterations} iterations: {original_duration:.6f} seconds")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    if CHROME_PROFILE: prof.export_chrome_trace("basic_example_async.json")

# Sequential approach for torch.compiler -> TorchInductor -> Triton backend
def attn_compiler_seq(q,k,v):
    """
    Perform number_heads x attn_opperations using torch.compile
    """
    torch.cuda.empty_cache()

    compiled_attention = torch.compile(attention, backend="inductor")

    # warmup
    _ = compiled_attention(q, k, v)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        with record_function("attention operation from torch"):
        # Benchmark the original attention function.
            start_time = time.perf_counter()
            for i in range(num_heads):
                _ = compiled_attention(q, k, v)

            original_duration = time.perf_counter() - start_time

    print(f"Sequential attn_op with torch.compiler time over {num_heads} number of heads: {original_duration:.6f} seconds")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    if CHROME_PROFILE: prof.export_chrome_trace("basic_example_torch_compile_async.json")

# CUDA graphs
def attn_compiler_graphs(q,k,v):
    #TODO
    pass
attn_seq(q,k,v) # CPU: 8.168ms GPU: 465us op_time: 3.049 ms
attn_compiler_seq(q,k,v) # CPU: 1.895ms GPU: 394us op_time: 2.078ms
