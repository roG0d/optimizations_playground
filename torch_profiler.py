import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
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

# Create sample tensors.
batch_size = 32
num_heads = 8
seq_length = 128
d_k = 64
device="cuda:5"
CHROME_PROFILE=False

# Create multiple CUDA streams for parallel execution
streams = [torch.cuda.Stream() for _ in range(num_heads)]

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

# Parallel attn by head with CUDA stream
def attn_streams(q,k,v):
    """
    Perform number_heads x attn_opperations parallel (one per cuda stream/per attn_head), leveraging parallel compute from GPUs (with CUDA streams)
    """
    torch.cuda.empty_cache()
    # Split input tensors by attention head
    q_heads = list(q.split(1, dim=1))  # Split `num_heads`
    k_heads = list(k.split(1, dim=1))
    v_heads = list(v.split(1, dim=1))

    # warmup
    _ = attention(q, k, v)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        with record_function("attention operation from torch"):
        # Benchmark the original attention function.
            start_time = time.perf_counter()
            for i in range(num_heads):
                with torch.cuda.stream(streams[i]):  # Assign a separate stream per head
                    _ = attention(q_heads[i], k_heads[i], v_heads[i])

            torch.cuda.synchronize()
            original_duration = time.perf_counter() - start_time

    print(f"Parallel attn_op time over {num_heads} number of heads: {original_duration:.6f} seconds")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    if CHROME_PROFILE: prof.export_chrome_trace("basic_example_streams.json")

# Sequential approach for torch.compiler
def attn_compiler_seq(q,k,v):
    """
    Perform number_heads x attn_opperations using torch.compile
    """
    torch.cuda.empty_cache()

    compiled_attention = torch.compile(attention)

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

# Parallel approach per head for torch.compiler
def attn__compiler_streams(q,k,v):
    """
    Perform number_heads x attn_opperations parallel (one per cuda stream/per attn_head), leveraging parallel compute from GPUs (with CUDA streams)
    """
    torch.cuda.empty_cache()
    # Split input tensors by attention head
    q_heads = list(q.split(1, dim=1))  # Split `num_heads`
    k_heads = list(k.split(1, dim=1))
    v_heads = list(v.split(1, dim=1))

    compiled_attention = torch.compile(attention)

    # warmup
    _ = compiled_attention(q, k, v)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        with record_function("attention operation from torch"):
        # Benchmark the original attention function.
            start_time = time.perf_counter()
            for i in range(num_heads):
                with torch.cuda.stream(streams[i]):  # Assign a separate stream per head
                    _ = compiled_attention(q_heads[i], k_heads[i], v_heads[i])

            torch.cuda.synchronize()
            original_duration = time.perf_counter() - start_time

    print(f"Parallel attn_op with torch.compile time over {num_heads} number of heads: {original_duration:.6f} seconds")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    if CHROME_PROFILE: prof.export_chrome_trace("basic_example_torch.compile_streams.json")

attn_seq(q,k,v) # CPU: 3.420ms GPU: 465us
attn_streams(q,k,v) # CPU: 3.338ms GPU: 159us
attn_compiler_seq(q,k,v) # CPU: 2.140ms GPU: 394us
attn__compiler_streams(q,k,v) # CPU: 220ms GPU: 141us <- Big overhead in the CPU. Recompiling attn for each head instead the whole operation