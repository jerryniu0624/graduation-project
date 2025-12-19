#!/usr/bin/env python3
"""修改后的基准测试脚本 - 手动管理CUDA上下文"""

import tensorrt as trt
import onnxruntime as ort
import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime

# 尝试手动初始化 CUDA
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    cuda_available = True
    print("✓ PyCUDA 自动初始化成功")
except Exception as e:
    print(f"⚠ PyCUDA 自动初始化失败: {e}")
    print("尝试手动初始化 CUDA...")
    try:
        import pycuda.driver as cuda
        cuda.init()
        device = cuda.Device(0)  # 使用第一个 GPU
        context = device.make_context()
        cuda_available = True
        print("✓ PyCUDA 手动初始化成功")
        
        # 注册退出时清理上下文
        import atexit
        atexit.register(context.pop)
    except Exception as e2:
        print(f"✗ PyCUDA 手动初始化也失败: {e2}")
        print("将使用 ONNX CPU 模式进行测试")
        cuda_available = False

def benchmark_onnx_ort(onnx_path, input_shape, num_iterations=100, warmup=10):
    """使用 ONNX Runtime 基准测试 ONNX 模型"""
    print(f"\n{'='*60}")
    print(f"ONNX Runtime 基准测试 - 尺寸: {input_shape[2]}x{input_shape[3]}")
    print(f"{'='*60}")
    
    if not os.path.exists(onnx_path):
        print(f"✗ ONNX 模型文件不存在: {onnx_path}")
        return None
    
    try:
        # 准备输入数据
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # 根据可用性选择提供者
        available_providers = ort.get_available_providers()
        print(f"可用提供者: {available_providers}")
        
        if cuda_available and 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif 'AzureExecutionProvider' in available_providers:
            providers = ['AzureExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # 创建 ONNX Runtime 会话
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # 获取输入输出信息
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        
        print(f"模型: {Path(onnx_path).name}")
        print(f"输入名称: {input_name}")
        print(f"输入形状: {input_info.shape}")
        print(f"使用提供者: {session.get_providers()}")
        
        # 预热
        print(f"预热 {warmup} 次...")
        for _ in range(warmup):
            session.run(None, {input_name: dummy_input})
        
        # 基准测试
        print(f"基准测试 {num_iterations} 次...")
        times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转换为毫秒
            
            if (i + 1) % 20 == 0:
                print(f"  已完成 {i+1}/{num_iterations}")
        
        # 计算统计
        times = np.array(times)
        stats = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }
        stats['fps'] = 1000 / stats['mean'] if stats['mean'] > 0 else 0
        
        # 显示结果
        print(f"\n✓ ONNX 基准测试完成:")
        print(f"  平均推理时间: {stats['mean']:.3f} ± {stats['std']:.3f} ms")
        print(f"  FPS: {stats['fps']:.1f}")
        print(f"  中位数: {stats['median']:.3f} ms")
        print(f"  范围: {stats['min']:.3f} - {stats['max']:.3f} ms")
        
        return stats
        
    except Exception as e:
        print(f"✗ ONNX 基准测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_tensorrt(trt_path, input_shape, num_iterations=100, warmup=10):
    """TensorRT 基准测试 - 如果 CUDA 不可用则跳过"""
    if not cuda_available:
        print(f"\n⚠ CUDA 不可用，跳过 TensorRT 测试")
        return None
    
    print(f"\n{'='*60}")
    print(f"TensorRT 基准测试 - 尺寸: {input_shape[2]}x{input_shape[3]}")
    print(f"{'='*60}")
    
    if not os.path.exists(trt_path):
        print(f"✗ TensorRT 引擎文件不存在: {trt_path}")
        return None
    
    try:
        print(f"TensorRT 版本: {trt.__version__}")
        print(f"模型: {Path(trt_path).name}")
        
        # 准备输入数据
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # 加载 TensorRT 引擎
        logger = trt.Logger(trt.Logger.WARNING)
        
        with open(trt_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print("✗ 加载 TensorRT 引擎失败")
            return None
        
        # 创建执行上下文
        context = engine.create_execution_context()
        
        # 使用传统绑定索引
        input_idx = 0
        output_idx = 1
        
        # 设置输入形状
        context.set_binding_shape(input_idx, dummy_input.shape)
        
        # 获取输出形状
        try:
            output_shape = context.get_binding_shape(output_idx)
        except:
            output_shape = (1, 20, input_shape[2], input_shape[3])  # 默认形状
        
        # 分配 GPU 内存
        h_input = cuda.pagelocked_empty(dummy_input.shape, dtype=np.float32)
        np.copyto(h_input, dummy_input)
        d_input = cuda.mem_alloc(h_input.nbytes)
        
        # 分配输出内存
        h_output = cuda.pagelocked_empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        bindings = [int(d_input), int(d_output)]
        stream = cuda.Stream()
        
        # 预热
        print(f"预热 {warmup} 次...")
        for _ in range(warmup):
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            stream.synchronize()
        
        # 基准测试
        print(f"基准测试 {num_iterations} 次...")
        times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            stream.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 毫秒
            
            if (i + 1) % 20 == 0:
                print(f"  已完成 {i+1}/{num_iterations}")
        
        # 清理内存
        d_input.free()
        d_output.free()
        
        # 计算统计
        times = np.array(times)
        stats = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }
        stats['fps'] = 1000 / stats['mean'] if stats['mean'] > 0 else 0
        
        # 显示结果
        print(f"\n✓ TensorRT 基准测试完成:")
        print(f"  平均推理时间: {stats['mean']:.3f} ± {stats['std']:.3f} ms")
        print(f"  FPS: {stats['fps']:.1f}")
        print(f"  中位数: {stats['median']:.3f} ms")
        print(f"  范围: {stats['min']:.3f} - {stats['max']:.3f} ms")
        
        return stats
        
    except Exception as e:
        print(f"✗ TensorRT 基准测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_results(onnx_stats, trt_stats, model_name, onnx_path, trt_path):
    """保存基准测试结果到文件"""
    result_file = "/mnt/nas/xinjiang/code/nyz/FreeNet-master/trained_models/benchmark_results.txt"
    
    with open(result_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TensorRT vs ONNX 基准测试结果\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"TensorRT 版本: {trt.__version__}\n")
        f.write(f"CUDA 可用: {cuda_available}\n")
        f.write(f"模型名称: {model_name}\n\n")
        
        f.write(f"ONNX 模型: {onnx_path}\n")
        f.write(f"TensorRT 引擎: {trt_path}\n\n")
        
        f.write("ONNX 性能指标:\n")
        f.write("-" * 40 + "\n")
        if onnx_stats:
            for key, value in onnx_stats.items():
                if isinstance(value, float):
                    if key == 'fps':
                        f.write(f"  {key:<10}: {value:.1f}\n")
                    else:
                        f.write(f"  {key:<10}: {value:.3f} ms\n")
                else:
                    f.write(f"  {key:<10}: {value}\n")
        else:
            f.write("  ONNX 测试失败\n")
        
        f.write("\nTensorRT 性能指标:\n")
        f.write("-" * 40 + "\n")
        if trt_stats:
            for key, value in trt_stats.items():
                if isinstance(value, float):
                    if key == 'fps':
                        f.write(f"  {key:<10}: {value:.1f}\n")
                    else:
                        f.write(f"  {key:<10}: {value:.3f} ms\n")
                else:
                    f.write(f"  {key:<10}: {value}\n")
            
            # 计算加速比
            if onnx_stats and onnx_stats['mean'] > 0 and trt_stats['mean'] > 0:
                speedup = onnx_stats['mean'] / trt_stats['mean']
                fps_ratio = trt_stats['fps'] / onnx_stats['fps']
                
                f.write("\n" + "="*70 + "\n")
                f.write("性能对比\n")
                f.write("="*70 + "\n")
                f.write(f"平均推理时间加速比: {speedup:.2f}x\n")
                f.write(f"FPS 提升: {fps_ratio:.2f}x\n")
                f.write(f"推理时间减少: {(1 - 1/speedup)*100:.1f}%\n")
        else:
            f.write("  TensorRT 测试失败或跳过\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("测试配置\n")
        f.write("="*70 + "\n")
        f.write(f"迭代次数: 100\n")
        f.write(f"预热次数: 10\n")
        f.write(f"CUDA 可用: {cuda_available}\n")
    
    print(f"\n✓ 详细结果已保存到: {result_file}")
    return result_file

def main():
    """主函数"""
    print("="*70)
    print("TensorRT vs ONNX 性能对比工具")
    print("="*70)
    
    # 配置参数
    ONNX_MODEL_PATH = "/mnt/nas/xinjiang/code/nyz/FreeNet-master/trained_models/model.onnx"
    TRT_MODEL_PATH = "/mnt/nas/xinjiang/code/nyz/FreeNet-master/trained_models/model.plan"
    
    # 定义要测试的三组尺寸
    test_shapes = [
        (1, 100, 380, 1050),
        (1, 100, 400, 2500), 
        (1, 100, 400, 1000)
    ]
    
    # 检查文件是否存在
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"✗ ONNX 模型文件不存在: {ONNX_MODEL_PATH}")
        return
    
    print(f"ONNX 模型: {ONNX_MODEL_PATH}")
    print(f"TensorRT 引擎: {TRT_MODEL_PATH}")
    print(f"CUDA 可用: {cuda_available}")
    
    all_results = []
    
    # 逐个测试每个尺寸
    for i, input_shape in enumerate(test_shapes):
        print(f"\n\n测试 {i+1}/{len(test_shapes)}: 尺寸 {input_shape[2]}x{input_shape[3]}")
        print("="*60)
        
        # 1. 测试 ONNX 性能
        onnx_stats = benchmark_onnx_ort(
            onnx_path=ONNX_MODEL_PATH,
            input_shape=input_shape,
            num_iterations=100
        )
        
        if onnx_stats:
            onnx_stats['height'] = input_shape[2]
            onnx_stats['width'] = input_shape[3]
            all_results.append(('ONNX', onnx_stats))
        
        # 2. 测试 TensorRT 性能
        trt_stats = benchmark_tensorrt(
            trt_path=TRT_MODEL_PATH,
            input_shape=input_shape,
            num_iterations=100
        )
        
        if trt_stats:
            trt_stats['height'] = input_shape[2]
            trt_stats['width'] = input_shape[3]
            all_results.append(('TensorRT', trt_stats))
        
        # 3. 保存当前尺寸的结果
        if onnx_stats or trt_stats:
            model_name = f"尺寸_{input_shape[2]}x{input_shape[3]}"
            save_results(onnx_stats, trt_stats, model_name, ONNX_MODEL_PATH, TRT_MODEL_PATH)
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)

if __name__ == "__main__":
    main()