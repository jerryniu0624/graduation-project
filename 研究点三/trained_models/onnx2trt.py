import tensorrt as trt
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT with dynamic shapes")
    parser.add_argument('--onnx_path', type=str, default='trained_models/model.onnx', help='Path to ONNX model')
    parser.add_argument('--engine_path', type=str, default='trained_models/model.plan', help='Path to save TensorRT engine')
    parser.add_argument('--fp16_mode', type=bool, default=True, help='Use FP16 precision')
    parser.add_argument('--max_batch_size', type=int, default=1, help='Maximum batch size')
    parser.add_argument('--max_workspace_size', type=int, default=2<<30, help='Maximum workspace size in bytes')
    parser.add_argument('--min_h', type=int, default=128, help='Minimum height for dynamic shape')
    parser.add_argument('--min_w', type=int, default=128, help='Minimum width for dynamic shape')
    parser.add_argument('--opt_h', type=int, default=384, help='Optimal height for dynamic shape')
    parser.add_argument('--opt_w', type=int, default=384, help='Optimal width for dynamic shape')
    parser.add_argument('--max_h', type=int, default=1024, help='Maximum height for dynamic shape')
    parser.add_argument('--max_w', type=int, default=1024, help='Maximum width for dynamic shape')
    return parser.parse_args()

def onnx_to_tensorrt(args):
    """
    将ONNX模型转换为TensorRT引擎
    使用TensorRT 8.x+ API
    """
    onnx_path = args.onnx_path
    engine_path = args.engine_path
    fp16_mode = args.fp16_mode
    max_batch_size = args.max_batch_size
    
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    
    # 创建构建器
    builder = trt.Builder(TRT_LOGGER)
    
    # 创建网络定义
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 创建ONNX解析器
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 解析ONNX模型
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("解析失败:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print("✓ 成功解析ONNX模型")
    print(f"网络层数: {network.num_layers}")
    
    # 获取输入信息
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = input_tensor.shape
    print(f"输入名称: {input_name}")
    print(f"输入形状: {input_shape}")
    
    # 获取输出信息
    output_tensor = network.get_output(0)
    print(f"输出形状: {output_tensor.shape}")  # 从图片中看到是 Float[1,20,-1,-1]
    
    # 创建构建配置
    config = builder.create_builder_config()
    
    # 设置工作空间大小 - TensorRT 8.x+ 新API
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.max_workspace_size)
        print(f"✓ 设置工作空间大小: {args.max_workspace_size} bytes")
    except AttributeError as e:
        print(f"⚠ 无法设置工作空间大小: {e}")
        print("⚠ 将使用默认工作空间大小")
    
    # 设置精度
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ 启用FP16精度")
    
    # 设置动态形状
    profile = builder.create_optimization_profile()
    
    if len(input_shape) == 4:  # 确保是4D输入
        # 解析输入形状格式: [batch, channels, height, width]
        batch_size_idx = 0
        channels_idx = 1
        height_idx = 2
        width_idx = 3
        
        # 获取通道数
        if input_shape[channels_idx] > 0:
            num_channels = input_shape[channels_idx]
        else:
            num_channels = 100  # 默认值
        
        print(f"输入通道数: {num_channels}")
        
        # 设置动态形状范围
        min_shape = (1, num_channels, args.min_h, args.min_w)
        opt_shape = (max_batch_size, num_channels, args.opt_h, args.opt_w)
        max_shape = (max_batch_size, num_channels, args.max_h, args.max_w)
        
        print(f"\n设置动态形状范围:")
        print(f"  最小形状: {min_shape}")
        print(f"  最优形状: {opt_shape}")
        print(f"  最大形状: {max_shape}")
        
        # 设置优化配置文件
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    
    # 构建引擎 - 使用新的API
    print("\n正在构建TensorRT引擎，可能需要一些时间...")
    
    try:
        # 使用 build_serialized_network (TensorRT 8.x+推荐)
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("构建序列化引擎失败")
            return None
        
        # 保存引擎
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"✓ TensorRT引擎已保存到: {engine_path}")
        engine_size = os.path.getsize(engine_path) / 1024 / 1024
        print(f"引擎大小: {engine_size:.2f} MB")
        
        # 反序列化以获取引擎信息
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        if engine:
            print(f"\n引擎信息:")
            # TensorRT 8.x+ 移除了 max_batch_size 属性
            # 但我们可以通过其他方式获取信息
            
            # 获取绑定数量
            num_bindings = engine.num_bindings
            print(f"  绑定数量: {num_bindings}")
            
            # 获取绑定信息
            for i in range(num_bindings):
                try:
                    # TensorRT 8.x+ API
                    name = engine.get_binding_name(i)
                    is_input = engine.binding_is_input(i)
                    
                    # 获取形状
                    shape = engine.get_binding_shape(i)
                    
                    # 获取数据类型
                    dtype = engine.get_binding_dtype(i)
                    dtype_str = str(dtype).split('.')[-1]
                    
                    binding_type = "输入" if is_input else "输出"
                    print(f"  {binding_type} {i}: {name}")
                    print(f"    数据类型: {dtype_str}")
                    print(f"    形状: {shape}")
                    
                except Exception as e:
                    print(f"  获取绑定{i}信息失败: {e}")
        
        return engine
        
    except Exception as e:
        print(f"构建引擎失败: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    args = parse_args()
    
    print("="*60)
    print("ONNX 转 TensorRT 转换器")
    print("="*60)
    print(f"ONNX 路径: {args.onnx_path}")
    print(f"引擎路径: {args.engine_path}")
    print(f"FP16 模式: {args.fp16_mode}")
    print(f"最大批次大小: {args.max_batch_size}")
    print(f"动态H范围: {args.min_h} -> {args.opt_h} -> {args.max_h}")
    print(f"动态W范围: {args.min_w} -> {args.opt_w} -> {args.max_w}")
    print("="*60)
    
    if not os.path.exists(args.onnx_path):
        print(f"错误: ONNX文件不存在: {args.onnx_path}")
        exit(1)
    
    engine = onnx_to_tensorrt(args)
    
    if engine is not None:
        print("\n" + "="*60)
        print("转换成功!")
        print("="*60)
        print(f"\n引擎文件已保存: {args.engine_path}")
        print(f"您可以继续运行基准测试:")
        print(f"  python benchmark.py --engine_path {args.engine_path}")
    else:
        print("\n转换失败!")
        exit(1)