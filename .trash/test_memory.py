#!/usr/bin/env python3
"""
测试内存管理功能
"""

import sys

sys.path.insert(0, ".")
from batch_face_swap import BatchFaceSwap


def test_memory_management():
    """测试内存管理功能"""
    processor = BatchFaceSwap()

    print("📊 测试内存使用率获取:")
    memory_usage = processor.get_memory_usage()
    print(f"   当前内存使用率: {memory_usage}%")

    print()
    print("💤 测试系统休息功能 (10秒演示):")
    processor.system_rest(rest_seconds=10)

    print()
    print("✅ 内存管理功能测试完成！")


if __name__ == "__main__":
    test_memory_management()
