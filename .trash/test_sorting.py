#!/usr/bin/env python3
"""
测试批量换脸工具的数字排序功能
"""

import sys

sys.path.insert(0, ".")
from batch_face_swap import BatchFaceSwap


def test_sorting():
    """测试数字提取和排序功能"""
    processor = BatchFaceSwap()

    test_files = [
        "1.0万-吃饭#宝宝辅食.mp4",
        "5.2万-热门视频.mp4",
        "3.8万-搞笑集锦.mp4",
        "8500-日常生活.mp4",
        "2.1万-旅游vlog.mp4",
        "950-测试视频.mkv",
        "12万-超级热门.mp4",
        "500-小视频.mp4",
    ]

    print("🧪 测试数字提取功能:")
    print("-" * 50)
    for filename in test_files:
        number = processor.extract_number_from_filename(filename)
        print(f"   {filename:<30} → {number:>10,.0f}")

    print()
    print("📊 排序测试结果（从大到小）:")
    print("-" * 50)

    # 测试排序功能
    sorted_files = processor.sort_videos_by_number(test_files)

    print()
    print("✅ 排序测试完成！")


if __name__ == "__main__":
    test_sorting()
