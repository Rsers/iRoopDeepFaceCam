#!/usr/bin/env python3
"""
批量视频换脸工具
基于 iRoopDeepFaceCam 项目实现批量视频处理
支持截图中显示的所有功能：Face Enhancer、Keep fps、Keep Audio、Keep Frames等
"""

import os
import sys
import glob
import argparse
import time
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import shutil
import psutil
import re

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量以优化性能
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import modules.globals
import modules.metadata
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
    detect_fps,
    create_video,
    extract_frames,
    get_temp_frame_paths,
    restore_audio,
    create_temp,
    move_temp,
    clean_temp,
    normalize_output_path,
)
from modules.face_analyser import initialize_face_analyser


class BatchFaceSwap:
    """批量换脸处理类"""

    def __init__(self):
        self.setup_default_config()
        self.setup_system_thresholds()

    def setup_default_config(self):
        """设置默认配置，基于截图中的设置"""
        # 基本配置
        modules.globals.headless = True
        modules.globals.keep_fps = True
        modules.globals.keep_audio = True
        modules.globals.keep_frames = True
        modules.globals.many_faces = False
        modules.globals.nsfw_filter = False

        # 截图中显示的高级功能
        modules.globals.both_faces = False
        modules.globals.flip_faces = False
        modules.globals.detect_face_right = False
        modules.globals.show_target_face_box = False
        modules.globals.mouth_mask = True  # 截图中显示开启
        modules.globals.show_mouth_mask_box = False
        modules.globals.face_tracking = True  # 截图中显示开启

        # Face Enhancer 功能开启
        modules.globals.frame_processors = ["face_swapper", "face_enhancer"]
        modules.globals.fp_ui = {"face_enhancer": True}

        # 视频编码设置
        modules.globals.video_encoder = "libx264"
        modules.globals.video_quality = 18

        # 系统设置
        modules.globals.max_memory = 4 if sys.platform == "darwin" else 16
        modules.globals.execution_threads = 8
        # 在macOS上使用coreml，其他系统使用cpu
        if sys.platform == "darwin":
            modules.globals.execution_providers = [
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            modules.globals.execution_providers = ["CPUExecutionProvider"]

        # 面部跟踪参数
        modules.globals.mask_feather_ratio = 8
        modules.globals.mask_down_size = 0.50
        modules.globals.mask_size = 1
        modules.globals.sticky_face_value = 0.20
        modules.globals.pseudo_face_threshold = 0.20

        # 嵌入权重设置
        modules.globals.embedding_weight_size = 0.60
        modules.globals.weight_distribution_size = 1.00
        modules.globals.position_size = 0.40
        modules.globals.old_embedding_weight = 0.90
        modules.globals.new_embedding_weight = 0.10

    def setup_system_thresholds(self):
        """设置系统监控阈值"""
        # 温度阈值（摄氏度）
        self.temp_threshold = 75.0  # CPU温度超过75度时等待
        self.temp_safe = 65.0  # 降到65度以下时继续

        # 内存使用率阈值（百分比）
        self.memory_threshold = 85.0  # 内存使用率超过85%时等待
        self.memory_safe = 70.0  # 降到70%以下时继续

        # CPU使用率阈值（百分比）
        self.cpu_threshold = 90.0  # CPU使用率超过90%时等待
        self.cpu_safe = 60.0  # 降到60%以下时继续

        # 监控间隔
        self.monitor_interval = 30  # 每30秒检查一次

    def validate_inputs(
        self, source_path: str, input_dir: str, output_dir: str
    ) -> bool:
        """验证输入参数"""
        if not os.path.isfile(source_path):
            print(f"❌ 错误：源人脸图片不存在: {source_path}")
            return False

        if not is_image(source_path):
            print(f"❌ 错误：源文件不是有效的图片格式: {source_path}")
            return False

        if not os.path.isdir(input_dir):
            print(f"❌ 错误：输入视频目录不存在: {input_dir}")
            return False

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        return True

    def get_video_files(self, input_dir: str) -> List[str]:
        """获取输入目录中的所有视频文件"""
        video_extensions = ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv", "*.flv"]
        video_files = []

        for ext in video_extensions:
            pattern = os.path.join(input_dir, "**", ext)
            video_files.extend(glob.glob(pattern, recursive=True))
            # 也搜索大写扩展名
            pattern = os.path.join(input_dir, "**", ext.upper())
            video_files.extend(glob.glob(pattern, recursive=True))

        # 去重
        video_files = list(set(video_files))

        # 按照文件名中的数字排序（从大到小）
        return self.sort_videos_by_number(video_files)

    def extract_number_from_filename(self, filepath: str) -> float:
        """从文件名中提取数字，支持万、千等单位"""
        import re

        filename = os.path.basename(filepath)

        # 匹配开头的数字和单位，例如：1.0万、2.5万、500等
        pattern = r"^(\d+(?:\.\d+)?)(万|千)?"
        match = re.match(pattern, filename)

        if match:
            number = float(match.group(1))
            unit = match.group(2)

            # 转换单位
            if unit == "万":
                number *= 10000
            elif unit == "千":
                number *= 1000

            return number

        # 如果没有匹配到数字，返回0（排在最后）
        return 0

    def sort_videos_by_number(self, video_files: List[str]) -> List[str]:
        """按照文件名中的数字从大到小排序"""

        def sort_key(filepath):
            return self.extract_number_from_filename(filepath)

        # 按照数字从大到小排序（reverse=True）
        sorted_files = sorted(video_files, key=sort_key, reverse=True)

        # 打印排序结果用于调试
        print("📊 视频文件按数字排序结果（从大到小）：")
        for i, filepath in enumerate(sorted_files[:10], 1):  # 只显示前10个
            filename = os.path.basename(filepath)
            number = self.extract_number_from_filename(filepath)
            print(f"   {i}. {filename} (数值: {number:,.0f})")

        if len(sorted_files) > 10:
            print(f"   ... 还有 {len(sorted_files) - 10} 个文件")

        return sorted_files

    def check_laptop_lid_open(self) -> bool:
        """检查并提醒用户打开笔记本盖子"""
        print("\n" + "=" * 60)
        print("🔥 重要提醒：散热检查")
        print("=" * 60)
        print("📱 为了确保系统安全运行，请确保：")
        print("   1. 笔记本盖子已经完全打开")
        print("   2. 散热风扇可以正常运行")
        print("   3. 通风口没有被遮挡")
        print("   4. 周围环境通风良好")
        print("")

        while True:
            confirm = (
                input("❓ 请确认笔记本盖子已打开并散热良好 (输入 'yes' 继续): ")
                .strip()
                .lower()
            )
            if confirm in ["yes", "y", "是", "确认"]:
                print("✅ 确认完成，开始系统监控...")
                return True
            elif confirm in ["no", "n", "否", "取消"]:
                print("❌ 请打开笔记本盖子后再运行程序")
                return False
            else:
                print("请输入 'yes' 或 'no'")

    def get_cpu_temperature(self) -> Optional[float]:
        """获取CPU温度（摄氏度）"""
        try:
            # 尝试使用powermetrics获取温度，使用预设密码
            cmd = ["sudo", "-S", "powermetrics", "-n", "1", "-s", "cpu_power"]
            result = subprocess.run(
                cmd,
                input="cbl058518\n",
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # 解析输出查找CPU温度
                for line in result.stdout.split("\n"):
                    if "CPU die temperature" in line:
                        temp_match = re.search(r"(\d+\.\d+)", line)
                        if temp_match:
                            return float(temp_match.group(1))

            # 备用方法：使用istats（如果安装了）
            result = subprocess.run(
                ["istats", "cpu", "temp"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                temp_match = re.search(r"(\d+\.\d+)°C", result.stdout)
                if temp_match:
                    return float(temp_match.group(1))

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            pass

        return None

    def get_cpu_usage(self) -> float:
        """获取CPU使用率（百分比）"""
        try:
            return psutil.cpu_percent(interval=1)
        except:
            return 0.0

    def get_memory_usage(self) -> float:
        """获取内存使用率（百分比）"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception:
            # 备用方法：使用vm_stat命令
            try:
                result = subprocess.run(["vm_stat"], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.split("\n")
                    pages_free = 0
                    pages_active = 0
                    pages_inactive = 0

                    for line in lines:
                        if "Pages free:" in line:
                            pages_free = int(line.split(":")[1].strip().rstrip("."))
                        elif "Pages active:" in line:
                            pages_active = int(line.split(":")[1].strip().rstrip("."))
                        elif "Pages inactive:" in line:
                            pages_inactive = int(line.split(":")[1].strip().rstrip("."))

                    if pages_free + pages_active + pages_inactive > 0:
                        memory_usage = (
                            (pages_active + pages_inactive)
                            / (pages_free + pages_active + pages_inactive)
                        ) * 100
                        return round(memory_usage, 1)
            except Exception as e:
                print(f"⚠️  无法获取内存使用率: {e}")

            return 0.0

    def cleanup_memory(self):
        """清理系统内存"""
        try:
            print("🧹 正在清理系统内存...")
            # 执行purge命令清理内存，使用预设密码
            result = subprocess.run(
                ["sudo", "-S", "purge"],
                input="cbl058518\n",
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                print("✅ 内存清理完成")
            else:
                print(f"⚠️  内存清理可能失败: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⚠️  内存清理超时，继续执行")
        except Exception as e:
            print(f"⚠️  内存清理出错: {e}")

    def system_rest(self, rest_seconds: int = 180):
        """系统休息并显示状态"""
        print(f"💤 系统休息 {rest_seconds} 秒...")

        # 显示休息前的系统状态
        memory_before = self.get_memory_usage()
        print(f"📊 休息前内存使用率: {memory_before}%")

        # 清理内存
        self.cleanup_memory()

        # 等待一秒让清理生效
        time.sleep(1)

        # 显示清理后的内存状态
        memory_after = self.get_memory_usage()
        memory_freed = memory_before - memory_after
        print(f"📊 清理后内存使用率: {memory_after}% (释放了 {memory_freed:.1f}%)")

        # 倒计时休息
        for remaining in range(rest_seconds, 0, -30):
            if remaining <= 30:
                print(f"⏰ 还有 {remaining} 秒继续下一个任务...")
                time.sleep(remaining)
                break
            else:
                print(f"⏰ 还有 {remaining} 秒继续下一个任务...")
                time.sleep(30)

        print("🚀 休息完成，继续处理...")

    def get_system_status(self) -> Dict[str, float]:
        """获取系统状态"""
        cpu_temp = self.get_cpu_temperature()
        cpu_usage = self.get_cpu_usage()
        memory_usage = self.get_memory_usage()

        return {
            "temperature": cpu_temp,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
        }

    def is_system_safe(self, status: Dict[str, float]) -> bool:
        """检查系统是否处于安全状态"""
        # 检查温度
        if status["temperature"] is not None:
            if status["temperature"] > self.temp_threshold:
                return False

        # 检查内存使用率
        if status["memory_usage"] > self.memory_threshold:
            return False

        # 检查CPU使用率
        if status["cpu_usage"] > self.cpu_threshold:
            return False

        return True

    def wait_for_safe_system(self) -> None:
        """等待系统达到安全状态"""
        status = self.get_system_status()

        # 显示当前状态
        temp_str = f"{status['temperature']:.1f}°C" if status["temperature"] else "未知"
        print(
            f"📊 系统状态 - 温度: {temp_str} | CPU: {status['cpu_usage']:.1f}% | 内存: {status['memory_usage']:.1f}%"
        )

        if self.is_system_safe(status):
            print("✅ 系统状态正常")
            return

        # 系统状态不安全，需要等待
        print("⚠️  系统状态需要等待:")

        if (
            status["temperature"] is not None
            and status["temperature"] > self.temp_threshold
        ):
            print(
                f"   🔥 CPU温度过高: {status['temperature']:.1f}°C (阈值: {self.temp_threshold}°C)"
            )

        if status["memory_usage"] > self.memory_threshold:
            print(
                f"   💾 内存使用率过高: {status['memory_usage']:.1f}% (阈值: {self.memory_threshold}%)"
            )

        if status["cpu_usage"] > self.cpu_threshold:
            print(
                f"   ⚡ CPU使用率过高: {status['cpu_usage']:.1f}% (阈值: {self.cpu_threshold}%)"
            )

        print("🕒 等待系统降温/减负...")

        # 循环等待直到系统安全
        while True:
            time.sleep(self.monitor_interval)

            status = self.get_system_status()
            temp_str = (
                f"{status['temperature']:.1f}°C" if status["temperature"] else "未知"
            )
            print(
                f"📊 重新检查 - 温度: {temp_str} | CPU: {status['cpu_usage']:.1f}% | 内存: {status['memory_usage']:.1f}%"
            )

            if self.is_system_safe(status):
                print("✅ 系统状态恢复正常，继续处理...")
                break

    def process_single_video(
        self, source_path: str, target_path: str, output_path: str
    ) -> bool:
        """处理单个视频"""
        try:
            print(f"📹 开始处理: {os.path.basename(target_path)}")

            # 设置全局变量
            modules.globals.source_path = source_path
            modules.globals.target_path = target_path
            modules.globals.output_path = output_path

            # 初始化帧处理器
            frame_processors = get_frame_processors_modules(
                modules.globals.frame_processors
            )
            for frame_processor in frame_processors:
                if not frame_processor.pre_start():
                    print(f"❌ 帧处理器初始化失败: {frame_processor.NAME}")
                    return False

            print("   📁 创建临时资源...")
            create_temp(target_path)

            print("   🎬 提取视频帧...")
            extract_frames(target_path)
            temp_frame_paths = get_temp_frame_paths(target_path)

            if not temp_frame_paths:
                print("   ❌ 错误：无法提取视频帧")
                return False

            print(f"   🔄 处理 {len(temp_frame_paths)} 帧...")

            # 应用帧处理器
            for frame_processor in frame_processors:
                print(f"   ⚙️  应用 {frame_processor.NAME}...")
                frame_processor.process_video(source_path, temp_frame_paths)

            # 检测FPS
            if modules.globals.keep_fps:
                print("   📊 检测原始FPS...")
                fps = detect_fps(target_path)
                print(f"   🎯 使用 {fps} FPS 创建视频...")
                create_video(target_path, fps)
            else:
                print("   🎯 使用默认 30.0 FPS 创建视频...")
                create_video(target_path)

            # 恢复音频
            if modules.globals.keep_audio:
                print("   🔊 恢复音频...")
                restore_audio(target_path, output_path)
            else:
                move_temp(target_path, output_path)

            # 清理临时文件
            if not modules.globals.keep_frames:
                clean_temp(target_path)

            if os.path.isfile(output_path):
                print(f"   ✅ 处理完成: {os.path.basename(output_path)}")
                return True
            else:
                print(f"   ❌ 处理失败: 输出文件未生成")
                return False

        except Exception as e:
            print(f"   ❌ 处理过程中出现错误: {str(e)}")
            # 清理临时文件
            try:
                clean_temp(target_path)
            except:
                pass
            return False

    def process_batch(
        self,
        source_path: str,
        input_dir: str,
        output_dir: str,
        recursive: bool = True,
        rest_time: int = 180,
        auto_clean: bool = True,
    ) -> None:
        """批量处理视频"""
        print("🚀 开始批量视频换脸处理")
        print("=" * 50)
        print(f"📷 源人脸图片: {source_path}")
        print(f"📁 输入目录: {input_dir}")
        print(f"📁 输出目录: {output_dir}")
        print(f"🔄 递归搜索: {'是' if recursive else '否'}")
        print(
            f"✨ Face Enhancer: {'开启' if modules.globals.fp_ui.get('face_enhancer') else '关闭'}"
        )
        print(f"🎵 保持音频: {'是' if modules.globals.keep_audio else '否'}")
        print(f"📊 保持FPS: {'是' if modules.globals.keep_fps else '否'}")
        print("=" * 50)

        # 检查笔记本盖子是否打开
        if not self.check_laptop_lid_open():
            print("❌ 用户取消操作")
            return

        # 验证输入
        if not self.validate_inputs(source_path, input_dir, output_dir):
            return

        # 获取视频文件列表
        print("🔍 搜索视频文件...")
        if recursive:
            video_files = self.get_video_files(input_dir)
        else:
            video_files = []
            for ext in ["mp4", "avi", "mkv", "mov", "wmv", "flv"]:
                video_files.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
                video_files.extend(
                    glob.glob(os.path.join(input_dir, f"*.{ext.upper()}"))
                )

        if not video_files:
            print("❌ 未找到任何视频文件")
            return

        print(f"📹 找到 {len(video_files)} 个视频文件")

        # 批量处理前的系统状态检查和准备
        print("📊 批量处理前系统状态检查:")
        initial_memory = self.get_memory_usage()
        print(f"   💾 当前内存使用率: {initial_memory}%")

        if initial_memory > 75:
            print("⚠️  内存使用率较高，建议先清理系统内存")
            user_input = input("是否现在清理内存？(y/n): ").strip().lower()
            if user_input in ["y", "yes", "是"]:
                self.cleanup_memory()
                time.sleep(2)
                new_memory = self.get_memory_usage()
                print(f"   💾 清理后内存使用率: {new_memory}%")

        print("🧠 初始化面部分析器...")
        initialize_face_analyser()

        # 处理统计
        success_count = 0
        failed_count = 0

        # 逐个处理视频
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] 处理视频:")

            # 生成输出路径，使用自定义命名格式
            rel_path = os.path.relpath(video_path, input_dir)
            video_dir = os.path.dirname(rel_path)
            video_name = os.path.splitext(os.path.basename(rel_path))[0]
            output_filename = f"{video_name}-swapped-iroop.mp4"
            output_path = os.path.join(output_dir, video_dir, output_filename)

            # 创建输出子目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 处理视频
            if self.process_single_video(source_path, video_path, output_path):
                success_count += 1
                print(f"✅ 成功完成第 {i} 个视频")

                # 根据最佳实践指南，每个任务完成后进行系统监控和维护
                if auto_clean and i < len(
                    video_files
                ):  # 不是最后一个视频才进行监控和休息
                    print("=" * 60)
                    print("🔄 任务完成后系统监控与维护")

                    # 检查系统状态，如果不安全则等待至安全状态
                    print("🔍 检查系统状态...")
                    self.wait_for_safe_system()

                    # 进行常规的内存清理和休息
                    self.system_rest(rest_seconds=rest_time)
                    print("=" * 60)
            else:
                failed_count += 1
                print(f"❌ 第 {i} 个视频处理失败")

                # 即使失败也要进行系统检查和清理
                if auto_clean and i < len(video_files):
                    print("=" * 60)
                    print("🔄 失败后系统检查与清理")

                    # 检查系统状态
                    print("🔍 检查系统状态...")
                    self.wait_for_safe_system()

                    # 休息时间减半
                    self.system_rest(rest_seconds=rest_time // 2)
                    print("=" * 60)

        # 输出统计结果
        print("\n" + "=" * 50)
        print("📊 批量处理完成!")
        print(f"✅ 成功: {success_count} 个")
        print(f"❌ 失败: {failed_count} 个")
        print(f"📁 输出目录: {output_dir}")

        # 最终系统状态检查和清理
        print("\n🔍 最终系统状态:")
        final_memory = self.get_memory_usage()
        print(f"   💾 最终内存使用率: {final_memory}%")

        print("\n🧹 执行最终系统清理...")
        self.cleanup_memory()
        time.sleep(2)
        cleaned_memory = self.get_memory_usage()
        print(f"   💾 清理后内存使用率: {cleaned_memory}%")

        print("\n💡 建议:")
        print("   1. 让系统休息5-10分钟再进行其他重型任务")
        print("   2. 检查输出视频的质量")
        print("   3. 清理不需要的临时文件")

        print("=" * 50)


def interactive_mode():
    """交互式模式"""
    print("🎭 批量视频换脸工具 - 交互模式")
    print("=" * 50)

    # 第一次交互：询问源图片
    while True:
        print("📷 第一步：选择源人脸图片")
        source_path = input("请输入源人脸图片路径 (支持 JPG, PNG 格式): ").strip()

        if not source_path:
            print("❌ 路径不能为空，请重新输入")
            continue

        if not os.path.isfile(source_path):
            print(f"❌ 文件不存在: {source_path}")
            continue

        if not is_image(source_path):
            print(f"❌ 不是有效的图片格式: {source_path}")
            continue

        print(f"✅ 源图片已选择: {source_path}")
        break

    print()

    # 第二次交互：询问输入视频目录
    while True:
        print("📁 第二步：选择待换脸的视频目录")
        input_dir = input("请输入视频目录路径: ").strip()

        if not input_dir:
            print("❌ 目录路径不能为空，请重新输入")
            continue

        if not os.path.isdir(input_dir):
            print(f"❌ 目录不存在: {input_dir}")
            continue

        # 检查是否有视频文件
        batch_processor = BatchFaceSwap()
        video_files = batch_processor.get_video_files(input_dir)
        if not video_files:
            print(f"❌ 目录中未找到任何视频文件: {input_dir}")
            print("支持的格式: MP4, AVI, MKV, MOV, WMV, FLV")
            continue

        print(f"✅ 视频目录已选择: {input_dir}")
        print(f"📹 找到 {len(video_files)} 个视频文件")
        break

    print()

    # 自动生成输出目录
    print("💾 第三步：自动设置输出目录")
    input_dir_name = os.path.basename(input_dir.rstrip("/\\"))
    output_dir = os.path.join(input_dir, f"{input_dir_name}-swapped-iroop")

    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ 输出目录已自动创建: {output_dir}")
    except Exception as e:
        print(f"❌ 无法创建输出目录: {str(e)}")
        return

    print()
    print("📋 配置确认:")
    print(f"📷 源图片: {source_path}")
    print(f"📁 输入目录: {input_dir}")
    print(f"💾 输出目录: {output_dir}")
    print(f"🏷️  输出格式: xxx-swapped-iroop.mp4")
    print()

    # 确认开始处理
    while True:
        confirm = input("是否开始批量处理？(y/n): ").strip().lower()
        if confirm in ["y", "yes", "是"]:
            break
        elif confirm in ["n", "no", "否"]:
            print("🚫 用户取消操作")
            return
        else:
            print("请输入 y 或 n")

    # 开始批量处理
    batch_processor.process_batch(source_path, input_dir, output_dir)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量视频换脸工具 - 基于 iRoopDeepFaceCam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 交互模式（推荐）
  python batch_face_swap.py
  
  # 命令行模式
  python batch_face_swap.py -s face.jpg -i videos/ -o output/
  python batch_face_swap.py -s face.jpg -i videos/ -o output/ --no-recursive
  python batch_face_swap.py -s face.jpg -i videos/ -o output/ --no-audio --no-enhancer
        """,
    )

    parser.add_argument("-s", "--source", help="源人脸图片路径")
    parser.add_argument("-i", "--input", help="输入视频目录")
    parser.add_argument("-o", "--output", help="输出视频目录")
    parser.add_argument("--no-recursive", action="store_true", help="不递归搜索子目录")
    parser.add_argument("--no-audio", action="store_true", help="不保持原始音频")
    parser.add_argument("--no-fps", action="store_true", help="不保持原始FPS")
    parser.add_argument("--no-enhancer", action="store_true", help="禁用Face Enhancer")
    parser.add_argument("--keep-frames", action="store_true", help="保留临时帧文件")
    parser.add_argument(
        "--video-quality",
        type=int,
        default=18,
        choices=range(52),
        metavar="[0-51]",
        help="视频质量 (0-51, 默认18)",
    )
    parser.add_argument(
        "--rest-time",
        type=int,
        default=180,
        help="每个视频完成后的休息时间(秒, 默认180秒即3分钟)",
    )
    parser.add_argument(
        "--no-auto-clean", action="store_true", help="禁用自动内存清理和休息"
    )

    args = parser.parse_args()

    # 如果没有提供参数，进入交互模式
    if not args.source and not args.input and not args.output:
        interactive_mode()
        return

    # 检查命令行模式的必需参数
    if not all([args.source, args.input, args.output]):
        print("❌ 命令行模式需要提供所有必需参数: -s, -i, -o")
        print("或者不提供任何参数以进入交互模式")
        parser.print_help()
        return

    # 创建批量处理器
    batch_processor = BatchFaceSwap()

    # 根据命令行参数调整配置
    if args.no_audio:
        modules.globals.keep_audio = False
    if args.no_fps:
        modules.globals.keep_fps = False
    if args.no_enhancer:
        modules.globals.frame_processors = ["face_swapper"]
        modules.globals.fp_ui["face_enhancer"] = False
    if args.keep_frames:
        modules.globals.keep_frames = True

    modules.globals.video_quality = args.video_quality

    # 开始批量处理
    batch_processor.process_batch(
        source_path=args.source,
        input_dir=args.input,
        output_dir=args.output,
        recursive=not args.no_recursive,
        rest_time=args.rest_time,
        auto_clean=not args.no_auto_clean,
    )


if __name__ == "__main__":
    main()
