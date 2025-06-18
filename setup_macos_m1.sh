#!/bin/bash

echo "=========================================="
echo "iRoopDeepFaceCam MacOS M1 部署脚本"
echo "=========================================="

# 检查Python版本
echo "检查Python版本..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装。请先安装Python 3.9或更高版本。"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python版本: $PYTHON_VERSION"

# 检查系统架构
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "❌ 此脚本专为Apple M1/M2芯片设计，当前架构: $ARCH"
    exit 1
fi
echo "✅ 系统架构: $ARCH (Apple Silicon)"

# 检查ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "📦 安装ffmpeg..."
    if command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        echo "❌ 请先安装Homebrew，然后运行: brew install ffmpeg"
        exit 1
    fi
else
    echo "✅ FFmpeg已安装"
fi

# 创建虚拟环境
echo "📦 创建Python虚拟环境..."
if [ -d "venv" ]; then
    echo "⚠️  虚拟环境已存在，是否删除重建？(y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm -rf venv
        python3 -m venv venv
    fi
else
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "📦 升级pip..."
pip install --upgrade pip

# 创建适合M1的requirements
echo "📝 创建M1专用依赖配置..."
cat > requirements_m1.txt << 'EOF'
numpy==1.24.3
opencv-python==4.8.1.78
cv2_enumerate_cameras==1.1.15
onnx==1.16.0
insightface==0.7.3
psutil==5.9.8
tk==0.1.0
customtkinter==5.2.2
pillow==10.0.0
torch==2.0.1
torchvision==0.15.2
onnxruntime-silicon==1.16.3
tensorflow-macos==2.13.0
tensorflow-metal==1.0.1
opennsfw2==0.10.2
protobuf==4.23.2
tqdm==4.66.4
gfpgan==1.3.8
sympy>=1.7
EOF

# 安装依赖
echo "📦 安装Python依赖（这可能需要几分钟）..."
pip install -r requirements_m1.txt

# 检查模型文件夹
echo "📁 检查模型文件夹..."
if [ ! -d "models" ]; then
    mkdir -p models
fi

# 下载模型文件
echo "📥 下载模型文件..."
cd models

# 下载GFPGAN模型
if [ ! -f "GFPGANv1.4.pth" ]; then
    echo "📥 下载 GFPGAN 模型..."
    curl -L -o GFPGANv1.4.pth "https://huggingface.co/ivideogameboss/iroopdeepfacecam/resolve/main/GFPGANv1.4.pth"
else
    echo "✅ GFPGAN模型已存在"
fi

# 下载inswapper模型
if [ ! -f "inswapper_128_fp16.onnx" ]; then
    echo "📥 下载 inswapper 模型..."
    curl -L -o inswapper_128_fp16.onnx "https://huggingface.co/ivideogameboss/iroopdeepfacecam/resolve/main/inswapper_128_fp16.onnx"
else
    echo "✅ inswapper模型已存在"
fi

cd ..

# 创建启动脚本
echo "📝 创建启动脚本..."
cat > run_macos.sh << 'EOF'
#!/bin/bash
echo "启动 iRoopDeepFaceCam..."
source venv/bin/activate
python3 run.py --execution-provider coreml --max-memory 4
EOF

chmod +x run_macos.sh

# 测试运行
echo "🧪 测试安装..."
source venv/bin/activate
python3 -c "
try:
    import torch
    import cv2
    import insightface
    import onnxruntime
    print('✅ 所有主要依赖导入成功')
    print(f'✅ PyTorch版本: {torch.__version__}')
    print(f'✅ OpenCV版本: {cv2.__version__}')
    print(f'✅ ONNX Runtime版本: {onnxruntime.__version__}')
except Exception as e:
    print(f'❌ 导入错误: {e}')
    exit(1)
"

echo ""
echo "=========================================="
echo "🎉 部署完成！"
echo "=========================================="
echo ""
echo "启动方式："
echo "1. 使用启动脚本: ./run_macos.sh"
echo "2. 手动启动:"
echo "   source venv/bin/activate"
echo "   python3 run.py --execution-provider coreml"
echo ""
echo "注意事项："
echo "- 首次运行可能需要30秒初始化摄像头"
echo "- M1 Mac使用CoreML加速，性能优于CPU模式"
echo "- 建议在良好光线环境下使用"
echo "" 