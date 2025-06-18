from typing import Any, List, Optional
import insightface

import modules.globals
from modules.typing import Frame, Face

FACE_ANALYSER = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        initialize_face_analyser()
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Optional[Face]:
    faces = FACE_ANALYSER.get(frame, max_num=1)
    return faces[0] if faces else None


def get_many_faces(frame: Frame) -> List[Face]:
    return FACE_ANALYSER.get(frame)


def initialize_face_analyser():
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        print("🧠 正在初始化面部分析器...")
        try:
            # 使用与GUI版本相同的初始化方式
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name="buffalo_l", providers=modules.globals.execution_providers
            )
            print("✅ 使用自定义执行提供者初始化成功")
        except Exception as e:
            print(f"⚠️  使用自定义执行提供者失败: {str(e)}")
            try:
                # 尝试使用默认设置
                FACE_ANALYSER = insightface.app.FaceAnalysis(name="buffalo_l")
                print("✅ 使用默认设置初始化成功")
            except Exception as e2:
                print(f"❌ 面部分析器初始化失败: {str(e2)}")
                print("📥 请确保模型文件已正确下载")
                raise e2

        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
        print("🎯 面部分析器准备完成")


def get_one_face_left(frame: Frame) -> Optional[Face]:
    faces = FACE_ANALYSER.get(frame)
    return min(faces, key=lambda x: x.bbox[0]) if faces else None


def get_one_face_right(frame: Frame) -> Optional[Face]:
    faces = FACE_ANALYSER.get(frame)
    return max(faces, key=lambda x: x.bbox[0]) if faces else None


def get_two_faces(frame: Frame) -> List[Face]:
    faces = FACE_ANALYSER.get(frame, max_num=2)
    return sorted(faces, key=lambda x: x.bbox[0])
