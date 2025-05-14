# coding=utf-8
import os
import json
from pathlib import Path


def calculate_char_positions(text_block):
    """
    根据文本段坐标自动计算每个字符的位置
    :param text_block: {
        "text": "姓名：张三",
        "top_left": [100, 200],
        "bottom_right": [300, 230]
    }
    :return: [
        {"char": "姓", "position": [100,200,120,220], "text_block": "姓名：张三"},
        ...
    ]
    """
    chars = []
    text = text_block["text"]
    x1, y1 = text_block["top_left"]
    x2, y2 = text_block["bottom_right"]

    # 计算每个字符的宽度（均匀分布）
    char_width = (x2 - x1) / len(text)

    for i, char in enumerate(text):
        char_x1 = x1 + i * char_width
        char_x2 = x1 + (i + 1) * char_width
        chars.append({
            "char": char,
            "position": [round(char_x1, 1), y1, round(char_x2, 1), y2],
            "text_block": text  # 保留所属文本段信息
        })
    return chars


def generate_rag_files(text_blocks_dir="ocr/PICL4", output_dir="rag/PICL4"):
    """
    从文本段坐标生成RAG文件（含完整医疗隐私类别）
    :param text_blocks_dir: 包含文本段坐标JSON的目录
    :param output_dir: 输出目录
    """
    Path(output_dir).mkdir(exist_ok=True)

    # 完整医疗隐私分类体系（分组+扁平化）
    MEDICAL_PRIVACY_CATEGORIES = {
        # 核心个人信息
        "个人标识": ["姓名", "性别", "出生日期", "身份证号", "手机号", "住址", "职业", "婚姻状态"],
        # 医疗标识符
        "医疗编号": ["病历号", "门诊号", "住院号", "检查编号", "床号", "病人编号"],
        # 时间信息
        "时间记录": ["报告日期", "检查日期", "入院日期", "出院日期", "出生日期"],
        # 医务人员
        "医疗人员": ["报告医生", "审核医生", "主治医生", "护理人员", "职称", "科室"],
        # 临床数据
        "临床信息": ["诊断结果", "用药记录", "检查项目", "医嘱内容"]
    }

    for block_file in Path(text_blocks_dir).glob("*.json"):
        with open(block_file, 'r', encoding='utf-8') as f:
            text_blocks = json.load(f)  # 读取文本段数据

        # 计算所有字符位置
        all_chars = []
        for block in text_blocks:
            all_chars.extend(calculate_char_positions(block))

        # 构建文本段→字符的映射
        text_char_map = {}
        for char in all_chars:
            text = char["text_block"]
            if text not in text_char_map:
                text_char_map[text] = []
            text_char_map[text].append(char)

        # 生成RAG文件
        rag_context = {
            "document_id": block_file.stem,
            "text_blocks": text_blocks,
            "character_details": all_chars,
            "text_character_mapping": text_char_map,
            "privacy_categories": {
                # 两种视图：分组版（供人工查阅）+扁平化列表（供程序处理）
                "grouped": MEDICAL_PRIVACY_CATEGORIES,
                "flat_list": [item for sublist in MEDICAL_PRIVACY_CATEGORIES.values() for item in sublist]
            },
            "metadata": {
                "generator": "MedicalRAG v2.0",
                "coordinate_precision": "1 decimal place",
                "coordinate_system": {
                    "text_blocks": ["top_left", "bottom_right"],
                    "characters": ["x1,y1,x2,y2"]
                }
            }
        }

        # 保存结果
        output_path = Path(output_dir) / f"{block_file.stem}_rag.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rag_context, f, ensure_ascii=False, indent=4)

        print(f"生成: {output_path} (含{len(all_chars)}个字符，{len(MEDICAL_PRIVACY_CATEGORIES)}类隐私标签)")


if __name__ == "__main__":
    generate_rag_files()