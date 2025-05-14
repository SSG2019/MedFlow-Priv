# coding=utf-8
import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def setup_logging(output_dir):
    """创建日志目录和日志文件"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    return log_file


def log_message(log_file, message, is_error=False):
    """记录日志信息"""
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        prefix = "[ERROR]" if is_error else "[INFO]"
        f.write(f"{timestamp} {prefix} {message}\n")


def get_raw_gpt_response(ocr_data, log_file):
    """
    获取大模型的原始输出，不做任何修改
    :param ocr_data: 原始OCR JSON数据
    :param log_file: 日志文件路径
    :return: 大模型的原始输出
    """
    client = OpenAI(
        api_key=''
        , base_url='',
        timeout=60.0)  # 替换为实际API密钥

    prompt = (f"请从以下OCR JSON数据中提取隐私信息及其坐标：\n"
              f"{json.dumps(ocr_data, ensure_ascii=False, indent=2)}\n\n"
              f"要求：\n"
              f"1. 提取包含：姓名、性别、年龄、身份证号、手机号、住址、病历号、检查日期、医生信息\n"
              f"2. 保留原始JSON结构（text/top_left/bottom_right）\n"
              f"3. 返回JSON格式结果")

    # 记录输入
    log_message(log_file, f"发送给GPT的OCR数据:\n{json.dumps(ocr_data, ensure_ascii=False, indent=2)}")
    log_message(log_file, f"使用的Prompt:\n{prompt}")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个OCR隐私信息提取器，请直接返回JSON格式结果"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        raw_response = response.choices[0].message.content
        log_message(log_file, f"GPT原始响应:\n{raw_response}")

        return raw_response  # 直接返回原始字符串

    except Exception as e:
        error_msg = f"API调用出错: {str(e)}"
        log_message(log_file, error_msg, is_error=True)
        return json.dumps({"error": str(e)})  # 错误时返回错误信息的JSON


def process_json_files(input_dir="ocr/PICL4", output_dir="gptprivacyjiu/PICL4"):
    """处理目录下的所有JSON文件"""
    # 初始化日志
    log_file = setup_logging(output_dir)
    log_message(log_file, f"开始处理目录: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_privacy.json")

        try:
            log_message(log_file, f"\n=== 开始处理文件: {filename} ===")

            # 读取原始数据
            with open(input_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
                log_message(log_file, f"原始数据条目数: {len(ocr_data) if isinstance(ocr_data, list) else 1}")

            # 获取大模型原始输出
            raw_output = get_raw_gpt_response(ocr_data, log_file)

            # 直接保存原始输出（不做任何解析）
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(raw_output)  # 直接写入原始字符串

            log_message(log_file, f"文件处理完成，原始输出已保存到: {output_path}")
            print(f"处理完成: {filename} -> 原始输出已保存")

        except Exception as e:
            error_msg = f"处理文件 {filename} 时出错: {str(e)}"
            log_message(log_file, error_msg, is_error=True)
            print(error_msg)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"error": str(e)}, f)

    log_message(log_file, "=== 所有文件处理完成 ===")


if __name__ == '__main__':
    process_json_files()