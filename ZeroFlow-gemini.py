# coding=utf-8
import os
import json
import concurrent.futures
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 全局变量
MAX_WORKERS = 4  # 根据CPU核心数调整，通常设置为CPU核心数的2-4倍
TIMEOUT = 60  # 单个文件处理超时时间(秒)


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


def get_raw_gemini_response(ocr_data, log_file):
    """获取Gemini的原始输出"""
    try:
        # 配置Gemini
        genai.configure(
            api_key='',
            transport="rest",
            client_options={"api_endpoint": ""},
        )
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = (f"请从以下OCR JSON数据中提取隐私信息及其坐标：\n"
                  f"{json.dumps(ocr_data, ensure_ascii=False, indent=2)}\n\n"
                  f"要求：\n"
                  f"1. 提取包含：姓名、性别、年龄、身份证号、手机号、住址、病历号、检查日期、医生信息\n"
                  f"2. 保留原始JSON结构（text/top_left/bottom_right）\n"
                  f"3. 返回JSON格式结果")

        # 记录输入
        log_message(log_file, f"发送给Gemini的OCR数据:\n{json.dumps(ocr_data, ensure_ascii=False, indent=2)}")
        log_message(log_file, f"使用的Prompt:\n{prompt}")

        # 调用Gemini API
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 4000,
                "temperature": 0
            }
        )

        raw_response = response.text
        log_message(log_file, f"Gemini原始响应:\n{raw_response}")

        return raw_response

    except Exception as e:
        error_msg = f"API调用出错: {str(e)}"
        log_message(log_file, error_msg, is_error=True)
        return json.dumps({"error": str(e)})


def process_single_file(input_path, output_path, log_file):
    """处理单个文件"""
    try:
        log_message(log_file, f"\n=== 开始处理文件: {os.path.basename(input_path)} ===")

        with open(input_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
            log_message(log_file, f"原始数据条目数: {len(ocr_data) if isinstance(ocr_data, list) else 1}")

        raw_output = get_raw_gemini_response(ocr_data, log_file)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(raw_output)

        log_message(log_file, f"文件处理完成，保存到: {output_path}")
        return True
    except Exception as e:
        error_msg = f"处理文件 {os.path.basename(input_path)} 时出错: {str(e)}"
        log_message(log_file, error_msg, is_error=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"error": str(e)}, f)
        return False


def process_json_files(input_dir="ocr/PICL4", output_dir="geminiprivacyjiu/PICL4"):
    """多线程处理目录下的所有JSON文件"""
    log_file = setup_logging(output_dir)
    log_message(log_file, f"开始处理目录: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 准备任务列表
    tasks = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_privacy.json")
            tasks.append((input_path, output_path, log_file))

    # 使用线程池处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for task in tasks:
            futures.append(executor.submit(process_single_file, *task))

        # 显示进度
        completed = 0
        total = len(tasks)
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            print(f"\r处理进度: {completed}/{total} ({completed / total:.1%})", end="")
            try:
                future.result(timeout=TIMEOUT)
            except concurrent.futures.TimeoutError:
                log_message(log_file, f"文件处理超时: {task[0]}", is_error=True)

    log_message(log_file, f"\n=== 所有文件处理完成 ===")
    print(f"\n处理完成，共处理 {len(tasks)} 个文件")


if __name__ == '__main__':
    process_json_files()