"""
统一 RAG 推理脚本 - 支持多种 RAG 方法和切片并行执行

用法:
    python -m scripts.run --method BasicRAG --model-path /path/to/model
    python -m scripts.run --method FlareRAG --model-path /path/to/model --task-id 0 --total-tasks 4
"""

import sys
import os
import re
import argparse
from datetime import datetime

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, get_project_root
from rag import RAG_METHODS


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一 RAG 推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
支持的 RAG 方法:
{chr(10).join(f"  - {name}: {cls.__doc__.strip().split(chr(10))[0] if cls.__doc__ else ''}" for name, cls in RAG_METHODS.items())}

示例:
  python -m scripts.run --method BasicRAG --model-path /path/to/model
  python -m scripts.run --method FlareRAG --model-path /path/to/model --task-id 0 --total-tasks 4
  python -m scripts.run --method FixedLengthRAG --model-path /path/to/model --retrieval-interval 200
""",
    )

    parser.add_argument("--method", type=str, required=True, choices=list(RAG_METHODS.keys()), help="RAG 方法")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--task-id", type=int, default=0, help="当前任务 ID（用于切片并行）")
    parser.add_argument("--total-tasks", type=int, default=1, help="总任务数")
    parser.add_argument("--output-file", type=str, default=None, help="输出文件名")
    parser.add_argument("--batch-size", type=int, default=5, help="每 N 条保存一次")
    parser.add_argument("--retrieval-interval", type=int, default=50, help="FixedLengthRAG 检索间隔")
    parser.add_argument("--attention-threshold-percentile", type=float, default=75.0, help="TokenRAG 注意力阈值百分位")
    parser.add_argument("--retrieval-top-k", type=int, default=None, help="检索 top-k")
    parser.add_argument("--max-length", type=int, default=2560, help="最大生成长度")
    parser.add_argument("--input-file", type=str, default=None, help="输入文件路径")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--truncate-chinese", action="store_true", help="截断中文字符之前的内容")

    return parser.parse_args()


def create_rag_instance(method: str, config: Config, retrieval_interval: int = 50,
                        attention_threshold_percentile: float = 75.0, retrieval_top_k: int = 3):
    """根据方法名创建 RAG 实例"""
    rag_class = RAG_METHODS[method]

    if method == "FixedLengthRAG":
        return rag_class(config, retrieval_interval=retrieval_interval)
    elif method == "FixedSentenceRAG":
        return rag_class(config, sentence_delimiter=";")
    elif method == "TokenRAG":
        instance = rag_class(config)
        if attention_threshold_percentile is not None:
            instance.attention_threshold_percentile = attention_threshold_percentile
        if retrieval_top_k is not None:
            instance.retrieval_top_k = retrieval_top_k
        return instance
    else:
        return rag_class(config)


def truncate_at_chinese(s):
    match = re.search(r"[\u4e00-\u9fff]", s)
    if match:
        return s[: match.start()]
    return s


def run_inference(args):
    method = args.method
    model_path = args.model_path
    task_id = args.task_id
    total_tasks = args.total_tasks
    batch_size = args.batch_size
    truncate_chinese = True

    root = get_project_root()

    # 加载配置
    config_path = args.config if args.config else str(root / "config/default.yaml")
    config = Config.from_yaml(config_path) if os.path.exists(config_path) else Config()
    config.llm.model_name = model_path

    # 路径设置
    input_file = args.input_file if args.input_file else config.resolve_path(config.rag.requirements_test_path)
    output_dir = args.output_dir if args.output_dir else config.resolve_path(config.output.result_dir)
    output_file = args.output_file or f"generated_code_{os.path.basename(model_path)}_{method.lower()}_attn{int(args.attention_threshold_percentile)}_top{args.retrieval_top_k}.xlsx"

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = f"task{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(os.path.join(log_dir, log_filename), "w", encoding="utf-8")

    def log_print(*msg, **kwargs):
        print(*msg, **kwargs)
        if log_file:
            print(*msg, **kwargs, file=log_file)
            log_file.flush()

    log_print("=" * 80)
    log_print(f"RAG 统一推理脚本")
    log_print(f"RAG 方法: {method}")
    log_print(f"模型路径: {model_path}")
    log_print(f"输入文件: {input_file}")
    log_print(f"输出文件: {output_file}")
    log_print(f"任务分片: {task_id + 1}/{total_tasks}")
    log_print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 80)

    # 创建 RAG 实例
    rag = create_rag_instance(
        method, config, args.retrieval_interval, args.attention_threshold_percentile, args.retrieval_top_k
    )

    # 读取数据
    from utils.file_utils import FileReader, ExcelWriter

    reader = FileReader(input_file)
    writer = ExcelWriter(output_dir)
    df = reader.read_excel()

    total_rows = len(df)
    rows_per_task = (total_rows + total_tasks - 1) // total_tasks
    start_idx = task_id * rows_per_task
    end_idx = min(start_idx + rows_per_task, total_rows)
    df_slice = df.iloc[start_idx:end_idx]

    log_print(f"\n数据分片: 总量={total_rows}, 范围=[{start_idx}, {end_idx}), 当前={len(df_slice)}")

    results = []
    total = len(df_slice)

    for idx, (index, row) in enumerate(df_slice.iterrows()):
        req_id = row["req_id"]
        req_text = row["req_text"]

        log_print(f"\n[{idx + 1}/{total}] (全局索引: {index}) 处理需求: {req_id}")

        query = req_id + "\n" + req_text
        if truncate_chinese:
            query = truncate_at_chinese(query)

        try:
            rag.clear_cache()
            code = rag.inference(query)
            retrieval_count = getattr(rag, "retrieval_count", -1)

            results.append({
                "req_id": req_id,
                "req_text": req_text,
                "generated_code": code,
                "retrieval_count": retrieval_count,
                "method": method,
            })

            log_print(f"  生成成功 | 检索次数: {retrieval_count}")

            if (idx + 1) % batch_size == 0:
                writer.write(results, output_file)
                log_print(f"  >>> 已保存 {idx + 1} 条结果")

        except Exception as e:
            log_print(f"  处理需求 {req_id} 时发生错误: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                "req_id": req_id,
                "req_text": req_text,
                "generated_code": f"ERROR: {str(e)}",
                "retrieval_count": -1,
                "method": method,
            })

    writer.write(results, output_file)

    successful = sum(1 for r in results if r["retrieval_count"] >= 0)
    total_retrievals = sum(r["retrieval_count"] for r in results if r["retrieval_count"] >= 0)
    avg_retrievals = total_retrievals / successful if successful > 0 else 0

    log_print("\n" + "=" * 80)
    log_print(f"完成！共处理 {total} 条需求")
    log_print(f"成功: {successful} | 失败: {total - successful}")
    log_print(f"总检索次数: {total_retrievals} | 平均检索次数: {avg_retrievals:.2f}")
    log_print(f"结果保存至: {output_dir}/{output_file}")
    log_print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 80 + "\n")

    if log_file:
        log_file.close()

    return results


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
