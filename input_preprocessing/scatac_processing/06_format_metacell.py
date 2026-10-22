from __future__ import annotations

import argparse
import os
import re

import pandas as pd


def format_row_name(raw_name):

    if pd.isna(raw_name) or raw_name == "":
        return raw_name
    match = re.search(r'(\S+)\s+(\d+)\s+(\d+)', str(raw_name))
    if match:
        return f"{match.group(1)}:{match.group(2)}-{match.group(3)}"
    return raw_name


def main():
    parser = argparse.ArgumentParser(description="Format metacell matrix row names.")
    parser.add_argument("--input", "-i", type=str, default="metacell_accessibility_matrix_0_1mb.csv",
                        help="Input metacell CSV")
    parser.add_argument("--output", "-o", type=str, default="metacell_accessibility_matrix_0_1mb_formatted.csv",
                        help="Output formatted CSV")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 找不到文件 {args.input}")
        return

    print("正在读取文件并转换格式...")
    df = pd.read_csv(args.input, index_col=0)
    df.index = [format_row_name(idx) for idx in df.index]
    df.to_csv(args.output)
    print(f"处理完成，结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
