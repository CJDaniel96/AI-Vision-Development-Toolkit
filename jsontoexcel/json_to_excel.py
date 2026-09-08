#!/usr/bin/env python3
"""
將指定格式的 JSON 轉換成 Excel。

JSON 格式：
{
    "PartNumberA": ["Component1", "Component2"],
    "PartNumberB": ["Component3"]
}

Excel 欄位：
- Part Number
- Component Name
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import xlsxwriter


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="將 JSON 的 key 與 list value 轉換成兩欄 Excel 表格。"
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="輸入 JSON 檔案路徑",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="輸出 Excel 路徑；預設為輸入檔同名的 .xlsx",
    )
    parser.add_argument(
        "--separator",
        default=", ",
        help='Component Name 欄位使用的分隔符號，預設為 ", "',
    )
    parser.add_argument(
        "--sheet-name",
        default="Components",
        help='Excel 工作表名稱，預設為 "Components"',
    )
    return parser.parse_args()


def load_json(json_path: Path) -> dict[str, list[Any]]:
    if not json_path.is_file():
        raise FileNotFoundError(f"找不到輸入檔案：{json_path}")

    try:
        with json_path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON 格式錯誤，第 {exc.lineno} 行、第 {exc.colno} 欄：{exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("JSON 最外層必須是 object，也就是 key-value 結構。")

    invalid_keys = [
        str(key) for key, value in data.items() if not isinstance(value, list)
    ]
    if invalid_keys:
        preview = ", ".join(invalid_keys[:10])
        suffix = " ..." if len(invalid_keys) > 10 else ""
        raise ValueError(
            f"以下 key 的 value 不是 list：{preview}{suffix}"
        )

    return data


def stringify_component(value: Any) -> str:
    """
    將 list 內元素轉成可寫入儲存格的字串。
    dict/list 等複合型別會保留為 JSON 字串。
    """
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def validate_sheet_name(sheet_name: str) -> str:
    invalid_chars = set(r'[]:*?/\\')
    if not sheet_name:
        raise ValueError("工作表名稱不可為空。")
    if len(sheet_name) > 31:
        raise ValueError("Excel 工作表名稱不可超過 31 個字元。")
    if any(char in invalid_chars for char in sheet_name):
        raise ValueError(
            'Excel 工作表名稱不可包含以下字元：[]:*?/\\'
        )
    return sheet_name


def write_excel(
    data: dict[str, list[Any]],
    output_path: Path,
    separator: str,
    sheet_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = xlsxwriter.Workbook(output_path)
    try:
        worksheet = workbook.add_worksheet(sheet_name)

        header_format = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#D9EAF7",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        )
        cell_format = workbook.add_format(
            {
                "border": 1,
                "valign": "top",
            }
        )
        component_format = workbook.add_format(
            {
                "border": 1,
                "valign": "top",
                "text_wrap": True,
            }
        )

        worksheet.write(0, 0, "Part Number", header_format)
        worksheet.write(0, 1, "Component Name", header_format)

        for row_index, (part_number, components) in enumerate(
            data.items(), start=1
        ):
            component_names = separator.join(
                stringify_component(item) for item in components
            )
            worksheet.write(row_index, 0, str(part_number), cell_format)
            worksheet.write(row_index, 1, component_names, component_format)

        last_row = max(len(data), 1)
        worksheet.autofilter(0, 0, last_row, 1)
        worksheet.freeze_panes(1, 0)
        worksheet.set_column(0, 0, 24)
        worksheet.set_column(1, 1, 60)
        worksheet.set_row(0, 22)
    finally:
        workbook.close()


def main() -> int:
    args = parse_arguments()

    try:
        sheet_name = validate_sheet_name(args.sheet_name)
        data = load_json(args.input_json)
        output_path = args.output or args.input_json.with_suffix(".xlsx")

        if output_path.suffix.lower() != ".xlsx":
            output_path = output_path.with_suffix(".xlsx")

        write_excel(
            data=data,
            output_path=output_path,
            separator=args.separator,
            sheet_name=sheet_name,
        )
    except (FileNotFoundError, PermissionError, ValueError, OSError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1

    print(f"轉換完成：{output_path.resolve()}")
    print(f"共輸出 {len(data)} 筆資料。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
