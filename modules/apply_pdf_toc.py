#!/usr/bin/env python3
"""
解析JSON格式的目录信息并将其添加到PDF文件中
基于pdf.tocgen项目的pdftocgen和pdftocio工具的现代化重构
"""

import argparse
import json
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF


@dataclass
class ToCEntry:
    """目录条目"""
    level: int
    title: str
    page_num: int
    # y坐标位置，用于排序
    y_pos: Optional[float] = None

    @staticmethod
    def key(e) -> Tuple[int, float]:
        """排序用的键"""
        return (e.page_num, 0 if e.y_pos is None else e.y_pos)

    def to_fitz_entry(self) -> List:
        """转换为PyMuPDF格式的目录条目"""
        if self.y_pos is not None:
            return [self.level, self.title, self.page_num, {"kind": 1, "to": fitz.Point(0, self.y_pos)}]
        return [self.level, self.title, self.page_num]


class ToCFilter:
    """目录过滤器，用于从PDF中提取匹配的标题"""
    def __init__(self, filter_dict: Dict[str, Any]):
        self.level = filter_dict.get('level', 1)
        self.font_name = filter_dict.get('font', {}).get('name', '')
        self.font_size = filter_dict.get('font', {}).get('size')
        self.font_size_tolerance = 1e-5
        self.font_color = filter_dict.get('font', {}).get('color')
        self.bbox_left = filter_dict.get('bbox', {}).get('left')
        self.bbox_top = filter_dict.get('bbox', {}).get('top')
        self.bbox_right = filter_dict.get('bbox', {}).get('right')
        self.bbox_bottom = filter_dict.get('bbox', {}).get('bottom')
        self.bbox_tolerance = 1e-5

    def admits(self, span: Dict[str, Any]) -> bool:
        """检查span是否匹配过滤条件"""
        # 检查字体名称
        if self.font_name and self.font_name not in span.get('font', ''):
            return False

        # 检查字体大小
        if self.font_size is not None:
            span_size = span.get('size')
            if span_size is None or abs(self.font_size - span_size) > self.font_size_tolerance:
                return False

        # 检查字体颜色
        if self.font_color is not None and self.font_color != span.get('color'):
            return False

        # 检查边界框
        if any(x is not None for x in [self.bbox_left, self.bbox_top, self.bbox_right, self.bbox_bottom]):
            bbox = span.get('bbox', [0, 0, 0, 0])

            if self.bbox_left is not None and abs(self.bbox_left - bbox[0]) > self.bbox_tolerance:
                return False
            
            if self.bbox_top is not None and abs(self.bbox_top - bbox[1]) > self.bbox_tolerance:
                return False
            
            if self.bbox_right is not None and abs(self.bbox_right - bbox[2]) > self.bbox_tolerance:
                return False
            
            if self.bbox_bottom is not None and abs(self.bbox_bottom - bbox[3]) > self.bbox_tolerance:
                return False
        
        return True


def extract_toc_from_pdf(doc: fitz.Document, recipe: List[Dict[str, Any]]) -> List[ToCEntry]:
    """
    使用配方从PDF文档中提取目录
    
    参数:
      doc: PyMuPDF文档对象
      recipe: 目录配方（JSON格式加载的数据）
    
    返回:
      目录条目列表
    """
    result = []
    
    # 创建过滤器列表
    filters = [ToCFilter(entry) for entry in recipe]
    
    for page_num, page in enumerate(doc, 1):
        page_dict = page.get_text("dict")
        
        for blk in page_dict.get('blocks', []):
            if blk.get('type') != 0:  # 不是文本块
                continue
            
            y_pos = blk.get('bbox', [0, 0, 0, 0])[1]
            
            for line in blk.get('lines', []):
                for span in line.get('spans', []):
                    for fltr in filters:
                        if fltr.admits(span):
                            title = span.get('text', '').strip()
                            if title:
                                result.append(ToCEntry(
                                    level=fltr.level,
                                    title=title,
                                    page_num=page_num,
                                    y_pos=y_pos
                                ))
    
    # 按页码和位置排序
    result.sort(key=ToCEntry.key)
    return result


def write_toc_to_pdf(doc: fitz.Document, toc_entries: List[ToCEntry]) -> None:
    """
    将目录写入PDF文档
    
    参数:
      doc: PyMuPDF文档对象
      toc_entries: 目录条目列表
    """
    fitz_toc = [entry.to_fitz_entry() for entry in toc_entries]
    doc.set_toc(fitz_toc)


def generate_toc_from_json(json_data: List[Dict[str, Any]], doc: fitz.Document) -> List[ToCEntry]:
    """
    直接从JSON数据生成目录，无需重新扫描PDF
    
    参数:
      json_data: JSON格式的目录配方
      doc: PyMuPDF文档对象，用于验证页码范围
    
    返回:
      目录条目列表
    """
    result = []
    
    for entry in json_data:
        if 'text' in entry and entry['text'].strip():
            level = entry.get('level', 1)
            title = entry['text'].strip()
            # 使用提供的页码或根据字体属性搜索页码
            page_num = entry.get('page_num', 1)
            y_pos = entry.get('bbox', {}).get('top')
            
            # 确保页码在有效范围内
            if 1 <= page_num <= len(doc):
                result.append(ToCEntry(
                    level=level,
                    title=title,
                    page_num=page_num,
                    y_pos=y_pos
                ))
    
    # 按页码和位置排序
    result.sort(key=ToCEntry.key)
    return result


def pretty_print_toc(entries: List[ToCEntry]) -> str:
    """
    美观打印目录
    
    参数:
      entries: 目录条目列表
    
    返回:
      格式化的字符串
    """
    return '\n'.join([
        f"{(entry.level - 1) * '    '}{entry.title} ··· {entry.page_num}"
        for entry in entries
    ])


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='解析JSON格式的目录信息并将其添加到PDF文件中',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  使用JSON配方文件从PDF中提取目录:
  $ python apply_toc.py -r recipe.json input.pdf -o output.pdf
  
  直接显示生成的目录，不修改PDF:
  $ python apply_toc.py -r recipe.json input.pdf --print-only
"""
    )
    
    parser.add_argument('input_pdf', help='输入的PDF文件路径')
    parser.add_argument('-r', '--recipe', required=True, help='目录配方JSON文件路径')
    parser.add_argument('-o', '--output', help='输出PDF文件路径')
    parser.add_argument('-p', '--print-only', action='store_true', 
                        help='仅打印目录，不修改PDF文件')
    parser.add_argument('-d', '--direct', action='store_true',
                        help='直接使用JSON数据中的文本作为目录，跳过PDF扫描')
    
    args = parser.parse_args()
    
    try:
        # 加载JSON配方
        with open(args.recipe, 'r', encoding='utf-8') as f:
            recipe = json.load(f)
    except Exception as e:
        print(f"错误: 无法加载JSON配方文件 {args.recipe}", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)
    
    try:
        # 打开PDF文件
        doc = fitz.open(args.input_pdf)
    except Exception as e:
        print(f"错误: 无法打开PDF文件 {args.input_pdf}", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)
    
    # 生成目录
    if args.direct:
        # 直接使用JSON数据中的文本作为目录
        toc_entries = generate_toc_from_json(recipe, doc)
    else:
        # 使用配方从PDF中提取目录
        toc_entries = extract_toc_from_pdf(doc, recipe)
    
    # 如果只需打印目录
    if args.print_only:
        print(pretty_print_toc(toc_entries))
        doc.close()
        return
    
    # 将目录写入PDF
    write_toc_to_pdf(doc, toc_entries)
    
    # 保存修改后的PDF
    if args.output:
        try:
            doc.save(args.output)
            print(f"目录已添加到 {args.output}")
        except Exception as e:
            print(f"错误: 无法保存PDF文件 {args.output}", file=sys.stderr)
            print(e, file=sys.stderr)
            sys.exit(1)
    else:
        # 无输出文件时，将结果写入标准输出
        print("警告: 未指定输出文件，请使用 -o 参数指定输出文件", file=sys.stderr)
    
    doc.close()


if __name__ == "__main__":
    main() 