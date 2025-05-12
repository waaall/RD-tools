"""
    ==========================README===========================
    create date:    20250424
    change date:
    creator:        zhengxu
    function:       提取PDF文件中的目录信息并保存为JSON格式
    details:

    #===========本代码子函数参考下面网址=================
    https://github.com/Krasjet/pdf.tocgen
    =================================================
"""

import argparse
import json
import re
import sys
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF


# =========================================================
# =======               PDF目录提取器              =========
# =========================================================
class PDFTocExtractor:
    def __init__(self, pdf_path: str = None):
        """
        初始化元数据提取器
        参数:
          pdf_path: PDF文件路径, 如果不提供则仅作为配方管理器使用
        """
        self.doc = None
        self.recipe = []
        self.pdf_path = None

        # 常用的目录模式定义为实例变量
        self.patterns = {
            # 一级标题模式：匹配章节号+标题
            1: r'(^第[\s\S]{1,4}[章篇]|^[0-9]+[\.\s]|^[一二三四五六七八九十]+[\.\s]|'
               r'Chapter\s+[0-9]+|^CHAPTER\s+[0-9]+|^[0-9]+[A-Z]+\s|^\d+\s*[^\d\s]+)',
            # 二级标题模式：匹配如1.1这样的节号+标题
            2: r'(^[0-9]+\.[0-9]+[\.\s]|^第[\s\S]{1,4}节|^[0-9]+\-[0-9]+[\.\s]|'
               r'Section\s+[0-9]+\.[0-9]+|^\d+\.\d+\s*[^\d\s\.]+)',
            # 三级标题模式：匹配如1.1.1这样的小节号
            3: r'(^[0-9]+\.[0-9]+\.[0-9]+|^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)'
        }

        if pdf_path:
            try:
                self.doc = fitz.open(pdf_path)
                self.pdf_path = pdf_path
            except Exception as e:
                raise ValueError(f"无法打开PDF文件: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'doc') and self.doc:
            self.doc.close()

    def extract_meta(self, page_num: Optional[int] = None, level: int = 1,
                     ignore_case: bool = False) -> List[Dict[str, Any]]:
        """
        从PDF文档中提取指定文本模式的元数据
        参数:
          page_num: 页码(从1开始), 如果为None则搜索整个文档
          level: 标题级别, 用于选择默认模式
          ignore_case: 是否忽略大小写
        返回:
          包含元数据的字典列表
        """
        if not self.doc:
            raise ValueError("未打开PDF文件")

        # 读取默认模式
        pattern = self.patterns.get(level, self.patterns[1])

        raw_results = []

        if page_num is None:
            pages = self.doc.pages()
        elif 1 <= page_num <= len(self.doc):
            pages = [self.doc[page_num - 1]]
        else:  # 页码超出范围
            return []

        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)

        for p in pages:
            raw_results.extend(self._search_in_page(regex, p))
            
        # 处理和合并相关的文本片段
        processed_results = self.process_extracted_metadata(raw_results, level)
            
        return processed_results

    def _search_in_page(self, regex: re.Pattern, page: fitz.Page) -> List[Dict[str, Any]]:
        """
        在页面中搜索匹配模式的文本并提取元数据

        参数:
          regex: 正则表达式对象
          page: 页面对象
        返回:
          元数据字典列表
        """
        result = []

        page_dict = page.get_text("dict")
        page_num = page.number + 1  # PyMuPDF使用0基索引，但我们需要1基索引

        for blk in page_dict.get('blocks', []):
            for ln in blk.get('lines', []):
                for spn in ln.get('spans', []):
                    text = spn.get('text', "")
                    if regex.search(text):
                        # 添加页码信息
                        spn['page'] = page_num
                        result.append(spn)

        return result

    @staticmethod
    def process_meta_for_json(spn: Dict[str, Any], level: int) -> Dict[str, Any]:
        """
        处理元数据以构建适合JSON格式的对象
        参数:
          spn: span字典
          level: 目录级别
        返回:
          JSON友好的元数据对象
        """
        # 处理字体子集前缀
        font_name = spn['font']
        before, sep, after = font_name.partition('+')
        font_name = after if sep else before

        flags = spn.get('flags', 0)

        metadata = {
            "level": level,
            "text": spn.get('text', ''),
            "font": {
                "name": font_name,
                "size": spn.get('size', 0),
                "color": spn.get('color', 0),
                "flags": {
                    "superscript": bool(flags & 0b00001),
                    "italic": bool(flags & 0b00010),
                    "serif": bool(flags & 0b00100),
                    "monospace": bool(flags & 0b01000),
                    "bold": bool(flags & 0b10000)
                }
            },
            "bbox": {
                "left": spn.get('bbox', [0, 0, 0, 0])[0],
                "top": spn.get('bbox', [0, 0, 0, 0])[1],
                "right": spn.get('bbox', [0, 0, 0, 0])[2],
                "bottom": spn.get('bbox', [0, 0, 0, 0])[3]
            },
            "page": spn.get('page', 0),
            "origin": {
                "x": spn.get('origin', [0, 0])[0],
                "y": spn.get('origin', [0, 0])[1]
            },
            "greedy": True
        }

        return metadata

    def load_from_file(self, file_path: str) -> bool:
        """
        从文件加载配方
        参数:
          file_path: 配方文件路径
        返回:
          是否成功加载
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.recipe = json.load(f)
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def add_entry(self, entry: Dict[str, Any]):
        """
        添加配方条目

        参数:
          entry: 配方条目
        """
        self.recipe.append(entry)

    def save_to_file(self, file_path: str = None) -> bool:
        """
        保存配方到文件
        参数:
          file_path: 配方文件路径
        返回:
          是否成功保存
        """
        if file_path is None:
            # 处理可能的不同格式的扩展名，确保生成正确的JSON文件名
            if self.pdf_path.lower().endswith(('.PDF')):
                file_path = self.pdf_path.replace('.PDF', '.json')
            else:
                file_path = self.pdf_path.replace('.pdf', '.json')

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.recipe, f, ensure_ascii=False, indent=2)
            print(f"解析字幕文件已保存到 {file_path}")
            return True
        except Exception:
            return False

    def process_extracted_metadata(self, entries: List[Dict[str, Any]], 
                                level: int) -> List[Dict[str, Any]]:
        """
        处理提取的元数据，尝试合并相关联的文本片段为完整标题
        
        参数:
          entries: 提取的元数据条目列表
          level: 目录级别
        返回:
          处理后的元数据列表
        """
        if not entries:
            return []
            
        # 按页码和垂直位置排序
        entries.sort(key=lambda x: (x.get('page', 0), x.get('bbox', {}).get('top', 0)))
        
        result = []
        current_group = []
        
        # 根据位置信息将可能属于同一标题的条目分组
        for i, entry in enumerate(entries):
            if not current_group:
                current_group.append(entry)
                continue
                
            prev_entry = current_group[-1]
            
            # 判断当前条目与前一条目是否属于同一行（垂直位置接近）
            same_line = (entry.get('page', 0) == prev_entry.get('page', 0) and
                        abs(entry.get('bbox', {}).get('top', 0) - 
                            prev_entry.get('bbox', {}).get('top', 0)) < 5)
                        
            # 判断水平距离是否合理（同一行内的条目水平距离不应过大）
            reasonable_distance = True
            if same_line:
                distance = entry.get('bbox', {}).get('left', 0) - prev_entry.get('bbox', {}).get('right', 0)
                reasonable_distance = 0 <= distance < 50  # 设定一个合理的阈值
                
            if same_line and reasonable_distance:
                current_group.append(entry)
            else:
                # 处理并添加当前组
                if current_group:
                    merged_entry = self._merge_entries(current_group, level)
                    result.append(merged_entry)
                current_group = [entry]
        
        # 处理最后一组
        if current_group:
            merged_entry = self._merge_entries(current_group, level)
            result.append(merged_entry)
            
        return result
    
    def _merge_entries(self, entries: List[Dict[str, Any]], level: int) -> Dict[str, Any]:
        """
        合并多个条目为一个条目
        
        参数:
          entries: 要合并的条目列表
          level: 目录级别
        返回:
          合并后的条目
        """
        if not entries:
            return {}
            
        if len(entries) == 1:
            # 单个条目，直接调用process_meta_for_json处理
            return self.process_meta_for_json(entries[0], level)
            
        # 合并文本
        texts = [entry.get('text', '').strip() for entry in entries]
        merged_text = ' '.join(texts)
        
        # 以第一个条目为基础，更新其文本和边界框
        base_entry = entries[0].copy()
        base_entry['text'] = merged_text
        
        # 更新边界框，使其包含所有条目
        left = min(entry.get('bbox', (float('inf'), 0, 0, 0))[0] for entry in entries)
        top = min(entry.get('bbox', (0, float('inf'), 0, 0))[1] for entry in entries)
        right = max(entry.get('bbox', (0, 0, 0, 0))[2] for entry in entries)
        bottom = max(entry.get('bbox', (0, 0, 0, 0))[3] for entry in entries)
        
        base_entry['bbox'] = (left, top, right, bottom)
        
        # 使用process_meta_for_json处理
        return self.process_meta_for_json(base_entry, level)


# =====================main(单独执行时使用)=====================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='从PDF文件中提取目录信息并保存为JSON格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  提取PDF文件的一级标题:
  $ python extract_pdf_toc.py input.pdf -l 1 -o recipe.json

  使用特定模式提取二级标题:
  $ python extract_pdf_toc.py input.pdf -p 1 -l 2 -a recipe.json

  提取中文文档的章节标题:
  $ python extract_pdf_toc.py chinese_doc.pdf -l 1 -o recipe.json
"""
    )
    parser.add_argument('input_pdf', help='输入的PDF文件路径')
    parser.add_argument('-p', '--page', type=int, help='要搜索的页码（从1开始）')
    parser.add_argument('-l', '--level', type=int, default=1, help='标题级别（默认为1）')
    parser.add_argument('-i', '--ignore-case', action='store_true', help='忽略大小写')
    parser.add_argument('-o', '--output', help='输出JSON文件的路径')
    parser.add_argument('-a', '--append', help='追加到现有JSON文件')

    args = parser.parse_args()

    try:
        # 提取元数据
        with PDFTocExtractor(args.input_pdf) as extractor:
            entries = extractor.extract_meta(args.page, args.level, args.ignore_case)

            # 如果指定了追加模式, 先加载现有文件
            if args.append and not extractor.load_from_file(args.append):
                print(f"注意: 未找到或无法解析现有文件 {args.append}, 将创建新文件", file=sys.stderr)

            # 处理提取的元数据
            for spn in entries:
                metadata = PDFTocExtractor.process_meta_for_json(spn, args.level)
                extractor.add_entry(metadata)

            # 输出数据
            if args.output:
                if extractor.save_to_file(args.output):
                    print(f"元数据已保存到 {args.output}")
                else:
                    print(f"Error: 无法写入文件 {args.output}", file=sys.stderr)
                    sys.exit(1)
            elif args.append:
                if extractor.save_to_file(args.append):
                    print(f"元数据已追加到 {args.append}")
                else:
                    print(f"Error: 无法写入文件 {args.append}", file=sys.stderr)
                    sys.exit(1)
            else:
                extractor.save_to_file()

    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
