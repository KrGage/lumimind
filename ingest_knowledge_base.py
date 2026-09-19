"""
知识库文档导入脚本
===================

将 PDF / Word / TXT / Markdown 等文档解析、分块后导入 SimpleVectorStore。

用法:
    python ingest_knowledge_base.py                          # 导入默认的 论文 目录
    python ingest_knowledge_base.py --dir ./my_docs          # 导入指定目录
    python ingest_knowledge_base.py --dir ./papers --reset   # 清空后重新导入
    python ingest_knowledge_base.py --stats                  # 查看知识库统计

依赖的解析库（已在环境中检测）:
    - PyPDF2 / pdfplumber  → PDF
    - python-docx           → Word (.docx)
    - markdown              → Markdown (.md)
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 日志 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")

# ── 配置 ──────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 800          # 每块最大字符数
DEFAULT_CHUNK_OVERLAP = 100       # 块间重叠字符数
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


# ═══════════════════════════════════════════════════════════════════
# 文档解析器
# ═══════════════════════════════════════════════════════════════════

def parse_pdf(filepath: str) -> str:
    """解析 PDF，返回提取的文本。优先用 pdfplumber（对中文支持更好）。"""
    text = ""

    # 方案 A: pdfplumber（更准确的字符合并）
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            logger.debug("pdfplumber extracted %d chars from %s", len(text), os.path.basename(filepath))
            return text
    except Exception as e:
        logger.debug("pdfplumber failed for %s: %s", os.path.basename(filepath), e)

    # 方案 B: PyPDF2（回退）
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        logger.debug("PyPDF2 extracted %d chars from %s", len(text), os.path.basename(filepath))
    except Exception as e:
        logger.error("PyPDF2 also failed for %s: %s", os.path.basename(filepath), e)

    return text


def parse_docx(filepath: str) -> str:
    """解析 Word 文档。"""
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        logger.error("Failed to parse docx %s: %s", os.path.basename(filepath), e)
        return ""


def parse_txt(filepath: str) -> str:
    """解析纯文本文件，自动检测编码。"""
    for encoding in ["utf-8", "gbk", "gb2312", "latin-1"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    logger.error("Failed to decode %s with any encoding", os.path.basename(filepath))
    return ""


def parse_markdown(filepath: str) -> str:
    """解析 Markdown——保留纯文本，移除格式标记。"""
    text = parse_txt(filepath)
    if not text:
        return text
    # 移除代码块
    text = re.sub(r"```[\s\S]*?```", "", text)
    # 移除图片语法
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # 将链接转换为纯文本 [text](url) → text
    text = re.sub(r"\[([^\]]*?)\]\(.*?\)", r"\1", text)
    # 移除标题标记但保留文字
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # 移除加粗/斜体标记
    text = re.sub(r"\*{1,3}([^*]+?)\*{1,3}", r"\1", text)
    # 移除行内代码
    text = re.sub(r"`([^`]+?)`", r"\1", text)
    # 移除水平线
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text


PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_txt,
    ".md": parse_markdown,
}


def parse_document(filepath: str) -> Optional[str]:
    """根据扩展名选择合适的解析器。"""
    ext = Path(filepath).suffix.lower()
    parser = PARSERS.get(ext)
    if not parser:
        logger.warning("Unsupported format: %s", ext)
        return None

    text = parser(filepath)
    if not text or not text.strip():
        logger.warning("No text extracted from %s", os.path.basename(filepath))
        return None

    return text


# ═══════════════════════════════════════════════════════════════════
# 文本分块
# ═══════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    将长文本分割为有重叠的块。

    策略：
    1. 首先按段落（双换行）分割
    2. 如果只有 1 个段落且过长，尝试按单换行再分割
    3. 合并短段落直到接近 chunk_size
    4. 如果单段落过长，按句子分割后再合并
    """
    # 规范换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 按段落分割
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # 如果没有段落分隔，尝试按单换行拆分（常见于 PDF 提取的文本）
    if len(paragraphs) <= 1 and len(text) > chunk_size:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) > 3:
            paragraphs = lines

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 如果当前块加上新段落不超限，追加
        if not current_chunk:
            current_chunk = para
        elif len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk += "\n\n" + para
        else:
            # 当前块已满，保存并开始新块
            if current_chunk:
                chunks.append(current_chunk)

            # 如果单段落仍然超过 chunk_size，需要按句子再拆
            if len(para) > chunk_size:
                sub_chunks = _split_long_paragraph(para, chunk_size, overlap)
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = para

    # 保存最后一个块（如果过长则拆分）
    if current_chunk:
        if len(current_chunk) > chunk_size:
            chunks.extend(_split_long_paragraph(current_chunk, chunk_size, overlap))
        else:
            chunks.append(current_chunk)

    # 在块间添加重叠
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            # 从前一块末尾取 overlap 字符作为前缀
            if len(prev) > overlap:
                prefix = prev[-overlap:]
                # 从词边界截断
                cut = prefix.find(" ") if " " in prefix else overlap
                prefix = prefix[cut:].lstrip() if cut > 0 else prefix
                if prefix:
                    curr = prefix + "\n...\n" + curr
            overlapped.append(curr)
        chunks = overlapped

    return chunks


def _split_long_paragraph(text: str, chunk_size: int, overlap: int) -> List[str]:
    """分割过长的段落：按句子分，再合并到接近 chunk_size。"""
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sent in sentences:
        if not current:
            current = sent
        elif len(current) + len(sent) + 1 <= chunk_size:
            current += sent
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return chunks


# ═══════════════════════════════════════════════════════════════════
# 知识库导入
# ═══════════════════════════════════════════════════════════════════

def import_documents(
    input_dir: str,
    reset: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict:
    """
    扫描目录、解析文档、分块、导入向量存储。

    Returns:
        统计信息字典
    """
    input_path = Path(input_dir).resolve()
    if not input_path.exists():
        logger.error("Directory not found: %s", input_path)
        return {"error": f"Directory not found: {input_path}"}

    # 收集所有支持的文件
    files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(input_path.rglob(f"*{ext}"))

    if not files:
        logger.warning("No supported files found in %s", input_path)
        return {"error": f"No supported files in {input_path}", "files_found": 0}

    logger.info("Found %d supported file(s) in %s", len(files), input_path)

    # 导入 agent_framework
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from agent_framework.rag import get_vector_store
    except ImportError as e:
        logger.error("Cannot import agent_framework.rag: %s", e)
        return {"error": f"Import failed: {e}"}

    store = get_vector_store()

    # 可选：重置
    if reset:
        logger.info("Resetting vector store...")
        count_before = store.count()
        for rid in list(store.ids):
            store.delete(rid)
        logger.info("Deleted %d existing records.", count_before)

    stats = {
        "files_found": len(files),
        "files_processed": 0,
        "files_failed": 0,
        "chunks_total": 0,
        "chunks_added": 0,
        "errors": [],
        "details": [],
    }

    for filepath in sorted(files):
        fname = filepath.name
        rel_path = str(filepath.relative_to(input_path.parent)
                       if input_path.parent in filepath.parents
                       else filepath.name)

        logger.info("Processing: %s", rel_path)
        t0 = time.time()

        # 解析
        text = parse_document(str(filepath))
        if not text:
            stats["files_failed"] += 1
            stats["errors"].append(f"{rel_path}: no text extracted")
            continue

        # 分块
        chunks = chunk_text(text, chunk_size, overlap)
        if not chunks:
            stats["files_failed"] += 1
            stats["errors"].append(f"{rel_path}: empty after chunking")
            continue

        # 导入每个块
        file_stats = {"file": rel_path, "chunks": len(chunks), "chars": len(text)}
        added = 0
        for i, chunk_text_content in enumerate(chunks):
            chunk_id = f"{rel_path}#chunk{i}"
            metadata = {
                "source": rel_path,
                "filename": fname,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk_text_content),
            }
            try:
                store.add(chunk_text_content, metadata, chunk_id)
                added += 1
            except Exception as e:
                logger.warning("Failed to add chunk %d of %s: %s", i, rel_path, e)

        elapsed = time.time() - t0
        file_stats["added"] = added
        stats["chunks_total"] += len(chunks)
        stats["chunks_added"] += added
        stats["files_processed"] += 1
        stats["details"].append(file_stats)

        logger.info("  → %d/%d chunks added (%.1fs)", added, len(chunks), elapsed)

    logger.info(
        "Import complete: %d/%d files processed, %d/%d chunks added.",
        stats["files_processed"], stats["files_found"],
        stats["chunks_added"], stats["chunks_total"],
    )

    if stats["errors"]:
        logger.warning("Errors encountered:")
        for err in stats["errors"]:
            logger.warning("  - %s", err)

    return stats


def show_stats():
    """显示当前知识库统计。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from agent_framework.rag import get_vector_store
    except ImportError as e:
        logger.error("Cannot import agent_framework.rag: %s", e)
        return

    store = get_vector_store()
    total = store.count()

    if total == 0:
        logger.info("Knowledge base is empty.")
        return

    # 按来源文件统计
    sources: Dict[str, int] = {}
    for meta in store.metadatas:
        src = meta.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    logger.info("=" * 60)
    logger.info("Knowledge Base Statistics")
    logger.info("=" * 60)
    logger.info("  Total chunks:   %d", total)
    logger.info("  Unique sources: %d", len(sources))
    logger.info("  Storage file:   %s", store.filename)
    logger.info("-" * 60)
    for src, count in sorted(sources.items()):
        logger.info("  %4d chunks  |  %s", count, src)
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="将文档导入本地知识库（基于 agent_framework.rag）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python ingest_knowledge_base.py                           # 导入默认 论文 目录
  python ingest_knowledge_base.py --dir ../论文             # 指定目录
  python ingest_knowledge_base.py --dir ./docs --reset      # 清空后重新导入
  python ingest_knowledge_base.py --stats                   # 查看统计
        """,
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="文档目录路径（默认: ../论文）",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="导入前清空已有知识库",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"每块最大字符数（默认: {DEFAULT_CHUNK_SIZE}）",
    )
    parser.add_argument(
        "--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
        help=f"块间重叠字符数（默认: {DEFAULT_CHUNK_OVERLAP}）",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="仅显示当前知识库统计",
    )

    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    # 默认目录：脚本所在目录同级的 论文 文件夹
    if args.dir is None:
        script_dir = Path(__file__).resolve().parent
        default_dir = script_dir.parent / "论文"
        if default_dir.exists():
            input_dir = str(default_dir)
        else:
            logger.error("Default '论文' directory not found at %s. Use --dir to specify.", default_dir)
            sys.exit(1)
    else:
        input_dir = args.dir

    logger.info("Input directory: %s", input_dir)
    logger.info("Chunk size: %d, Overlap: %d", args.chunk_size, args.overlap)

    stats = import_documents(input_dir, reset=args.reset,
                             chunk_size=args.chunk_size, overlap=args.overlap)

    if "error" in stats:
        logger.error("Import failed: %s", stats["error"])
        sys.exit(1)

    # 导入后显示统计
    show_stats()


if __name__ == "__main__":
    main()
