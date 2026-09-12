"""
CPU-side PDF processing: full text extraction, pre-flight table detection,
YOLO visual detection, SLANet structure recognition, and cell text extraction.

Architecture (7-step MinerU-Lite pipeline):
  1. PyMuPDF heuristic pre-screen (page_has_table)
  2. YOLO visual coarse detection (doclayout_yolo_ft.pt)
  3. PyMuPDF magnetic alignment
  4. PP-LCNet table/non-table classifier (gatekeeper)
  5. SLANet cell topology parsing
  6. Coordinate reverse mapping (crop-local → image → PDF)
  7. PyMuPDF native text clip extraction
"""

from __future__ import annotations

import io as _io
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
import numpy as np
from PIL import Image as _Image

from assay.completion import complete_table
from assay.models import (
    TableModel,
    GPU_BATCH_SIZE,
)
from assay.output import (
    CellOutput,
    DocumentOutput,
    PageOutput,
    TableOutput,
    TextOutput,
)

_log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

PAGE_ZOOM = 2.0            # ~150 DPI rendering

# Pre-flight: dual-signal heuristic to skip non-table pages
MIN_DIGITS_FOR_TABLE = 40     # Chinese financial tables often mix text + numbers,
                               # resulting in lower digit counts than pure numeric tables.
MIN_DIGIT_DENSITY = 0.04      # same rationale — text-heavy table captions dilute density
MIN_DRAWING_STROKES = 30
MIN_DRAWING_FILLS = 50
MIN_DIGITS_SIGNAL2 = 30
MIN_TABLE_GRID_WIDTH = 200.0  # a drawn ruling grid at least this wide marks a
                              # page as table-bearing regardless of digit count

# PP-LCNet gatekeeper threshold
# PP-LCNet was trained on general (English) tables and systematically
# scores Chinese financial tables at ~0.29, while scoring clear non-tables
# even lower. Lowering the threshold recovers genuine tables while still
# filtering obvious false positives.
GATEKEEPER_THRESHOLD = 0.25  # lowered from 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight check
# ──────────────────────────────────────────────────────────────────────────────

def page_has_table(page: fitz.Page) -> bool:
    """
    Fast heuristic to determine if a page likely contains tables.

    Two independent signals (OR relationship):
      1. High digit density: >= 40 digits at >= 4% → borderless numeric tables
      2. Ruled grid: >= 30 strokes + >= 50 fills + >= 100 digits → ruled tables
    """
    text = page.get_text("text")
    total_chars = len(text)
    if total_chars == 0:
        return False

    digit_count = sum(1 for c in text if c.isdigit())
    density = digit_count / total_chars

    if digit_count >= MIN_DIGITS_FOR_TABLE and density >= MIN_DIGIT_DENSITY:
        return True

    if digit_count >= 100:
        drawings = page.get_drawings()
        strokes = sum(1 for d in drawings if d.get('type') == 's')
        fills = sum(1 for d in drawings if d.get('type') == 'f')
        if strokes >= MIN_DRAWING_STROKES and fills >= MIN_DRAWING_FILLS:
            return True

    # 3. Drawn ruling grid: a bordered table is a table no matter what text it
    #    holds.  Catches all-text tables (e.g. "主要资产 / 报告期内无重大变化")
    #    that carry no digits and would otherwise be skipped by the numeric
    #    heuristics above.  detect_ruling_rects already filters to real grids
    #    (>=120pt wide, >=18pt tall); require table width to skip small callouts.
    from assay.rulings import detect_ruling_rects
    rrects, _, _ = detect_ruling_rects(page)
    if any((r[2] - r[0]) >= MIN_TABLE_GRID_WIDTH for r in rrects):
        return True

    # 4. Borderless all-text table (related-party register, entity list):
    #    no digits, no rulings — only the column-gutter geometry gives it away.
    from assay.splitting import looks_text_table
    if looks_text_table(page, tuple(page.rect)):
        return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# Full page text extraction
# ──────────────────────────────────────────────────────────────────────────────


def _is_page_header(text: str) -> bool:
    """Detect page header/footer lines in Chinese financial reports."""
    import re
    if len(text) > 120:
        return False
    if not re.search(r'第\d+页\s*$', text):
        return False
    if text.count(' ') < 5 and '\t' not in text:
        return False
    return True


def _detect_heading_label(text: str, max_font: float,
                          is_bold: bool = False,
                          left_aligned: bool = True) -> str:
    """
    Classify text as heading based on structural patterns and font size.

    The paren-number rules carry a LENGTH CAP: genuine subsection
    headings are short phrases （"(1) 按国际财务报告会计准则编制的
    主要财务数据" = 22 字）, while foot-note ITEMS sharing the same
    "(2) xxx" prefix are full sentences （"(2) 香港中央结算（代理人）
    有限公司为香港交易及结算所有限公司之全资附属公司…" = 45+ 字）—
    without the cap every 注 item rendered as a ### heading.
    """
    import re
    text = text.strip()
    text_len = len(text)

    if re.match(r'^第[一二三四五六七八九十百\d]+节', text):
        return "section_heading"
    if re.match(r'^[一二三四五六七八九十]+、', text) and text_len <= 30:
        return "section_heading"
    if re.match(r'^[（(][一二三四五六七八九十\d]+[）)]', text) \
            and text_len <= 30:
        # 精判：真小节标题 = 编号后跟短语、全文本数字组 ≤2、且不以
        # 句号收尾（句子不是标题）。电话号码 "(852) 2899 2010"（3 个
        # 数字组）、裸标记 "(2)"（编号后无内容）、审计程序句
        # （"…审计程序。"句号收尾）统统不是标题
        content_after = re.sub(r'^[（(][一二三四五六七八九十\d]+[）)]\s*',
                               '', text)
        digit_groups = len(re.findall(r'\d[\d,，.]*\d|\d', text))
        if len(content_after) >= 2 and digit_groups <= 2 \
                and not content_after.endswith('。'):
            return "subsection_heading"
    if is_bold and left_aligned and text_len <= 40 \
            and not text.endswith(('。', '：', ':')) \
            and not re.fullmatch(r'[（(][^）)]{0,6}[）)]|\d[\d.、]*', text):
        # 描边假粗体（fill+stroke 双重渲染）是本报告标题的通用签名：
        # "12、报告期后事项"、"审计师确认"、"意见的基础" 都靠它识别。
        # 句号收尾的粗体行是强调句不是标题
        return "subsection_heading"
    if max_font >= 16 and text_len <= 40:
        return "section_heading"
    return "text"


_WRAP_ENUM_PREFIX = (
    # 小数排除："1.6"/"27.4"/"(5.9)" 是数值不是编号——负向先行否定
    # 数字才能放行 "1. 内容" 同时拦住 "1.6注：…" 幽灵拼接
    re.compile(r'^[（(]?\d{1,3}[.．](?!\d)\s*\S'),          # 1. / 3．内容
    re.compile(r'^[（(]?\d{1,3}[、)）]\s*\S'),               # (3) / 1、
    re.compile(r'^[（(]?[一二三四五六七八九十]{1,3}[)）、.．]\s*\S'),  # 一、/（一）
    re.compile(r'^\(?[ivxIVXa-zA-Z]{1,4}[.、)）]\s*\S'),   # (a) / iv.
    re.compile(r'^[•·●▪◦‣■○–─-]\s*\S'),                   # bullets
    re.compile(r'^[①-⑳㉑-㉟]\s*\S'),                        # circled
)


def _stroke_bold_keys(page: fitz.Page) -> set:
    """
    假粗体签名：本报告的标题字形以 fill+stroke 双重渲染（texttrace 里
    同一段文字出现 type=0 填充与 type=1 描边两条记录），正文字形只有
    填充。返回描边渲染的 (font, size, text) 键集合——dict 提取的 span
    命中即视为粗体行。
    """
    keys: set = set()
    try:
        traces = page.get_texttrace()
    except Exception:
        return keys
    for sp in traces:
        if sp.get("type") in (1, 2):
            try:
                txt = "".join(chr(c[0]) for c in sp.get("chars", []))
            except Exception:
                continue
            if txt.strip():
                keys.add((sp.get("font", ""), round(sp.get("size", 0), 1),
                          txt))
    return keys


def _span_is_bold(s: dict, stroke_keys: set) -> bool:
    t = s.get("text", "")
    if not t.strip():
        return False
    if s.get("flags", 0) & 16 or "bold" in s.get("font", "").lower():
        return True
    font, size = s.get("font", ""), round(s.get("size", 0), 1)
    if (font, size, t) in stroke_keys:
        return True
    # 粒度差异：描边跨度与 dict span 的切分不一致（"、报告期后事项"
    # vs 描边 "报告期后事项"），按包含关系匹配——但被包含方须占
    # span 文字的六成以上，否则正文句子包含标题措辞（"深化新能源
    # 上网电价市场化改革…"含页内粗体标题全文）会整行误判粗体
    for f, sz, tt in stroke_keys:
        if sz == size and len(tt) >= 2 \
                and (len(tt) >= 0.6 * len(t) and tt in t
                     or len(t) >= 0.6 * len(tt) and t in tt):
            return True
    return False


def _merge_siblin_rows(lines: List[dict], em: float) -> List[dict]:
    """
    同一视觉行上的相邻 line 对象按 x 拼成一条逻辑行。

    PyMuPDF 把编号列（"1" + "公司简介"）、混合字体、项目符号（"•" +
    正文）、"标签：值" 两栏、行内引用括号（"签署《…协议》" + "（"2024
    年衍生品框架协议"）。根据2024"）拆成独立 line 对象——y 中心相同
    而 x 相邻。不拼接则阅读顺序被打乱（引用括号插进句中）、编号被当
    成孤立行丢弃。左侧为"前缀样"片段（短编号/项目符号/冒号结尾标签）、
    或该行恰好只有两个片段时拼接（间隙 -14~60pt，容许轻度重叠）；
    三个以上片段共享一行是表格行碎片，不拼。
    """
    if not lines:
        return lines
    lines.sort(key=lambda l: ((l["y0"] + l["y1"]) / 2.0, l["x0"]))
    counts: List[int] = []
    for l in lines:
        cy = (l["y0"] + l["y1"]) / 2.0
        counts.append(sum(1 for m in lines
                          if abs((m["y0"] + m["y1"]) / 2.0 - cy) <= 2.5))
    out: List[dict] = []
    for l, row_n in zip(lines, counts):
        if out:
            p = out[-1]
            pcy = (p["y0"] + p["y1"]) / 2.0
            cy = (l["y0"] + l["y1"]) / 2.0
            gap = l["x0"] - p["x1"]
            ptxt = p["text"].rstrip()
            prefix_like = len(ptxt) <= 4 or ptxt.endswith(("：", ":"))
            if (prefix_like or row_n == 2) \
                    and abs(cy - pcy) <= 4.0 and -14.0 <= gap <= 100.0:
                joiner = " " if gap > 1.0 else ""
                p["text"] = ptxt + joiner + l["text"].lstrip()
                p["x1"] = max(p["x1"], l["x1"])
                p["y0"] = min(p["y0"], l["y0"])
                p["y1"] = max(p["y1"], l["y1"])
                if l["bold"] and len(l["text"]) >= len(ptxt):
                    p["bold"] = True
                p["size"] = max(p["size"], l["size"])
                continue
        out.append(l)
    return out


def extract_page_text(page: fitz.Page, page_idx: int) -> List[TextOutput]:
    """
    Extract page text with PARAGRAPH REFLOW.

    Annual-report PDFs carry no paragraph structure — every visual line
    arrives as its own block, so naive extraction splits sentences at
    every line end.  Chinese reports use the 首行缩进 convention: each
    paragraph's first line is indented ~2em from the body margin while
    continuation lines sit flush left.  That indent is the paragraph
    boundary signature:

      indent line (x0 ≈ body + 1.2~ em)   → new paragraph
      flush line   (x0 ≈ body margin)     → continuation, merged

    Headings, right-aligned annotations and table fragments match
    neither cluster and keep their own blocks (table-internal text is
    filtered at export time by _is_table_text).
    """
    from collections import Counter

    lines = []
    max_font_page = 0.0
    page_h = page.rect.height
    stroke_bold = _stroke_bold_keys(page)
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            total = bold_len = 0
            sizes: List[float] = []
            for s in spans:
                t = s.get("text", "")
                if not t or not t.strip():
                    # 行尾空白 span 可能带更大的字号（仿宋空格 15pt），
                    # 计入会把 12pt 正文行标成 15pt 而拒绝续行合并
                    continue
                total += len(t)
                sz = s.get("size", 0)
                sizes.append(sz)
                if _span_is_bold(s, stroke_bold):
                    bold_len += len(t)
            txt = "".join(s.get("text", "") for s in spans).strip()
            if not txt:
                continue
            x0, y0, x1, y1 = line["bbox"]
            size = max(sizes) if sizes else 0.0
            max_font_page = max(max_font_page, size)
            lines.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                          "text": txt, "size": size,
                          "bold": total > 0 and bold_len >= 0.5 * total})
    if not lines:
        return []

    # drop page-number furniture: dash-wrapped ("- 116 -") anywhere, bare
    # numbers only in the bottom strip — 附注节标题编号（"1 公司简介" 的
    # "1"）是独立 line 对象且位于正文区，绝不能当页码丢弃
    def _is_pageno(l: dict) -> bool:
        t = l["text"].strip()
        if re.match(r'^[-·•—–]\s*\d{1,3}\s*[-—]\s*$', t):
            return True
        if re.fullmatch(r'[-·•—–\s]*\d{1,3}[-—]?', t) \
                and l["y0"] > page_h - 72:
            return True
        return False

    lines = [l for l in lines if not _is_pageno(l)]
    lines = _merge_siblin_rows(lines, 12.0)

    # em width ≈ median per-character advance (for the indent-shift test)
    advs = sorted((l["x1"] - l["x0"]) / len(l["text"]) for l in lines
                  if len(l["text"]) >= 4)
    em = advs[len(advs) // 2] if advs else 12.0

    # walk top-down with the RELATIVE-SHIFT paragraph rule: a line is a
    # NEW paragraph iff it is indented ~+2em RELATIVE TO THE PREVIOUS
    # LINE (段首缩进签名); flush or left-shifted lines are continuations.
    # 参与约束：只有以通栏行（≥55% 页宽）开头的块能吸收续行——表格
    # 行碎片是窄行/共享行 y，永不参与合并；块首非通栏（表格碎片、
    # 短标签）也永不吸收。
    from collections import Counter
    ybucket = Counter(round(l["y0"] / 2.0) for l in lines)
    page_w = page.rect.width

    paragraphs: List[List[dict]] = []
    cur: List[dict] = []
    prev_x0 = None
    for l in lines:
        row_frag = ybucket[round(l["y0"] / 2.0)] >= 2
        shift = (l["x0"] - prev_x0) if prev_x0 is not None else None
        # 不设上限：右对齐落款（"承董事会命"）x0 前移几十 em，也是
        # 换段——旧上限会把落款吸进正文段尾
        new_para = shift is not None and shift >= 1.2 * em
        cur_text = "".join(x["text"] for x in cur).strip()
        cur_size = cur[-1]["size"] if cur else 0.0
        cur_starts_wide = cur and (cur[0]["x1"] - cur[0]["x0"]) >= 0.55 * page_w
        # 裸编号标记（审计报告 "(2)" 单独一行）：孤立短标记行吸收其
        # 内容段，标记与内容不再分离。粗体块（标题）不吸收；括号内是
        # 纯数字的（"(5.9)" 表末格数值泄漏）不是标记
        bare_marker = cur and not cur[-1]["bold"] and len(cur_text) <= 6 \
            and not re.fullmatch(r'[（(]\s*[\d.,，%+\-—\s]+\s*[）)]',
                                 cur_text) and (
            cur_text.startswith("(") or cur_text.startswith("（") or
            any(rx.match(cur_text) for rx in _WRAP_ENUM_PREFIX))
        # merge guards: size change (heading vs body), introductory
        # colon（"各位股东："）, dot-leader lines（目录条目）, boldness
        # flip（标题行绝不并入正文、正文绝不吞掉标题行）——但折行的
        # 长粗体标题（"（2）除中国石油集团外…（不包括香港…"）的续行
        # 仍须并入：块尾非句末标点且已有内容 >12 字时放行
        same_style = cur and abs(cur_size - l["size"]) <= 1.0
        intro_colon = cur_text.endswith("：") or cur_text.endswith(":")
        dot_leader = "...." in l["text"] or "····" in l["text"]
        bold_flip = cur and (cur[-1]["bold"] != l["bold"]) \
            and not (cur[-1]["bold"] and len(cur_text) > 12
                     and cur_text.rstrip()[-1:] not in "。；！？：")
        # 目录条目收尾（"……·····99"）后不再吸收——下一行是新条目
        toc_end = re.search(r'[·．.]{3,}\s*[-—]?\s*\d{1,3}$', cur_text)
        # "标签：值" 面板行（"公司注册中文名称：…"）各自成行：行内
        # 前段带冒号的行不再并入同样带冒号的上一行
        label_colon = re.match(r'^[^：:，。；\s]{1,20}[：:]', l["text"]) \
            and ("：" in cur_text or ":" in cur_text) \
            and not intro_colon
        # 枚举断段：上一行以句末标点收尾、本行以编号/项目符号开头
        # （"……(2) 香港中央结算…" 注释条目各自成段）
        prev_final = cur and cur[-1]["text"].rstrip()[-1:] in "。；！？!?"
        enum_break = (prev_final and not new_para and shift is not None
                      and abs(shift) < 1.2 * em
                      and any(rx.match(l["text"])
                              for rx in _WRAP_ENUM_PREFIX))
        # 短枚举行（"（1）截至…证券发行情况"、列表项单行）：不吸收
        # 后续行——标题/条目各自成段；顶满行宽的枚举行是折行条目的
        # 首行，续行照常并入
        enum_head = cur and len(cur_text) <= 40 \
            and any(rx.match(cur_text) for rx in _WRAP_ENUM_PREFIX) \
            and cur[-1]["x1"] < 0.82 * page_w
        if cur and (cur_starts_wide or bare_marker) and not row_frag \
                and not new_para and same_style and not intro_colon \
                and not dot_leader and not bold_flip and not enum_break \
                and not toc_end and not label_colon and not enum_head:
            cur.append(l)               # continuation line
        else:
            if cur:
                paragraphs.append(cur)
            cur = [l]
        prev_x0 = l["x0"]
    if cur:
        paragraphs.append(cur)

    outputs: List[TextOutput] = []
    for para in paragraphs:
        para.sort(key=lambda l: ((l["y0"] + l["y1"]) / 2.0, l["x0"]))
        full_text = ""
        for l in para:
            # 折行拼接：拉丁词在行尾断开时补空格（"DeGolyer and" +
            # "MacNaughton"），CJK 直接连排
            if full_text and re.search(r'[A-Za-z]$', full_text) \
                    and re.match(r'[A-Za-z(]', l["text"]):
                full_text += " "
            full_text += l["text"]
        full_text = full_text.strip()
        if not full_text:
            continue
        # Wingdings 项目符号的 ToUnicode 缺失：行首 'D'+中文 → '■'
        full_text = re.sub(r'^D(?=[\u4e00-\u9fff])', '■', full_text)
        full_text = re.sub(r'\nD(?=[\u4e00-\u9fff])', '\n■', full_text)
        px0 = min(l["x0"] for l in para)
        py0 = min(l["y0"] for l in para)
        px1 = max(l["x1"] for l in para)
        py1 = max(l["y1"] for l in para)
        # 孤立短数字块只可能是页脚残留（页码过滤器已按位置收窄），正文
        # 区的数字块是 "857" 这类标签值，必须保留
        if len(full_text) < 4 and (full_text.isdigit()
                                   or full_text in ("年报", "年", "页")) \
                and py0 > page_h - 72:
            continue
        if _is_page_header(full_text):
            continue
        max_font = max(l["size"] for l in para)
        label = _detect_heading_label(
            full_text, max_font,
            is_bold=all(l["bold"] for l in para),
            left_aligned=px0 <= 0.45 * page_w)
        outputs.append(TextOutput(
            text=full_text, page_no=page_idx, label=label,
            bbox=(round(px0, 1), round(py0, 1), round(px1, 1), round(py1, 1)),
        ))

    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# SLANet → TableOutput (Steps 5-7)
# ──────────────────────────────────────────────────────────────────────────────


def _expand_bbox(
    bbox: Tuple[float, float, float, float],
    pad_top: float = 15.0,
    pad_bottom: float = 30.0,
    pad_left: float = 10.0,
    pad_right: float = 10.0,
    page_rect: Optional[fitz.Rect] = None,
) -> Tuple[float, float, float, float]:
    """
    Greedy expansion: enlarge the YOLO bbox to ensure no table rows are
    clipped.  The expanded region is fed to SLANet, which determines the
    true table boundaries via its cell grid.  Non-table content (headers,
    footnotes) that gets included will be naturally excluded by the
    post-SLANet grid-collapse step.

    Philosophy: better to include too much (宁放过) than miss rows (勿切错).
    """
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, x0 - pad_left)
    y0 = max(0.0, y0 - pad_top)
    if page_rect is not None:
        x1 = min(page_rect.width, x1 + pad_right)
        y1 = min(page_rect.height, y1 + pad_bottom)
    else:
        x1 = x1 + pad_right
        y1 = y1 + pad_bottom
    return (x0, y0, x1, y1)


def _cell_bbox_to_rect(bbox_8: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert SLANet 8-point bbox (4 corners) to axis-aligned rect (x0,y0,x1,y1)."""
    xs = bbox_8[0::2]  # x0, x2, x4, x6
    ys = bbox_8[1::2]  # y0, y2, y4, y6
    return (float(np.min(xs)), float(np.min(ys)),
            float(np.max(xs)), float(np.max(ys)))


def _table_char_index(page: fitz.Page,
                      table_bbox: Tuple[float, float, float, float]
                      ) -> List[Tuple[float, float, float, float, str, float]]:
    """Visible glyphs inside the table region as
    (x0, y0, x1, y1, char, line_y) — the atoms for best-overlap
    assignment.  clip 提取（get_text clip）会把任何与单元格相交的字形
    都算进去，相邻表头的首几个字（"2024"→"202"）、折行邻格的尾字
    （"民币"、""）因此串格；按字形归属才能根治。"""
    raw = page.get_text(
        "rawdict",
        clip=fitz.Rect(table_bbox[0] - 4.0, table_bbox[1] - 4.0,
                       table_bbox[2] + 4.0, table_bbox[3] + 4.0))
    chars: List[Tuple[float, float, float, float, str, float]] = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            ly0 = line["bbox"][1]
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    c = ch.get("c", "")
                    if not c or not c.strip():
                        continue
                    bx = ch.get("bbox", (0, 0, 0, 0))
                    if bx[2] <= bx[0] or bx[3] <= bx[1]:
                        continue
                    chars.append((bx[0], bx[1], bx[2], bx[3], c, ly0))
    return chars


def _chars_to_text(chars: List[Tuple[float, float, float, float, str, float]]
                   ) -> str:
    """Assigned glyphs → cell text: 同行按 x 排接，行间（y 跨 >3pt）
    插 \\n，同行间隙 >1.2pt 补空格（拉丁词距）。"""
    if not chars:
        return ""
    chars = sorted(chars, key=lambda c: (round(c[5], 1), c[0]))
    parts: List[str] = []
    last_y = None
    last_x1 = None
    for x0, y0, x1, y1, c, ly in chars:
        if last_y is not None:
            if ly - last_y > 3.0:
                parts.append("\n")
            elif x0 - last_x1 > 1.2:
                parts.append(" ")
        parts.append(c)
        last_y = ly
        last_x1 = x1
    return "".join(parts).strip()


def build_table_from_slanet(
    slanet_result: Dict,
    page: fitz.Page,
    page_idx: int,
    scale_x: float,
    scale_y: float,
    table_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[TableOutput]:
    """
    Convert SLANet output into a TableOutput with extracted cell text.

    Steps 5-7: map coords (crop-local → image → PDF), extract text via clip.

    Args:
        slanet_result: dict with 'cell_bboxes' (N,8), 'logic_points' (N,4),
                       'offset' (cx, cy), 'img_size' (w, h)
        page:          PyMuPDF page
        page_idx:      page number (0-based)
        scale_x, scale_y: image-pixel → PDF-point conversion factors
        table_bbox:    overall table bbox in PDF points (YOLO + magnetic align result)

    Returns:
        TableOutput or None if no valid cells.
    """
    cell_bboxes = slanet_result['cell_bboxes']
    logic_points = slanet_result['logic_points']
    crop_x0, crop_y0 = slanet_result['offset']

    n_cells = len(cell_bboxes)

    def _rebuild_fallback(why: str) -> Optional[TableOutput]:
        """
        Geometric rebuild when SLANet cannot produce a usable grid on a
        region the detector and gatekeeper already vouched for.  An
        empty/mismatched SLANet result is a model failure mode, never a
        "not a table" verdict — index-pairing mismatched arrays would
        silently misalign every cell, so the only safe salvage is the
        geometric rebuild.  The downstream single-column veto applies to
        the rebuilt grid exactly as to a SLANet grid.
        """
        if table_bbox is None:
            return None
        from assay.completion import rebuild_grid
        cells, nr, nc, _edges, _rows = rebuild_grid(page, table_bbox)
        if not cells:
            return None
        _log.info(f"p{page_idx + 1} SLANet {why} — geometric rebuild, "
                  f"bbox={table_bbox}")
        return TableOutput(
            page_no=page_idx,
            num_rows=nr,
            num_cols=nc,
            cells=cells,
            table_bbox=table_bbox,
            completed=True,
        )

    if n_cells == 0:
        return _rebuild_fallback("empty output")
    if len(logic_points) != n_cells:
        _log.warning(f"p{page_idx + 1} cell_bboxes({n_cells}) != "
                     f"logic_points({len(logic_points)})")
        return _rebuild_fallback("array mismatch")

    # Determine grid dimensions from logic_points
    max_row = int(logic_points[:, 1].max())  # end_row
    max_col = int(logic_points[:, 3].max())  # end_col
    num_rows = max_row + 1
    num_cols = max_col + 1

    if num_rows == 0 or num_cols == 0:
        return None

    table_cells: List[CellOutput] = []

    # PyMuPDF native text extraction (Step 7) — glyph best-overlap
    # assignment instead of clip: a glyph goes to the cell that contains
    # its centre (smallest containing cell wins on drift overlap), or to
    # the cell it overlaps most when its centre sits between cells
    # （右对齐数值的首个 "(" 中心偏出 1.5pt 仍归本格）.  clip 会把相
    # 邻表头压线的字形整段吸进来（"202"、"民币" 幽灵残字的根源）
    all_chars: List = []
    if table_bbox is not None:
        try:
            all_chars = _table_char_index(page, table_bbox)
        except Exception:
            all_chars = []

    cell_rects: List[Tuple[float, float, float, float]] = []
    cell_meta: List[Tuple[int, int, int, int, tuple]] = []
    for i in range(n_cells):
        bbox_8 = cell_bboxes[i]
        lp = logic_points[i]
        start_row, end_row, start_col, end_col = int(lp[0]), int(lp[1]), int(lp[2]), int(lp[3])
        cx0, cy0, cx1, cy1 = _cell_bbox_to_rect(bbox_8)
        ix0, iy0 = cx0 + crop_x0, cy0 + crop_y0
        ix1, iy1 = cx1 + crop_x0, cy1 + crop_y0
        pdf_bbox = (ix0 * scale_x, iy0 * scale_y,
                    ix1 * scale_x, iy1 * scale_y)
        cell_meta.append((start_row, end_row, start_col, end_col, pdf_bbox))
        cell_rects.append(pdf_bbox)

    assigned: List[List] = [[] for _ in range(n_cells)]
    if all_chars and cell_rects:
        areas = [(r[2] - r[0]) * (r[3] - r[1])
                 if r[2] > r[0] and r[3] > r[1] else 1e18
                 for r in cell_rects]
        for ch in all_chars:
            cx, cy = (ch[0] + ch[2]) / 2.0, (ch[1] + ch[3]) / 2.0
            carea = (ch[2] - ch[0]) * (ch[3] - ch[1])
            contain = [ci for ci, r in enumerate(cell_rects)
                       if r[2] > r[0] and r[3] > r[1]
                       and r[0] - 0.5 <= cx <= r[2] + 0.5
                       and r[1] - 0.5 <= cy <= r[3] + 0.5]
            if contain:
                assigned[min(contain, key=lambda ci: areas[ci])].append(ch)
                continue
            best, best_score = -1, 0.30
            for ci, r in enumerate(cell_rects):
                if r[3] + 2 < cy or r[1] - 2 > cy:
                    continue
                ix0, iy0 = max(ch[0], r[0]), max(ch[1], r[1])
                ix1, iy1 = min(ch[2], r[2]), min(ch[3], r[3])
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                score = (ix1 - ix0) * (iy1 - iy0) / carea
                if score > best_score:
                    best, best_score = ci, score
            if best >= 0:
                assigned[best].append(ch)
                continue
            # 最近格兜底：SLANet 窄格边缘的字形（"…31 日" 的"日"）中心
            # 落在所有格之外 5pt 以内时归最近的格，避免边缘字丢失
            best, best_dist = -1, 5.0
            for ci, r in enumerate(cell_rects):
                if r[2] <= r[0] or r[3] <= r[1]:
                    continue
                dx = max(r[0] - cx, 0.0, cx - r[2])
                dy = max(r[1] - cy, 0.0, cy - r[3])
                d = (dx * dx + dy * dy) ** 0.5
                if d < best_dist:
                    best, best_dist = ci, d
            if best >= 0:
                assigned[best].append(ch)

    for i in range(n_cells):
        start_row, end_row, start_col, end_col, pdf_bbox = cell_meta[i]
        # Validate: PDF bbox must be non-degenerate
        if pdf_bbox[2] <= pdf_bbox[0] or pdf_bbox[3] <= pdf_bbox[1]:
            table_cells.append(CellOutput(
                text="", bbox=(0, 0, 0, 0),
                row=start_row, col=start_col,
                row_span=1, col_span=1,
            ))
            continue
        cell_text = _chars_to_text(assigned[i]) if all_chars else \
            page.get_text("text", clip=fitz.Rect(*pdf_bbox)).strip()

        row_span = end_row - start_row + 1
        col_span = end_col - start_col + 1

        table_cells.append(CellOutput(
            text=cell_text,
            bbox=pdf_bbox,
            row=start_row, col=start_col,
            row_span=row_span,
            col_span=col_span,
        ))

    return TableOutput(
        page_no=page_idx,
        num_rows=num_rows,
        num_cols=num_cols,
        cells=table_cells,
        table_bbox=table_bbox or (0, 0, 0, 0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Extractor (public API)
# ──────────────────────────────────────────────────────────────────────────────


class AssayExtractor:
    """
    Extract tables and text from a PDF file using YOLO + SLANet + PyMuPDF.

    Usage:
        from assay import AssayExtractor
        from assay.models import TableModel

        model = TableModel()
        extractor = AssayExtractor(model)
        doc = extractor.extract("report.pdf")
        doc.save("output/")
    """

    def __init__(self, model: TableModel):
        self.model = model

    def extract(self, pdf_path: str) -> DocumentOutput:
        """
        Extract full content from a PDF: text with headings AND tables.

        Pipeline (7 steps):
          1. Pre-scan: classify pages (table/text) via heuristic
          2. Text extraction: all pages → paragraphs with heading detection
          3. YOLO + magnetic align: table pages → precise bboxes
          4. PP-LCNet gatekeeper: filter false positives
          5. SLANet structure: cell topology (row/col + spanning)
          6. Coordinate mapping: crop-local → image → PDF
          7. PyMuPDF clip: native text extraction
        """
        _log.info(f"Processing: {Path(pdf_path).name}")
        start_time = time.time()

        doc = fitz.open(pdf_path)
        result = DocumentOutput(file_name=Path(pdf_path).stem)

        try:
            total_pages = len(doc)

            # ── Phase 1: classify pages & extract text from ALL pages ────
            from assay.detection import detect_tables_on_page, top_remnant_candidate

            table_page_indices: List[int] = []

            for page_idx in range(total_pages):
                page = doc[page_idx]
                rect = page.rect

                result.pages.append(PageOutput(
                    page_no=page_idx,
                    width=rect.width,
                    height=rect.height,
                ))

                texts = extract_page_text(page, page_idx)
                for t in texts:
                    result.texts.append(t)

                if page_has_table(page) or top_remnant_candidate(page):
                    # The remnant escape hatch admits pages holding ONLY a
                    # spilled continuation row + prose: the four preflight
                    # signals are all blind to them, but the cross-page
                    # remnant test is precise (detection.top_remnant_candidate).
                    table_page_indices.append(page_idx)

            # ── Phase 2: YOLO detect + PP-LCNet gate + SLANet structure ──
            mat = fitz.Matrix(PAGE_ZOOM, PAGE_ZOOM)
            # gpu_results maps page_idx → list of (slanet_result, pdf_bbox) tuples
            gpu_results: Dict[int, List[Tuple[Dict, Tuple[float, float, float, float]]]] = {}
            # Cache pixmap + scale for Phase 3 (avoids redundant re-render)
            _page_cache: Dict[int, Tuple[float, float]] = {}

            # Step A: collect all crops across all pages (PP-LCNet gatekeeping inline)
            all_crops: List[Dict] = []
            all_crop_meta: List[Dict] = []

            for page_idx in table_page_indices:
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=mat)
                scale_x = page.rect.width / pix.width
                scale_y = page.rect.height / pix.height

                # Cache scale for Phase 3 (no need to re-render pix)
                _page_cache[page_idx] = (scale_x, scale_y)

                # Step 2-3: YOLO detect + magnetic align → PDF bboxes
                yolo_tables = detect_tables_on_page(page, pix)
                if not yolo_tables:
                    continue

                # Load full page image once for cropping
                full_img = _Image.open(_io.BytesIO(pix.tobytes("png")))

                for (pdf_x0, pdf_y0, pdf_x1, pdf_y1, conf) in yolo_tables:
                    # The detection bbox is already tight (ruling-line snap or
                    # text shrink).  Pad only a few points so SLANet sees the
                    # full glyphs sitting on the border lines — never the old
                    # greedy expansion that swallowed headers / body text.
                    orig_bbox = (pdf_x0, pdf_y0, pdf_x1, pdf_y1)
                    sub_x0, sub_y0, sub_x1, sub_y1 = _expand_bbox(
                        orig_bbox, pad_top=4.0, pad_bottom=4.0,
                        pad_left=4.0, pad_right=4.0, page_rect=page.rect
                    )

                    # Map PDF bbox → image-pixel coords
                    img_x0 = max(0, int(sub_x0 / scale_x))
                    img_y0 = max(0, int(sub_y0 / scale_y))
                    img_x1 = min(pix.width, int(sub_x1 / scale_x))
                    img_y1 = min(pix.height, int(sub_y1 / scale_y))

                    if img_x1 <= img_x0 or img_y1 <= img_y0:
                        continue

                    # Crop table region
                    crop_img = full_img.crop((img_x0, img_y0, img_x1, img_y1))

                    # Step 4: PP-LCNet gatekeeper
                    if not self.model.is_table(crop_img, threshold=GATEKEEPER_THRESHOLD):
                        continue

                    all_crops.append({
                        'image': crop_img,
                        'offset': (img_x0, img_y0),
                    })
                    all_crop_meta.append({
                        'page_idx': page_idx,
                        # store the tight detection bbox, NOT the padded crop
                        'table_bbox': orig_bbox,
                    })

            # Step B: batch SLANet structure recognition (Step 5)
            for batch_start in range(0, len(all_crops), GPU_BATCH_SIZE):
                batch_end = min(batch_start + GPU_BATCH_SIZE, len(all_crops))
                batch_crops = all_crops[batch_start:batch_end]
                batch_meta = all_crop_meta[batch_start:batch_end]

                slanet_results = self.model.process_crops(batch_crops)
                for j, sres in enumerate(slanet_results):
                    meta = batch_meta[j]
                    gpu_results.setdefault(meta['page_idx'], []).append(
                        (sres, meta['table_bbox'])
                    )

            # ── Phase 3: coordinate mapping + text extraction ────────────
            from assay.rulings import is_bordered_cell_table, is_partitioned_grid
            from assay.splitting import looks_text_table
            for page_idx in table_page_indices:
                page = doc[page_idx]
                slanet_data = gpu_results.get(page_idx, [])
                if not slanet_data:
                    continue

                scale_x, scale_y = _page_cache[page_idx]

                for sres, table_bbox in slanet_data:
                    table_out = build_table_from_slanet(
                        sres, page, page_idx, scale_x, scale_y, table_bbox
                    )
                    if table_out is None:
                        continue
                    # table_bbox is the trusted ruling-line / value-grid box
                    # from detection; it already bounds the table geometrically.
                    # No text-content row trimming here (that used to amputate
                    # legitimate all-text rows like a signatory/address row) and
                    # no grid-collapse reverse.  Each ruled table is its own
                    # detection candidate, so fission is unnecessary.
                    # Reject phantom tables: a single-column "table" is usually a
                    # 1×1 prose fragment, a 2×1 page header or garbled SLANet
                    # output.  Exceptions: a closed drawn grid whose interior
                    # dividers make cells of non-uniform height is a genuine
                    # bordered table (e.g. the "本次现金分红情况" dividend
                    # disclosure), and a borderless text table whose rows share
                    # a column gutter is one too — SLANet reads those as one
                    # column (often one giant 1×1 cell) because no ruling
                    # separates the cells (紫光 p146).  The gutter test brings
                    # its own multi-row requirement, so it needs no SLANet row
                    # count.  Both are decided geometrically — the ML table
                    # classifier actually scores such all-text tables *lower*
                    # than bordered prose callouts, so it cannot be used here.
                    if table_out.num_cols == 1:
                        if not ((table_out.num_rows >= 2 and
                                 (is_bordered_cell_table(page, table_bbox) or
                                  is_partitioned_grid(page, table_bbox))) or
                                looks_text_table(page, table_bbox)):
                            continue
                    if table_out.num_rows > 0:
                        # Content-coverage completion: pure content layer
                        # running AFTER every detection veto, on tables
                        # that survived on their own merits.  It replaces
                        # an under-segmented SLANet grid with a geometric
                        # rebuild (never adds/removes tables — see
                        # assay.completion for its self-checks).
                        new_cells, nnr, nnc, done = complete_table(
                            page, table_bbox, table_out.cells,
                            table_out.num_rows, table_out.num_cols)
                        if done:
                            table_out.cells = new_cells
                            table_out.num_rows = nnr
                            table_out.num_cols = nnc
                            table_out.completed = True
                        result.tables.append(table_out)

        except Exception as e:
            _log.error(f"Error processing {pdf_path}: {e}", exc_info=True)
            result.status = "ERROR"
            result.error = str(e)
        finally:
            doc.close()

        elapsed = time.time() - start_time
        _log.info(f"Done: {Path(pdf_path).name} | "
                  f"{len(result.tables)} tables | "
                  f"{len(result.texts)} text blocks | "
                  f"{elapsed:.1f}s")
        return result
