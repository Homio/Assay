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
                          left_aligned: bool = True,
                          _body_em: float = 12.0,
                          _p90: float = 12.0,
                          _short_head: bool = False,
                          _unit_word: bool = False) -> str:
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
    m_num = re.match(r'^(\d{1,3})[.．]\s*(\S[^。；]{1,30})$', text)
    if m_num and not text.endswith(('。', '；', '：', ':')) \
            and not re.search(r'\d[.,，%]', m_num.group(2)):
        # 附注节标题 "34. 盈余公积"：编号+短名词、无句末标点、
        # 标题部分无数字格式（数值句 "1.6 万" 会带）——不依赖粗体
        return "section_heading"
    if _short_head and 2 <= text_len <= 30 \
            and not text.endswith(('。', '；', '：', ':', '，', ',')) \
            and not re.match(r'^[□√■●◆•○□▪]', text) \
            and not re.fullmatch(r'[（(][^）)]{0,6}[）)]', text):
        # 顶格短行块（左缘贴正文边距、单行、无句末标点）——渲染无差异
        # 时的最后结构信号。右对齐落款/单位行/勾选框均已排除
        return "subsection_heading"
    if is_bold and 2 <= text_len <= 40 and not _unit_word \
            and not text.endswith(('。', '：', ':')) \
            and not re.fullmatch(r'[（(][^）)]{0,6}[）)]|\d[\d.、]*', text):
        # 描边假粗体（fill+stroke 双重渲染）是本报告标题的通用签名：
        # "12、报告期后事项"、"审计师确认"、"意见的基础" 都靠它识别。
        # 句号收尾的粗体行是强调句不是标题
        return "subsection_heading"
    if max_font >= 1.25 * _p90 and text_len <= 40:
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
    """标题强调判定：粗体位 / 字体名 / 描边渲染 / 斜体位。斜体是与
    粗体正交的渲染属性——平安附注的小标题（'后续计量''短期薪酬'）
    全部以斜体排版，石油书无斜体标题，互不干扰。"""
    t = s.get("text", "")
    if not t.strip():
        return False
    if s.get("flags", 0) & (16 | 2) or "bold" in s.get("font", "").lower():
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


def _merge_siblin_rows(lines: List[dict], em: float,
                       page_w: float) -> List[dict]:
    """
    同一视觉行上的相邻 line 对象按 x 拼成一条逻辑行。

    PyMuPDF 把编号列（"34." + "盈余公积"）、混合字体、项目符号、
    "标签：值" 两栏、行内引用括号拆成独立 line 对象。同视觉行判定
    用 y 区间重叠过半（行框高度不一会让中心距产生 0.8pt 级别的颠倒，
    按中心排序会把编号排到标题后面拼出倒序），行内按 x0 升序拼接，
    编号列永远回到句首。
    """
    if not lines:
        return lines
    lines.sort(key=lambda l: (l["y0"], l["x0"]))
    n = len(lines)
    sibs: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        li = lines[i]
        for j in range(i + 1, n):
            lj = lines[j]
            if lj["y0"] >= li["y1"]:
                break
            ov = min(li["y1"], lj["y1"]) - max(li["y0"], lj["y0"])
            min_h = max(1.0, min(li["y1"] - li["y0"], lj["y1"] - lj["y0"]))
            if ov > 0.5 * min_h:
                # x 关系判定与 y 排序顺序无关：区间重叠（负值=间隙）
                xov = min(li["x1"], lj["x1"]) - max(li["x0"], lj["x0"])
                if xov >= -0.17 * page_w:
                    sibs[i].append(j)
                    sibs[j].append(i)
    used = [False] * n
    out: List[dict] = []
    for i in range(n):
        if used[i]:
            continue
        group = sorted((j for j in [i] + sibs[i] if not used[j]),
                       key=lambda j: lines[j]["x0"])
        if not group:
            continue
        base = dict(lines[group[0]])
        used[group[0]] = True
        for j in group[1:]:
            m = lines[j]
            used[j] = True
            if m["x0"] >= base["x0"]:
                gap = m["x0"] - base["x1"]
                joiner = " " if gap > 1.0 else ""
                base["text"] = base["text"].rstrip() + joiner + m["text"].lstrip()
            else:
                gap = base["x0"] - m["x1"]
                joiner = " " if gap > 1.0 else ""
                base["text"] = m["text"].rstrip() + joiner + base["text"].lstrip()
                base["x0"] = m["x0"]
            base["x1"] = max(base["x1"], m["x1"])
            base["y0"] = min(base["y0"], m["y0"])
            base["y1"] = max(base["y1"], m["y1"])
            base["size"] = max(base["size"], m["size"])
            base["bold"] = base["bold"] or m["bold"]
        out.append(base)
    return out


def extract_page_text(page: fitz.Page, page_idx: int,
                       header_sigs: Optional[set] = None) -> List[TextOutput]:
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
                and l["y0"] > page_h * 0.915:
            return True
        return False

    # 跨页共识页眉：同一签名在同一顶部 y 位重复于全书 ≥10% 页面 ——
    # 真实表格行不可能如此重复。签名由 calibrate_page_headers 全书
    # 学习（数据驱动，替代按书写的页眉样式匹配）。页眉行常由两个
    # line 对象组成（"四、" + "财务报表附注(续)"），删命中行时其
    # 同视觉行伙伴一并删除，避免残留孤立半截标题
    hdr_idx: set = set()
    for i, l in enumerate(lines):
        if l["y0"] < 90.0 and header_sigs:
            sig = re.sub(r"[\d\s]+", "", l["text"])
            if len(sig) >= 4 and sig in header_sigs:
                hdr_idx.add(i)
    if hdr_idx:
        ys = [(l["y0"], l["y1"]) for l in lines]
        grown: set = set()
        for i in hdr_idx:
            grown.add(i)
            y0, y1 = ys[i]
            for j, (a, b) in enumerate(ys):
                if j == i or j in grown:
                    continue
                if a < y1 + 2 and b > y0 - 2:
                    grown.add(j)
        lines = [l for i, l in enumerate(lines) if i not in grown]
    else:
        lines = [l for l in lines if not _is_pageno(l)]
    lines = [l for l in lines if not _is_pageno(l)]
    page_w = page.rect.width

    # em width ≈ median per-character advance (for the indent-shift test)
    advs = sorted((l["x1"] - l["x0"]) / len(l["text"]) for l in lines
                  if len(l["text"]) >= 4)
    em = advs[len(advs) // 2] if advs else 12.0
    lines = _merge_siblin_rows(lines, em, page_w)
    # 页内字号 90 分位：表格注释密集页的全页中位会被小字拖低，使
    # 8pt 注释超过 1.33×em 而成幽灵标题（平安 '## 注1：…'）。大字号
    # 标题判据改用分布头部——正文主导字号之上的 25% 余量
    size_sorted = sorted(l["size"] for l in lines if l["size"] > 1)
    p90_size = size_sorted[int(0.9 * (len(size_sorted) - 1))] \
        if size_sorted else 12.0

    # walk top-down with the RELATIVE-SHIFT paragraph rule: a line is a
    # NEW paragraph iff it is indented ~+2em RELATIVE TO THE PREVIOUS
    # LINE (段首缩进签名); flush or left-shifted lines are continuations.
    # 参与约束：只有以通栏行（≥55% 页宽）开头的块能吸收续行——表格
    # 行碎片是窄行/共享行 y，永不参与合并；块首非通栏（表格碎片、
    # 短标签）也永不吸收。
    from collections import Counter
    ybucket = Counter(round(l["y0"] / 2.0) for l in lines)

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
        # 编号节标题（平安附注 "34. 盈余公积" 形状：编号+短名词、
        # 无句末标点）独立成块——不依赖粗体的标题结构信号
        _mt = re.match(r'^\d{1,3}[.．]\s*(\S{1,30})$', l["text"])
        numtitle = bool(_mt) and "." not in _mt.group(1) \
            and "．" not in _mt.group(1) \
            and not l["text"].rstrip()[-1:] in "。；！？!?" 
        # 枚举断段：上一行以句末标点收尾、本行以编号/项目符号开头
        # （"……(2) 香港中央结算…" 注释条目各自成段）
        prev_final = cur and cur[-1]["text"].rstrip()[-1:] in "。；！？!?"
        # 悬挂缩进窗口用当前行自身行高（em 中位会被表格小字拖低）
        line_h = max(1.0, l["y1"] - l["y0"])
        enum_break = (prev_final and not new_para and shift is not None
                      and -3.5 * line_h <= shift < 1.2 * em
                      and any(rx.match(l["text"])
                              for rx in _WRAP_ENUM_PREFIX))
        # 空行断段：当前行与前一物理行之间有超过 2.2×行高 的垂直
        # 间隙（空行分段的约定），不是折行续行——阈值用行高，em 会被
        # 表格小字拖低（紫光 '…公司持'/'续经营…' 折行 gap 12pt 被
        # em=6.5 下的 11.7pt 阈值误断）
        gap_break = cur and not row_frag \
            and (l["y0"] - cur[-1]["y1"]) > 2.2 * max(1.0, l["y1"] - l["y0"])
        # 短行断段：前行句号收尾 + 本行顶格短行（宽<0.35 页宽、非枚举
        # 头/纯数字）——平安附注小标题与正文渲染完全同字体同字号、行距
        # 无差异，短行是唯一结构信号
        short_break = (prev_final and not row_frag
                       and (l["x1"] - l["x0"]) < 0.35 * page_w
                       and l["x0"] <= 0.25 * page_w
                       and not any(rx.match(l["text"])
                                   for rx in _WRAP_ENUM_PREFIX)
                       and not re.fullmatch(r'[-—·\s]*\d{1,3}[-—]?', l["text"])
                       and len(l["text"]) <= 20
                       and not l["text"].startswith(("（货币单位", "(货币单位")))
        # 句中断行必合并：前行以非句末标点收尾（且非枚举头）时，本行
        # 是同段折行续行——块首行窄（cur_starts_wide=False）也不能漏并
        # fold 仅当前行接近通栏（>0.4 页宽）时生效——窄短行（'短期薪酬'
        # 等小标题）虽以非句末标点结尾也不是"句中折行"，不能把正文吸进
        prev_open = cur and not prev_final and cur[-1]["text"].rstrip()[-1:] \
            not in "：:）、】」”" \
            and (cur[-1]["x1"] - cur[-1]["x0"]) > 0.4 * page_w
        fold = (cur and prev_open and not row_frag and same_style
                and abs(shift if shift is not None else 0) < 3.5 * line_h
                and not any(rx.match(l["text"])
                            for rx in _WRAP_ENUM_PREFIX))
        # 短枚举行（"（1）截至…证券发行情况"、列表项单行）：不吸收
        # 后续行——标题/条目各自成段；顶满行宽的枚举行是折行条目的
        # 首行，续行照常并入
        enum_head = cur and len(cur_text) <= 40 \
            and any(rx.match(cur_text) for rx in _WRAP_ENUM_PREFIX) \
            and cur[-1]["x1"] < 0.82 * page_w
        if cur and (cur_starts_wide or bare_marker or fold) and not row_frag \
                and not new_para and same_style and not intro_colon \
                and not dot_leader and not bold_flip and not enum_break \
                and not toc_end and not label_colon and not enum_head \
                and not numtitle and not gap_break and not short_break:
            cur.append(l)               # continuation line
        else:
            if cur:
                paragraphs.append(cur)
            cur = [l]
        prev_x0 = l["x0"]
    if cur:
        paragraphs.append(cur)

    def _width(pp) -> float:
        return max(x["x1"] for x in pp) - min(x["x0"] for x in pp)

    outputs: List[TextOutput] = []
    for pi, para in enumerate(paragraphs):
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
        prev_wide = pi > 0 and _width(paragraphs[pi - 1]) >= 0.5 * page_w
        next_wide = pi + 1 < len(paragraphs) \
            and _width(paragraphs[pi + 1]) >= 0.5 * page_w
        _unit_word = bool(re.fullmatch(
            r'[（(]?[\d.,，%\s]*'
            r'(人民币千元|人民币百万元|美元千元|人民币元|万美元'
            r'|人民币|外币|美元|港元|港币|欧元|日元|万元|千元|百万元|亿元|元|股)?'
            r'[\d.,，%\s]*[）)]?(年度|年|月|日)?', full_text)) \
            and len(full_text) <= 12
        label = _detect_heading_label(
            full_text, max_font,
            is_bold=all(l["bold"] for l in para),
            _body_em=em, _p90=p90_size,
            _short_head=(len(para) == 1 and para[0]["x0"] <= 0.20 * page_w
                         and (para[0]["x1"] - para[0]["x0"]) < 0.35 * page_w
                         and (prev_wide or next_wide)),
            _unit_word=_unit_word,
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


def _median_char_size(chars: list) -> float:
    sizes = sorted(c[6] for c in chars if len(c) > 6 and c[6] > 1)
    return sizes[len(sizes) // 2] if sizes else 9.0


def _table_scale(page: fitz.Page,
                 table_bbox: Tuple[float, float, float, float]) -> float:
    """区域文字中位字号——几何阈值的缩放基准（见 completion._table_scale）。"""
    sizes = []
    for block in page.get_text("dict",
                               clip=fitz.Rect(*table_bbox))["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for s in line.get("spans", []):
                t = s.get("text", "")
                if t.strip() and s.get("size", 0) > 1:
                    sizes.append(s["size"])
    if not sizes:
        return 9.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def _table_char_index(page: fitz.Page,
                      table_bbox: Tuple[float, float, float, float]
                      ) -> List[Tuple[float, float, float, float, str, float, float]]:
    """Visible glyphs inside the table region as
    (x0, y0, x1, y1, char, line_y) — the atoms for best-overlap
    assignment.  clip 提取（get_text clip）会把任何与单元格相交的字形
    都算进去，相邻表头的首几个字（"2024"→"202"）、折行邻格的尾字
    （"民币"、""）因此串格；按字形归属才能根治。"""
    scale = _table_scale(page, table_bbox)
    pad = 0.45 * scale
    raw = page.get_text(
        "rawdict",
        clip=fitz.Rect(table_bbox[0] - pad, table_bbox[1] - pad,
                       table_bbox[2] + pad, table_bbox[3] + pad))
    chars: List[Tuple[float, float, float, float, str, float, float]] = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            ly0 = line["bbox"][1]
            for span in line.get("spans", []):
                ssize = span.get("size", 0) or 9.0
                for ch in span.get("chars", []):
                    c = ch.get("c", "")
                    if not c or not c.strip():
                        continue
                    bx = ch.get("bbox", (0, 0, 0, 0))
                    if bx[2] <= bx[0] or bx[3] <= bx[1]:
                        continue
                    chars.append((bx[0], bx[1], bx[2], bx[3], c, ly0, ssize))
    return chars


_NUM_FULL_RE = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
_NUM_CHARS = frozenset("0123456789.,-")


def _glue_numeric_seeps(assigned: List[List], cell_meta: list,
                        med_cs: float) -> None:
    """密排数值列的拆粘归位。

    SLANet 对密排多列表给出的相邻格 bbox 互相重叠（右对齐数值列普遍
    重叠 6-8pt），跨格排版数字串的边缘字形中心落入重叠带时，含格判
    定（最小面积优先）会把它抢进邻格——"2,707,138.34" 拆成
    "2,707,138.3"（本格）+"4"（右格），"1,051,128,788.30" 的首字
    "1" 落进左格。同一排版文本串的字形间隙 ≤0.2×字号，而跨格的
    独立内容至少隔一个空格宽——按 x 邻接性把串格碎片归回原格。
    只移动纯数值碎片，且归位后的完整串必须满足千分位/普通数值格
    式；文本格、混合内容格不受影响。
    """
    if med_cs <= 0:
        return
    glue_gap = 0.2 * med_cs  # 同串字形间隙上限：串内数字对间隙远小于
                             # 此，单空格（≈0.25×字号）即超过——靠它把
                             # 空格分隔的相邻数值切成两串
    sep_gap = 0.5 * med_cs   # 碎片与格内其余内容的最小隔离：须大于单
                             # 空格宽，小于真实列间隔（≥3 空格）

    by_rc: Dict[Tuple[int, int], int] = {}
    for ci, (sr, _er, sc, _ec, _r) in enumerate(cell_meta):
        by_rc.setdefault((sr, sc), ci)

    def _lines(chars: list) -> list:
        lines: list = []
        for ch in sorted(chars, key=lambda c: (c[5], c[0])):
            if lines and abs(lines[-1][0][5] - ch[5]) <= 3.0:
                lines[-1].append(ch)
            else:
                lines.append([ch])
        for ln in lines:
            ln.sort(key=lambda c: c[0])
        return lines

    def _is_frag(run: list) -> bool:
        # 串格碎片体量约束：SLANet 相邻格 bbox 重叠 6-9pt，落入重叠带
        # 被抢走的只能是 1-2 个纯数字字形（含分隔符/更长的碎片属"列
        # 界整体偏移"的另一类缺陷，宿主方向需列对齐证据才能判定，本
        # 函数刻意不碰）
        return 1 <= len(run) <= 2 and all(c[4].isdigit() for c in run)

    def _num_run(line: list, from_right: bool) -> list:
        # 行首/行尾的边缘数值串：数字字符集且 x 邻接（间隙 ≤glue_gap，
        # 同排版串特征）。空格字形不入 assigned，不能靠字符集截断——
        # "0 2,707,138.3" 会被整行当一串，必须按间隙切成 ["0",
        # "2,707,138.3"]
        run: list = []
        seq = list(reversed(line)) if from_right else line
        for ch in seq:
            if ch[4] not in _NUM_CHARS:
                break
            if run:
                gap = (run[-1][0] - ch[2]) if from_right \
                    else (ch[0] - run[-1][2])
                if gap > glue_gap:
                    break
            run.append(ch)
        if from_right:
            run.reverse()
        return run

    def _glue_pair(lchars: list, rchars: list) -> bool:
        for rline in _lines(rchars):
            r_run = _num_run(rline, from_right=False)
            if not r_run:
                continue
            nxt = rline[len(r_run):]
            if nxt and nxt[0][0] - r_run[-1][2] < sep_gap:
                continue  # 与右格自身内容粘连，非串格碎片
            for lline in _lines(lchars):
                if abs(lline[0][5] - rline[0][5]) > 3.0:
                    continue
                l_run = _num_run(lline, from_right=True)
                if not l_run:
                    continue
                prv = lline[: len(lline) - len(l_run)]
                if prv and l_run[0][0] - prv[-1][2] < sep_gap:
                    continue
                if not (-0.5 <= r_run[0][0] - l_run[-1][2] <= glue_gap):
                    continue
                l_txt = "".join(c[4] for c in l_run)
                r_txt = "".join(c[4] for c in r_run)
                if not _NUM_FULL_RE.match(l_txt + r_txt):
                    continue
                if not _NUM_FULL_RE.match(r_txt):
                    # HEAD：右格丢了首段（",051,128,788.30" 的前导逗号
                    # 不可能是独立内容）→ 左格尾串是它的头，归右格
                    if not _is_frag(l_run):
                        continue
                    for ch in l_run:
                        lchars.remove(ch)
                    rchars.extend(l_run)
                else:
                    # TAIL：右格左端碎片是左格数字的尾段，归左格
                    if not _is_frag(r_run):
                        continue
                    for ch in r_run:
                        rchars.remove(ch)
                    lchars.extend(r_run)
                return True
        return False

    for sr in sorted({r for r, _c in by_rc}):
        cols = sorted(c for r, c in by_rc if r == sr)
        for a, b in zip(cols, cols[1:]):
            li, ri = by_rc[(sr, a)], by_rc[(sr, b)]
            if assigned[li] and assigned[ri]:
                _glue_pair(assigned[li], assigned[ri])


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
    for x0, y0, x1, y1, c, ly, _sz in chars:
        if last_y is not None:
            if ly - last_y > 3.0:
                parts.append("\n")
            elif x0 - last_x1 > 1.2:
                parts.append(" ")
        parts.append(c)
        last_y = ly
        last_x1 = x1
    return "".join(parts).strip()


def _running_header_rects(
    page: fitz.Page, header_sigs: Optional[set]
) -> List[Tuple[float, float, float, float]]:
    """C3 页眉防吸入（表格链路）：本书跨页共识页眉行的 bbox。

    extract_page_text 的页眉过滤只作用于文本链路；表格链路
    （_table_char_index 的 pad 外扩、completion 重建的 clip 吸入）
    会把表格 bbox 上缘附近的页眉行拉进单元格（诚志 p63 r0c1 粘入
    "诚志股份有限公司 2025 年年度报告全文"）。判定与 extract_page_text
    完全同源：签名 = 去数字去空白后的整行文本，命中全书共识
    （calibrate_page_headers 学习，≥HDR_MIN_PAGES 页重复），且行
    y0 < 90（顶部条带）。命中行的同视觉行伙伴（y 重叠 ±2pt，页眉
    常由两个 line 对象组成）一并计入。返回外扩 1pt 的 rect 列表，
    供表格字形归属阶段整行排除。
    """
    if not header_sigs:
        return []
    lines: List[Tuple[float, float, float, float, str]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            txt = "".join(s.get("text", "")
                          for s in line.get("spans", [])).strip()
            if not txt:
                continue
            x0, y0, x1, y1 = line["bbox"]
            lines.append((x0, y0, x1, y1, txt))
    if not lines:
        return []
    hdr_idx = set()
    for i, (_x0, y0, _x1, _y1, txt) in enumerate(lines):
        if y0 < 90.0:
            sig = re.sub(r"[\d\s]+", "", txt)
            if len(sig) >= 4 and sig in header_sigs:
                hdr_idx.add(i)
    if not hdr_idx:
        return []
    grown: set = set()
    for i in hdr_idx:
        grown.add(i)
        y0, y1 = lines[i][1], lines[i][3]
        for j, (a, b) in enumerate((l[1], l[3]) for l in lines):
            if j == i or j in grown:
                continue
            if a < y1 + 2 and b > y0 - 2:
                grown.add(j)
    rects: List[Tuple[float, float, float, float]] = []
    for i in grown:
        x0, y0, x1, y1 = lines[i][:4]
        # 共识页眉永远在顶部条带；同视觉行伙伴不可能深入正文区
        if y0 < 100.0:
            rects.append((x0 - 1.0, y0 - 1.0, x1 + 1.0, y1 + 1.0))
    return rects


def _strip_header_lines(
    cells: List[CellOutput], page: fitz.Page,
    header_rects: List[Tuple[float, float, float, float]]
) -> List[CellOutput]:
    """C3 页眉防吸入（重建链路）：从最终单元格文本中删除页眉行。

    build_table_from_slanet 的字形过滤覆盖 SLANet 网格路径；
    complete_table 的几何重建直接重读页面（_line_fragments 的
    pad 外扩），页眉行仍会被重建进首行单元格。这里按"归一化后
    整行相等 → 删行；归一化后包含 → 正则摘除"两级处理单元格文
    本，与全书共识页眉（绝不可能是表格数据）严格匹配。
    """
    hdr_norms = []
    for hx0, hy0, hx1, hy1 in header_rects:
        try:
            txt = page.get_text("text", clip=fitz.Rect(hx0, hy0,
                                                       hx1, hy1))
        except Exception:
            continue
        norm = re.sub(r"\s+", "", txt)
        if len(norm) >= 4:
            hdr_norms.append(norm)
    if not hdr_norms:
        return cells
    # 几何门控：页眉物理上只能被吸入表的顶部条带（页眉 rect 的
    # y1 之上）。平安银行页眉签名为公司名等通用文本，与表内数据行
    # 撞文本——纯文本匹配会误删表中部合法行（p5/p106/p218 等 23
    # 处 trec 回归的根因）。只对 bbox 顶部落在页眉带 +2pt 容差内
    # 的单元格做摘除，其余单元格原样保留。
    hdr_y_limit = max(hy1 for _hx0, hy0, _hx1, hy1 in header_rects) + 2.0
    hdr_set = set(hdr_norms)
    pats = [(n, re.compile(r"\s*".join(re.escape(c) for c in n)))
            for n in hdr_norms]
    for c in cells:
        if not c.text:
            continue
        if c.bbox[1] > hdr_y_limit:
            continue
        out_lines = []
        for ln in c.text.split("\n"):
            norm = re.sub(r"\s+", "", ln)
            if norm in hdr_set:
                continue
            stripped = ln
            for _n, pat in pats:
                if _n in norm and len(norm) > len(_n):
                    stripped = pat.sub("", stripped).strip()
            if stripped:
                out_lines.append(stripped)
        new_text = "\n".join(out_lines)
        if new_text != c.text:
            c.text = new_text
    return cells


def _enforce_line_atomicity(
    assigned: List[List], all_chars: List, cell_rects: List,
    cell_meta: List, med_cs: float
) -> List[List]:
    """C3 行原子化：同一视觉文本行（rawdict line，同一 ly0）的字形
    绝不跨网格行拆分；行内跨格拆分只允许发生在真实字形间隙
    （≥0.4×中位字号）且两侧字形都落在各自格的 x 范围内。

    乱跨度网格（恒逸 p105 SLANet 6x15）把折行正文按字形劈进同行
    多格（"上述关联交易有利/按类别…/定的电价为定价" 三段同格交
    错）、竖排名单尾字劈进邻列（诚志 p63 "清/诚/融" 落 c0）。违反
    上述两条的行/行内簇整体归"包含簇中心的格"（无含格时归多数
    格，并列取最左），与既有三级归属（containment→重叠→最近）互
    补：三级归属解决"格该归谁"，本函数解决"行不许被格边切碎"。
    """
    if not all_chars or not cell_rects:
        return assigned
    ch_cell: dict = {}
    for ci, chs in enumerate(assigned):
        for ch in chs:
            ch_cell[id(ch)] = ci
    gap_thr = 0.4 * med_cs
    by_line: dict = {}
    for ch in all_chars:
        by_line.setdefault(ch[5], []).append(ch)
    for _ly, chs in by_line.items():
        chs.sort(key=lambda c: c[0])
        clusters: List[List] = [[chs[0]]]
        for ch in chs[1:]:
            if ch[0] - clusters[-1][-1][2] >= gap_thr:
                clusters.append([ch])
            else:
                clusters[-1].append(ch)
        for cl in clusters:
            holders: dict = {}
            for ch in cl:
                ci = ch_cell.get(id(ch))
                if ci is not None:
                    holders.setdefault(ci, []).append(ch)
            if len(holders) < 2:
                continue
            # 纯数字簇跳过：密排数值列的跨格拆粘由 _glue_numeric_seeps
            # 专责（"2,707,138.3|4"），此处重聚会回退其归位结果
            if sum(1 for ch in cl if ch[4] in _NUM_CHARS) \
                    >= 0.6 * len(cl):
                continue
            rows = {cell_meta[ci][0] for ci in holders}
            cross_row = len(rows) >= 2
            # 同行拆分仍可能是真分列（"周 松" 双名列、混排日期）：
            # 要求每个持有格的字形都落在该格 x 范围（±1pt）内——经
            # 包络/兜底路径漂进远格的字形（交错签名的成因）不满足；
            # 且持有格 x 范围两两不重叠（真列间有 gutter，乱跨度网
            # 格的同行格互相压叠 x 领地，如恒逸 p105 r2c4 与 r2c11
            # 压叠 75pt，把同一折行行劈成两段）
            outside = any(
                ch[0] < cell_rects[ci][0] - 1.0
                or ch[2] > cell_rects[ci][2] + 1.0
                for ci, hchs in holders.items() for ch in hchs)
            if not cross_row and not outside:
                xr = sorted((cell_rects[ci][0], cell_rects[ci][2])
                            for ci in holders)
                overlap = any(
                    xr[i + 1][0] < xr[i][1] - 0.8 * med_cs
                    for i in range(len(xr) - 1))
                if not overlap:
                    continue
            cx = (cl[0][0] + cl[-1][2]) / 2.0
            cy = (min(c[1] for c in cl) + max(c[3] for c in cl)) / 2.0
            contain = [ci for ci in holders
                       if cell_rects[ci][0] - 0.5 <= cx
                       <= cell_rects[ci][2] + 0.5
                       and cell_rects[ci][1] - 0.5 <= cy
                       <= cell_rects[ci][3] + 0.5]
            if contain:
                target = min(
                    contain,
                    key=lambda ci: ((cell_rects[ci][2] - cell_rects[ci][0])
                                    * (cell_rects[ci][3] - cell_rects[ci][1])))
            else:
                target = max(
                    holders.items(),
                    key=lambda kv: (len(kv[1]), -kv[0]))[0]
            moved = [ch for hchs in holders.values() for ch in hchs]
            for ci in holders:
                drop = {id(ch) for ch in holders[ci]}
                assigned[ci] = [ch for ch in assigned[ci]
                                 if id(ch) not in drop]
            assigned[target].extend(moved)
    return assigned


def _region_xstart_columns(
    page: fitz.Page, table_bbox: Tuple[float, float, float, float],
    header_rects: Optional[
        List[Tuple[float, float, float, float]]] = None
) -> int:
    """C3 重建坍缩守卫用：区域文本的持久片段 x 起点簇数。

    竖排名单、日期、状态列的字形常与相邻列合并在同一 rawdict 行
    对象里（诚志 p63 "天府清2016年" 一行含公司列+日期列），行级
    x0 看不见这些列；按行内真实字形间隙（≥0.4×中位字号）切成片
    段后取片段 x0，同列折行/竖排块的片段 x0 一致（±0.6×中位字
    号），聚类统计"≥2 个片段成员"的簇数——即区域真实列起点数。
    页眉 rect 内的字形先行排除（表格上缘 pad 吸入的页眉会形成幻
    影簇）。
    """
    try:
        chars = _table_char_index(page, table_bbox)
    except Exception:
        return 0
    if header_rects:
        def _in_hdr(ch) -> bool:
            cx, cy = (ch[0] + ch[2]) / 2.0, (ch[1] + ch[3]) / 2.0
            return any(hx0 <= cx <= hx1 and hy0 <= cy <= hy1
                       for hx0, hy0, hx1, hy1 in header_rects)
        chars = [ch for ch in chars if not _in_hdr(ch)]
    if not chars:
        return 0
    med = _median_char_size(chars)
    by_line: dict = {}
    for ch in chars:
        by_line.setdefault(ch[5], []).append(ch)
    frag_x0s: List[float] = []
    gap_thr = 0.4 * med
    for chs in by_line.values():
        chs.sort(key=lambda c: c[0])
        frag_x0s.append(chs[0][0])
        for a, b in zip(chs, chs[1:]):
            if b[0] - a[2] >= gap_thr:
                frag_x0s.append(b[0])
    if not frag_x0s:
        return 0
    tol = 0.6 * med
    frag_x0s.sort()
    clusters: List[List[float]] = [[frag_x0s[0]]]
    for v in frag_x0s[1:]:
        if v - clusters[-1][-1] > tol:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    # 只统计"≥2 片段成员"的簇：单次出现的孤值（行内随机词距、
    # 一次性标注）不构成列签名
    counts: dict = {}
    for v in frag_x0s:
        for cl in clusters:
            if cl[0] - tol <= v <= cl[-1] + tol:
                counts[id(cl)] = counts.get(id(cl), 0) + 1
                break
    return sum(1 for cl in clusters if counts.get(id(cl), 0) >= 2)


def _region_has_wide_wrapped_prose(
    page: fitz.Page, table_bbox: Tuple[float, float, float, float],
    header_rects: Optional[
        List[Tuple[float, float, float, float]]] = None
) -> bool:
    """C3 重建坍缩守卫用：区域是否含通栏折行正文段。

    正文被吸进表格是 C3 的形态学签名（诚志 p63 600 字独立性承诺、
    恒逸 p105 关联交易说明）：≥3 条 y 相邻、同 x0（±0.6×中位字号）、
    每条 ≥8 字、宽 ≥0.4×区域宽、数字占比 <50% 的连续折行行。密排
    数值表（右对齐散布）与窄标签列不满足——防止幻影 x 簇误触发守卫
    拒收正常重建（三书门禁 31 处回归的根因）。
    """
    try:
        chars = _table_char_index(page, table_bbox)
    except Exception:
        return False
    if header_rects:
        def _in_hdr(ch) -> bool:
            cx, cy = (ch[0] + ch[2]) / 2.0, (ch[1] + ch[3]) / 2.0
            return any(hx0 <= cx <= hx1 and hy0 <= cy <= hy1
                       for hx0, hy0, hx1, hy1 in header_rects)
        chars = [ch for ch in chars if not _in_hdr(ch)]
    if not chars:
        return False
    med = _median_char_size(chars)
    bw = table_bbox[2] - table_bbox[0]
    by_line: dict = {}
    for ch in chars:
        by_line.setdefault(ch[5], []).append(ch)
    lns = []
    for chs in by_line.values():
        chs.sort(key=lambda c: c[0])
        txt = "".join(c[4] for c in chs)
        numc = sum(1 for c in chs if c[4] in _NUM_CHARS)
        lns.append({
            "x0": chs[0][0], "y0": min(c[1] for c in chs),
            "n": len(chs), "txt": txt,
            "prose": (len(chs) >= 8
                      and chs[-1][2] - chs[0][0] >= 0.4 * bw
                      and numc < 0.5 * len(chs))})
    lns.sort(key=lambda l: (l["y0"], l["x0"]))
    run = best = 0
    for i, l in enumerate(lns):
        if not l["prose"]:
            run = 0
            continue
        if i > 0 and lns[i - 1]["prose"] \
                and abs(l["x0"] - lns[i - 1]["x0"]) <= 0.6 * med \
                and 0 < l["y0"] - lns[i - 1]["y0"] <= 2.2 * med:
            run += 1
        else:
            run = 1
        best = max(best, run)
    return best >= 3


# ── C5 ruled-grid repair（B1 网格层）──────────────────────────────────────
#
# 根因（trackA §5 仲裁）：三形态跨格互粘的共同根因在 B1 的网格层——
# SLANet 在有完整矢量线网格的 bordered 表上给出劣性布局（幻影空列、
# 相邻列 bbox 包络互相压叠、行缘切进文本行中部）。B1 的字形三级归属
# 再准也救不了网格本身；completion 的几何重建不感知矢量线列边，列边
# 按文本 gutter 三候选竞争，名单折行文字超出线 2-3pt 时真列边（hint）
# 带穿字被少列 gutter 候选击败（yj p35 7 真列坍缩成 4 列）。
#
# 机制：SLANet 网格呈劣性拓扑签名、且表区矢量线自身构成完整网格
# （行线覆盖绝大多数文本带、全高列边 ≥3）时，用矢量线行列边重建
# cell_rects/cell_meta，字形归属仍走原有三级归属 + 行原子化 + 数值
# 归位，随后主流程对该表跳过 complete_table（straddle 检测会把合法
# 的高折行行误判成行漂移触发重建），仅保留幂等文本修复。
#
# 铁律：全部阈值为相对阈值（×中位字号 / ×bbox 宽高 / ×包络宽），零
# 页码/公司名/字符串常量；重建前以 _split_violations 守卫——矢量列
# 边切穿文本 span 的数量不得高于 SLANet 网格，否则放弃重建、保持基
# 线行为（保护跨列合并表头等真合并结构不被均匀网格拆散）。


def _ruled_line_segments(
    page: fitz.Page,
    table_bbox: Tuple[float, float, float, float],
) -> Tuple[List, List]:
    """表区 ±4pt 内的轴向矢量线段（H, V），自建提取。

    不复用 rulings.extract_line_segments：其 HEADER_ZONE 过滤会丢弃
    上边距表格的顶线（横版页表格 y0=56 < 72），而顶线正是行边锚点。
    """
    bx0, by0, bx1, by1 = table_bbox
    H: List = []
    V: List = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return [], []
    for d in drawings:
        for it in d.get("items", ()):
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) < 0.5 and abs(p1.x - p2.x) > 0.5:
                    H.append((min(p1.x, p2.x), max(p1.x, p2.x),
                              (p1.y + p2.y) / 2))
                elif abs(p1.x - p2.x) < 0.5 and abs(p1.y - p2.y) > 0.5:
                    V.append((min(p1.y, p2.y), max(p1.y, p2.y),
                              (p1.x + p2.x) / 2))
            elif it[0] == "re":
                r = it[1]
                w, h = r.width, r.height
                if d.get("fill") and min(w, h) < 1.5:
                    # 细填充矩形：边框描边常以填充矩形出现
                    if w <= h:
                        V.append((r.y0, r.y1, (r.x0 + r.x1) / 2))
                    else:
                        H.append((r.x0, r.x1, (r.y0 + r.y1) / 2))
                elif w < 0.5 and h > 0.5:
                    V.append((r.y0, r.y1, (r.x0 + r.x1) / 2))
                elif h < 0.5 and w > 0.5:
                    H.append((r.x0, r.x1, (r.y0 + r.y1) / 2))
    pad = 4.0
    H = [s for s in H if by0 - pad <= s[2] <= by1 + pad]
    V = [s for s in V if bx0 - pad <= s[2] <= bx1 + pad]
    return H, V


def _merge_axis_positions(segs: List, tol: float = 1.5) -> List:
    """按轴位置（容差 tol）合并线段簇；返回 [(pos, lo, hi)]，pos 为簇
    位置均值、lo/hi 为簇内区间的联合跨度。"""
    groups: List = []
    for s in sorted(segs, key=lambda t: t[2]):
        pos, lo, hi = s[2], s[0], s[1]
        if groups and abs(pos - groups[-1][0]) <= tol:
            p, l0, h0 = groups[-1]
            groups[-1] = ((p + pos) / 2, min(l0, lo), max(h0, hi))
        else:
            groups.append((pos, lo, hi))
    return groups


def _slanet_grid_pathology(cell_meta: List, cell_rects: List,
                           all_chars: List, med: float,
                           bbox: Tuple[float, float, float, float]
                           ) -> Optional[str]:
    """SLANet 网格的劣性拓扑签名（任一命中即劣性）。

    (a1) 相邻列 x 包络压叠 —— 列包络互相越过对方的包络中线附近，
         密排数值列的正常 6-8pt 边缘重叠远达不到该阈值；
    (a2) 幻影空窄列 —— 宽 ≤1.5×中位字号且包络内无任何字形中心；
    (c)  行缘切字 —— SLANet 行边切进文本行中部的行数（漂移网格
         把行界切在字带中间，健康网格行边落在行间空白）。
    """
    bx0, by0, bx1, by1 = bbox
    valid = [r for r in cell_rects if r[2] > r[0] and r[3] > r[1]]
    if not valid:
        return None
    col_env: dict = {}
    for (_sr, _er, sc, _ec, r) in cell_meta:
        if r[2] <= r[0]:
            continue
        lo, hi = col_env.get(sc, (r[0], r[2]))
        col_env[sc] = (min(lo, r[0]), max(hi, r[2]))
    envs = sorted(col_env.values())
    for (l0, h0), (l1, h1) in zip(envs, envs[1:]):
        ov = min(h0, h1) - max(l0, l1)
        if ov > 0.25 * max(1.0, min(h0 - l0, h1 - l1)):
            return "col-envelope-overlap"
    for l0, h0 in envs:
        if 0 < h0 - l0 <= 1.5 * med:
            if not any(l0 - 0.5 <= (c[0] + c[2]) / 2 <= h0 + 0.5
                       for c in all_chars):
                return "phantom-narrow-empty-col"
    lines: dict = {}
    for c in all_chars:
        key = round(c[5], 1)
        lo, hi = lines.get(key, (c[1], c[3]))
        lines[key] = (min(lo, c[1]), max(hi, c[3]))
    if lines:
        y_edges = [e for r in valid for e in (r[1], r[3])
                   if by0 + 2 < e < by1 - 2]
        cut = sum(1 for y0, y1 in lines.values()
                  if any(y0 + 1.0 < e < y1 - 1.0 for e in y_edges))
        if cut >= max(3, int(0.15 * len(lines))):
            return "row-edge-cuts-lines"
    return None


def _ruled_grid_repair(page: fitz.Page,
                       table_bbox: Tuple[float, float, float, float],
                       cell_meta: List, cell_rects: List,
                       all_chars: List,
                       ) -> Optional[Tuple[List, List, int, int]]:
    """矢量线网格重建：劣性 SLANet 网格 + 完整矢量线网格 → 新
    (cell_rects, cell_meta, num_rows, num_cols)；任一条件不满足返回
    None 保持基线。"""
    if not all_chars or not cell_rects:
        return None
    bx0, by0, bx1, by1 = table_bbox
    bw, bh = bx1 - bx0, by1 - by0
    if bw < 10 or bh < 10:
        return None
    med = _median_char_size(all_chars)
    reason = _slanet_grid_pathology(cell_meta, cell_rects, all_chars,
                                    med, table_bbox)
    if reason is None:
        return None

    H, V = _ruled_line_segments(page, table_bbox)
    # 行边：合并后的横线覆盖 ≥0.5×bbox 宽才算行线
    h_edges = []
    for pos, lo, hi in _merge_axis_positions(H):
        cov = min(hi, bx1) - max(lo, bx0)
        if cov >= 0.5 * bw:
            h_edges.append(pos)
    h_edges.sort()
    if len(h_edges) < 2:
        return None
    # 列边：全高（≥0.75×bbox 高）为全局列边；短于全高但 ≥max(4pt,
    # 1.5×字号) 的为局部列边（如信息表仅首行存在的竖线），逐行按
    # 覆盖 ≥0.9×行高采纳
    v_full: List = []
    v_part: List = []
    for pos, lo, hi in _merge_axis_positions(V):
        lo2, hi2 = max(lo, by0), min(hi, by1)
        if hi2 - lo2 >= 0.75 * bh:
            v_full.append(pos)
        elif hi2 - lo2 >= max(4.0, 1.5 * med):
            v_part.append((pos, lo2, hi2))
    if len(v_full) < 3:
        return None
    row_iv = [(a, b) for a, b in zip(h_edges, h_edges[1:])
              if b - a >= 4.0]
    if not row_iv:
        return None
    # 行线必须结构化文本带：≥0.8 的字形行完整落在某个行线区间内
    lines: dict = {}
    for c in all_chars:
        key = round(c[5], 1)
        lo, hi = lines.get(key, (c[1], c[3]))
        lines[key] = (min(lo, c[1]), max(hi, c[3]))
    if not lines:
        return None
    n_cov = sum(1 for y0, y1 in lines.values()
                if any(a - 1.0 <= y0 and y1 <= b + 1.0
                       for a, b in row_iv))
    if n_cov < 0.8 * len(lines):
        return None

    new_rects: List = []
    new_meta: List = []
    n_rows = len(row_iv)
    n_cols = 0
    for ri, (a, b) in enumerate(row_iv):
        xs = list(v_full)
        for pos, lo, hi in v_part:
            cov = min(hi, b) - max(lo, a)
            if cov >= 0.9 * (b - a):
                xs.append(pos)
        xs.sort()
        merged: List = []
        for x in xs:
            if merged and x - merged[-1] <= 1.5:
                continue
            merged.append(x)
        n_cols = max(n_cols, len(merged) - 1)
        for ci in range(len(merged) - 1):
            r = (merged[ci], a, merged[ci + 1], b)
            if r[2] - r[0] < 1.0:
                continue
            new_rects.append(r)
            new_meta.append((ri, ri, ci, ci, r))
    if not new_rects or n_cols < 2:
        return None

    # 守卫：矢量列边切穿文本 span 的数量不得高于 SLANet 网格（跨列
    # 合并表头的均匀化会切字、被此守卫拦下保持 SLANet 结构）
    from assay.completion import _split_violations
    from assay.splitting import _rows as _region_rows
    bands = _region_rows(page, fitz.Rect(*table_bbox))
    if bands:
        my_edges = sorted({v for r in new_rects for v in (r[0], r[2])})
        sl_edges = sorted({v for r in cell_rects
                            if r[2] > r[0] for v in (r[0], r[2])})
        if _split_violations(bands, my_edges) >                 _split_violations(bands, sl_edges):
            return None
    # 守卫G_row（行数塌缩）：仅当 SLANet 行数与文本带数对齐（其
    # 行结构此时可信）而矢量行数显著更少时，判定矢量线只有外框/极
    # 少数行线，重建会把多行塌缩成巨格；SLANet 行与文本带数失配
    # （互绞/幻影行）时其行数不构成证据，放行
    n_bands = len(bands)
    slanet_rows_n = len({m[0] for m in cell_meta}) or 1
    if (n_bands and abs(slanet_rows_n - n_bands) <= 0.25 * n_bands
            and n_rows * 100 < 85 * slanet_rows_n):
        return None
    # 守卫G_lat（区间内隐子表）：某行区间含 ≥3 条文本带且带间空隙
    # 中点聚出 ≥2 个高支持度、不被该行矢量列边解释的列界，说明区
    # 间内部存在未被矢量线刻画的子网格（无内线数据表），重建必错
    if bands and med:
        iv_bands: dict = {}
        for b3 in bands:
            cy = (b3[0] + b3[1]) / 2
            for (a, b4) in row_iv:
                if a - 1 <= cy <= b4 + 1:
                    iv_bands.setdefault((a, b4), []).append(b3)
                    break
        for (a, b4), bl in iv_bands.items():
            if len(bl) < 3:
                continue
            r_edges = sorted({v for r in new_rects
                              if r[1] == a and r[3] == b4
                              for v in (r[0], r[2])})
            gm: List = []
            for (_y0, _y1, spans) in bl:
                ss = sorted(spans)
                for p, q in zip(ss, ss[1:]):
                    if q[0] - p[1] >= max(3.0, 0.5 * med):
                        gm.append((p[1] + q[0]) / 2.0)
            if len(gm) < 2:
                continue
            gm.sort()
            gcl = [[gm[0]]]
            for m in gm[1:]:
                if m - gcl[-1][-1] <= 3.0:
                    gcl[-1].append(m)
                else:
                    gcl.append([m])
            strong = 0
            for cl in gcl:
                cen = sum(cl) / len(cl)
                if (len(cl) >= 0.6 * len(bl)
                        and not any(abs(cen - e) <= 3.0 for e in r_edges)):
                    strong += 1
            if strong >= 2:
                return None
    # 守卫R2（矢量漏列）：多 span 行的宽空隙中点若聚成高支持度列界
    # 且不被任何矢量列边解释，说明存在无内线的真实列界被矢量网格
    # 漏掉——此时保持基线，交由 complete_table 的文本候选处理
    # 守卫G_cover（覆盖）：矢量网格必须覆盖表内文本主体——网格外的
    # 大块文本（无线首末列、表头整带）会在归属阶段溢出成标题/散文本，
    # 说明矢量线只覆盖了表的一部分，不是全表网格
    if new_rects:
        gx0 = min(r[0] for r in new_rects)
        gx1 = max(r[2] for r in new_rects)
        gy0 = min(r[1] for r in new_rects)
        gy1 = max(r[3] for r in new_rects)
        xtol = max(8.0, 1.5 * med)
        in_box = [c for c in all_chars
                  if bx0 <= (c[0] + c[2]) / 2 <= bx1
                  and by0 <= (c[1] + c[3]) / 2 <= by1]
        if not in_box:
            in_box = list(all_chars)
        n_x = sum(1 for c in in_box
                  if c[0] < gx0 - xtol or c[2] > gx1 + xtol)
        if len(in_box) >= 3 and n_x >= 3 and n_x >= 0.02 * len(in_box):
            return None
        n_y = sum(1 for c in in_box
                  if not (gy0 - 1.0 <= (c[1] + c[3]) / 2 <= gy1 + 1.0))
        if n_y > 0.15 * len(in_box):
            return None
    if bands and med:
        col_all = sorted({v for r in new_rects for v in (r[0], r[2])})
        gap_thr = max(4.0, 1.0 * med)
        mids: List = []
        n_gap = 0
        for (y0, y1, _spans) in bands:
            cs = sorted((c for c in all_chars
                         if y0 - 1.0 <= (c[1] + c[3]) / 2 <= y1 + 1.0),
                        key=lambda c: c[0])
            hit = False
            for a, b2 in zip(cs, cs[1:]):
                if b2[0] - a[2] >= gap_thr:
                    mids.append((a[2] + b2[0]) / 2.0)
                    hit = True
            if hit:
                n_gap += 1
        if n_gap >= 3 and len(mids) >= 3:
            mids.sort()
            cls = [[mids[0]]]
            for m in mids[1:]:
                if m - cls[-1][-1] <= 3.0:
                    cls[-1].append(m)
                else:
                    cls.append([m])
            for cl in cls:
                cen = sum(cl) / len(cl)
                if (len(cl) >= 3 and len(cl) >= 0.4 * n_gap
                        and not any(abs(cen - e) <= 3.0 for e in col_all)):
                    return None
    return new_rects, new_meta, n_rows, n_cols, reason


def build_table_from_slanet(
    slanet_result: Dict,
    page: fitz.Page,
    page_idx: int,
    scale_x: float,
    scale_y: float,
    table_bbox: Optional[Tuple[float, float, float, float]] = None,
    header_rects: Optional[
        List[Tuple[float, float, float, float]]] = None,
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
        if header_rects:
            cells = _strip_header_lines(cells, page, header_rects)
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

    # C3 页眉防吸入：表格 bbox 上缘 pad 范围内的共识页眉字形整行
    # 排除（诚志 p63 页眉 y43-53 经 pad 外扩吸进 r0c1）。页眉行是
    # 全书重复的家具文本，绝不可能是表格数据
    if header_rects and all_chars:
        def _in_hdr(ch) -> bool:
            cx, cy = (ch[0] + ch[2]) / 2.0, (ch[1] + ch[3]) / 2.0
            return any(hx0 <= cx <= hx1 and hy0 <= cy <= hy1
                       for hx0, hy0, hx1, hy1 in header_rects)
        all_chars = [ch for ch in all_chars if not _in_hdr(ch)]

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

    # C5 矢量线网格修复：劣性 SLANet 布局 + 完整矢量线网格时，用矢量
    # 线行列边替换 cell_rects/cell_meta，字形归属（下方三级归属 + 行
    # 原子化 + 数值归位）按替换后的网格执行
    ruled_repair = False
    if table_bbox is not None:
        rep = _ruled_grid_repair(page, table_bbox, cell_meta, cell_rects,
                                 all_chars)
        if rep is not None:
            cell_rects, cell_meta, num_rows, num_cols, _why = rep
            n_cells = len(cell_rects)
            ruled_repair = True
            _log.info(f"p{page_idx + 1} C5 ruled-grid repair ({_why}): "
                      f"vector grid {num_rows}x{num_cols} replaces SLANet "
                      f"layout "
                      f"(bbox={tuple(round(v, 1) for v in table_bbox)})")

    # 列包络：同列所有格的 x 并集。右对齐列的表头格居中而数据值右探，
    # SLANet 表头格右界普遍比列的真实右界窄（"2023 年 12 月 31 日"
    # 尾字 '日'、"人民币" 尾字 '币' 落在格外 8-20pt）——按列包络归属
    # 而非单格 bbox，尾字回到本列表头格
    col_env: dict = {}
    for (_sr, _er, sc, _ec, r) in cell_meta:
        lo, hi = col_env.get(sc, (r[0], r[2]))
        col_env[sc] = (min(lo, r[0]), max(hi, r[2]))

    # 行包络：同行所有格的 y 并集——列包络的纵轴对称。SLANet 同行格的
    # 高度常不一致（折行表头只框住首条视觉线、页缘碎片表行高切进文字带），
    # 纵向中心略超单格下沿的字形（折行表头的续行）按行包络归属而非单格
    # bbox，续行字形回到本行格。应用位置与 col_env 一致（重叠/最近格两级），
    # 不含 containment——行带内字形本就被 raw 格包含，无需放宽。
    row_env: dict = {}
    for (sr, _er, _sc, _ec, r) in cell_meta:
        lo, hi = row_env.get(sr, (r[1], r[3]))
        row_env[sr] = (min(lo, r[1]), max(hi, r[3]))

    # 有界行包络：单侧扩界不超过 1×区域中位字号。行高不一致在一个行高
    # 以内是"同一逻辑行、格高量化噪声"（折行表头续行、碎片表行高切线）；
    # 超过一个行高说明邻格本就属于更高的带（跨行合并的数据带），扩界会把
    # 带间边界的孤字错吸进邻行——此时退回 raw 格，保持基线行为。
    _med_y = _median_char_size(all_chars) if all_chars else 9.0

    def _row_bounded_y(ci: int, r) -> Tuple[float, float]:
        renv = row_env.get(cell_meta[ci][0], (r[1], r[3]))
        ey0 = min(r[1], renv[0]) if r[1] - renv[0] <= _med_y else r[1]
        ey1 = max(r[3], renv[1]) if renv[1] - r[3] <= _med_y else r[3]
        return ey0, ey1

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
                ey0, ey1 = _row_bounded_y(ci, r)
                if ey1 + 2 < cy or ey0 - 2 > cy:
                    continue
                env = col_env.get(cell_meta[ci][2], (r[0], r[2]))
                ex0, ex1 = min(r[0], env[0]), max(r[2], env[1])
                ix0, iy0 = max(ch[0], ex0), max(ch[1], ey0)
                ix1, iy1 = min(ch[2], ex1), min(ch[3], ey1)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                score = (ix1 - ix0) * (iy1 - iy0) / carea
                if score > best_score:
                    best, best_score = ci, score
            if best >= 0:
                assigned[best].append(ch)
                continue
            # 最近格兜底：SLANet 窄格边缘的字形（"…31 日" 的"日"）中心
            # 落在所有格之外 0.55×字号 以内时归最近的格，避免边缘字丢失。
            # 仅限横向压线：字形 y 中心必须仍在格的 y 范围 ±0.25×字号 内——
            # 表格上方的表外行（平安 "（货币单位：人民币百万元）" 紧贴表头
            # 上沿，距最近格 5pt）纵向兜进表头会把单位行拆碎注入两格
            cs = 0.55 * _median_char_size(all_chars)
            best, best_dist = -1, cs
            for ci, r in enumerate(cell_rects):
                if r[2] <= r[0] or r[3] <= r[1]:
                    continue
                ey0, ey1 = _row_bounded_y(ci, r)
                if not (ey0 - 0.25 * cs <= cy <= ey1 + 0.25 * cs):
                    continue
                env = col_env.get(cell_meta[ci][2], (r[0], r[2]))
                ex0, ex1 = min(r[0], env[0]), max(r[2], env[1])
                dx = max(ex0 - cx, 0.0, cx - ex1)
                dy = max(ey0 - cy, 0.0, cy - ey1)
                d = (dx * dx + dy * dy) ** 0.5
                if d < best_dist:
                    best, best_dist = ci, d
            if best >= 0:
                assigned[best].append(ch)

    # 数值拆粘归位：SLANet 密排数值列相邻格 bbox 重叠导致的跨格串字
    # （"2,707,138.3|4"、"1|,051,128,788.30"），在转文本前归位
    if all_chars and cell_rects:
        _glue_numeric_seeps(assigned, cell_meta,
                            _median_char_size(all_chars))
        # C3 行原子化：行不被格边切碎（见 _enforce_line_atomicity）
        assigned = _enforce_line_atomicity(
            assigned, all_chars, cell_rects, cell_meta, _med_y)

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

    table_out = TableOutput(
        page_no=page_idx,
        num_rows=num_rows,
        num_cols=num_cols,
        cells=table_cells,
        table_bbox=table_bbox or (0, 0, 0, 0),
    )
    if ruled_repair:
        # C5：主流程据此跳过 complete_table（该表的行/列边已是矢量线
        # 事实，straddle 触发器会把合法高折行行误判成行漂移）
        table_out.ruled_grid = True
    return table_out


# ──────────────────────────────────────────────────────────────────────────────
# Extractor (public API)
# ──────────────────────────────────────────────────────────────────────────────


def prepare_page_slanet_data(
    page: fitz.Page,
    pix: fitz.Pixmap,
    yolo_tables: List[Tuple[float, float, float, float, float]],
    gate_fn,
    structure_fn,
    batch_size: int = GPU_BATCH_SIZE,
    fused_cslanet_fn=None,
    gate_threshold: float = None,
) -> List[Tuple[Dict, Tuple[float, float, float, float]]]:
    """Phase-2 body for ONE table-candidate page (track_p3b stage 3).

    Pure code motion out of ``AssayExtractor.extract``: crop the detected
    boxes, run the PP-LCNet gatekeeper on each crop, then SLANet structure
    recognition on the survivors.  The v2 page-granular pipeline calls this
    with GPU-server-backed gate_fn / structure_fn; the legacy in-process
    path passes bound TableModel methods.  The algorithm (bbox padding,
    image-coordinate mapping, gate order, structure chunking) is unchanged
    — ``batch_size`` only controls the SLANet request granularity (results
    are per-crop regardless of chunking).

    Args:
        page / pix:      candidate page and its PAGE_ZOOM rendering.
        yolo_tables:     detection boxes in PDF points from detect_tables_on_page.
        gate_fn:         crop_img -> bool (gatekeeper threshold baked in by caller).
        structure_fn:    crop_specs -> List[Dict] (TableModel.process_crops contract).
        fused_cslanet_fn: track_p5 optional fused GPU roundtrip:
                         (crop_imgs, offsets) -> List[Dict] with keys
                         'prob' (PP-LCNet gate probability), plus the
                         standard SLANet result keys ('cell_bboxes',
                         'logic_points', 'offset', 'img_size') of EVERY
                         crop, all in one request.  When provided,
                         gate_fn/structure_fn are not called; the gate
                         decision is applied locally against
                         ``gate_threshold`` (same >= comparison the legacy
                         gate_fn performs on the very same probability),
                         and gate-failing crops are dropped from the pairs
                         exactly as before.  SLANet on an eventually
                         gated-out crop is computed but discarded — the
                         observable output is identical.
        gate_threshold:  gate probability floor for the fused path.

    Returns:
        [(slanet_result, table_bbox), ...] in detection order.
    """
    scale_x = page.rect.width / pix.width
    scale_y = page.rect.height / pix.height
    if not yolo_tables:
        return []

    # QW#3: pix.samples → RGB ndarray → PIL Image.  No PNG round trip.
    _pix_arr = np.frombuffer(
        pix.samples, dtype=np.uint8
    ).reshape(pix.height, pix.width, pix.n)
    if _pix_arr.shape[2] > 3:
        _pix_arr = _pix_arr[:, :, :3]
    elif _pix_arr.shape[2] == 1:
        _pix_arr = np.repeat(_pix_arr, 3, axis=2)
    full_img = _Image.fromarray(_pix_arr)

    def _crop_one(yolo_t):
        # The detection bbox is already tight (ruling-line snap or
        # text shrink).  Pad only a few points so SLANet sees the
        # full glyphs sitting on the border lines — never the old
        # greedy expansion that swallowed headers / body text.
        (pdf_x0, pdf_y0, pdf_x1, pdf_y1, _conf) = yolo_t
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
            return None
        crop_img = full_img.crop((img_x0, img_y0, img_x1, img_y1))
        return crop_img, (img_x0, img_y0), orig_bbox

    # ── track_p5 fused path: ONE GPU roundtrip for gate + structure ──
    # The server runs the PP-LCNet gate and SLANet on every crop of the
    # request in parallel and returns per-crop (prob, slanet_result);
    # the gate comparison below is the same >= the legacy gate_fn applies
    # on the very same probability, so observable pairs are identical.
    if fused_cslanet_fn is not None:
        crop_imgs: List = []
        offsets: List[Tuple[int, int]] = []
        bboxes: List[Tuple[float, float, float, float]] = []
        for t in yolo_tables:
            r = _crop_one(t)
            if r is None:
                continue
            crop_imgs.append(r[0])
            offsets.append(r[1])
            bboxes.append(r[2])
        if not crop_imgs:
            return []
        fused = fused_cslanet_fn(crop_imgs, offsets)
        pairs: List[Tuple[Dict, Tuple[float, float, float, float]]] = []
        for i, fres in enumerate(fused):
            # fres: {'prob', 'cell_bboxes', 'logic_points', 'offset',
            #         'img_size'} —— 与 legacy slanet 回包同键 + prob。
            # 门控比较与 legacy gate_fn 完全同一 >=，同一概率值。
            if fres["prob"] < gate_threshold:
                continue
            # 剥掉 prob：sres 与 legacy structure_fn 回包逐键一致。
            sres = {k: fres[k] for k in
                    ("cell_bboxes", "logic_points", "offset", "img_size")}
            pairs.append((sres, bboxes[i]))
        return pairs

    page_crops: List[Dict] = []
    page_meta: List[Dict] = []

    for yolo_t in yolo_tables:
        r = _crop_one(yolo_t)
        if r is None:
            continue
        crop_img, offset, orig_bbox = r

        # Step 4: PP-LCNet gatekeeper
        if not gate_fn(crop_img):
            continue

        page_crops.append({
            'image': crop_img,
            'offset': offset,
        })
        page_meta.append({
            'page_idx': page.number,
            # store the tight detection bbox, NOT the padded crop
            'table_bbox': orig_bbox,
        })

    # Step B: batch SLANet on this page's crops only.
    pairs: List[Tuple[Dict, Tuple[float, float, float, float]]] = []
    for batch_start in range(0, len(page_crops), batch_size):
        batch_end = min(batch_start + batch_size, len(page_crops))
        batch_crops = page_crops[batch_start:batch_end]
        batch_meta = page_meta[batch_start:batch_end]

        slanet_results = structure_fn(batch_crops)
        for j, sres in enumerate(slanet_results):
            meta = batch_meta[j]
            pairs.append((sres, meta['table_bbox']))
    return pairs


def finalize_table_from_slanet(
    sres: Dict,
    page: fitz.Page,
    page_idx: int,
    scale_x: float,
    scale_y: float,
    table_bbox: Tuple[float, float, float, float],
    page_hdr_rects: Optional[List[Tuple[float, float, float, float]]] = None,
) -> Optional[TableOutput]:
    """Phase-3 body for ONE (slanet_result, table_bbox) pair (track_p3b).

    Pure code motion out of ``AssayExtractor.extract``: structure → PDF
    coordinate mapping, phantom-table rejection, content-coverage
    completion, the C3 rebuild-decline guard and header stripping — the
    exact legacy per-table block.  Returns the finished TableOutput or
    None (reject / zero-row).  The v2 build task (D) calls this in a worker
    process; the legacy path calls it from extract() in place.
    """
    from assay.rulings import is_bordered_cell_table, is_partitioned_grid
    from assay.splitting import looks_text_table

    table_out = build_table_from_slanet(
        sres, page, page_idx, scale_x, scale_y, table_bbox,
        header_rects=page_hdr_rects
    )
    if table_out is None:
        return None
    # table_bbox is the trusted ruling-line / value-grid box from detection;
    # it already bounds the table geometrically.  No text-content row
    # trimming here and no grid-collapse reverse.  Each ruled table is its
    # own detection candidate, so fission is unnecessary.
    # Reject phantom tables: a single-column "table" is usually a 1×1 prose
    # fragment, a 2×1 page header or garbled SLANet output.  Exceptions: a
    # closed drawn grid whose interior dividers make cells of non-uniform
    # height is a genuine bordered table (e.g. the "本次现金分红情况"
    # dividend disclosure), and a borderless text table whose rows share a
    # column gutter is one too — SLANet reads those as one column (often one
    # giant 1×1 cell) because no ruling separates the cells (紫光 p146).  The
    # gutter test brings its own multi-row requirement, so it needs no
    # SLANet row count.  Both are decided geometrically — the ML table
    # classifier actually scores such all-text tables *lower* than bordered
    # prose callouts, so it cannot be used here.
    if table_out.num_cols == 1:
        if not ((table_out.num_rows >= 2 and
                 (is_bordered_cell_table(page, table_bbox) or
                  is_partitioned_grid(page, table_bbox))) or
                looks_text_table(page, table_bbox)):
            return None
    if table_out.num_rows > 0:
        # Content-coverage completion: pure content layer running AFTER
        # every detection veto, on tables that survived on their own
        # merits.  It replaces an under-segmented SLANet grid with a
        # geometric rebuild (never adds/removes tables — see
        # assay.completion for its self-checks).
        if getattr(table_out, "ruled_grid", False):
            # C5 矢量线网格修复过的表：行列边已是矢量线事实，
            # complete_table 的结构触发器会把合法高折行行误判成行漂移而
            # 拆毁正确网格，仅保留幂等文本修复；done=False 使下方 done
            # 分支全部跳过
            try:
                from assay.completion import _apply_text_repairs
                table_out.cells = _apply_text_repairs(table_out.cells)
            except ImportError:
                pass
            done = False
        else:
            new_cells, nnr, nnc, done = complete_table(
                page, table_bbox, table_out.cells,
                table_out.num_rows, table_out.num_cols)
        # C3 重建坍缩拒收：几何重建把多列表压成少数粗列、而区域行 x
        # 起点簇数表明真实列数远多于此（诚志 p63：7 列承诺表压成 2 列，
        # 600 字正文与标签错挂、竖排名单劈散），拒绝重建、保留经行原子
        # 化 + 页眉过滤的 SLANet 网格。SLANet 误读出多余列的真坍缩场景
        # （区域 x 起点簇数本就少）不受影响，仍由 complete_table 自身
        # 守卫裁决
        if done and nnc < table_out.num_cols:
            _xcols = _region_xstart_columns(
                page, table_bbox, page_hdr_rects)
            # 第二条件收窄到 C3 形态学签名：区域内存
            # 在通栏折行正文段（≥3 条 y 邻接、同 x0、低数字占比的宽
            # 行）。密排数值表（右对齐散点起点簇多但无正文段）不受守
            # 卫影响，仍允许 complete_table 正常坍缩重建
            if _xcols >= max(table_out.num_cols, nnc + 3) \
                    and _region_has_wide_wrapped_prose(
                        page, table_bbox, page_hdr_rects):
                _log.info(
                    f"p{page_idx + 1} C3 rebuild declined: {nnc} cols < "
                    f"{_xcols} x-start columns")
                done = False
                # 与 complete_table 自身拒绝路径的修饰处理保持一致（空
                # 格/括号修复）。两个 helper 为 completion.py 侧新增接
                # 口，缺省时跳过修饰保持对基线的兼容
                try:
                    from assay.completion import (
                        _fix_cell_text_spaces, _repair_orphan_parens)
                    table_out.cells = _repair_orphan_parens(
                        _fix_cell_text_spaces(table_out.cells))
                except ImportError:
                    pass
        if done:
            table_out.cells = new_cells
            table_out.num_rows = nnr
            table_out.num_cols = nnc
            table_out.completed = True
        if page_hdr_rects:
            table_out.cells = _strip_header_lines(
                table_out.cells, page, page_hdr_rects)
        return table_out
    return None


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
            from assay.detection import (detect_tables_on_page,
                                         top_remnant_candidate,
                                         calibrate_page_headers)

            # 全书学习运行页眉签名（跨页重复共识），供各页文本提取过滤
            header_sigs = calibrate_page_headers(doc)

            table_page_indices: List[int] = []

            for page_idx in range(total_pages):
                page = doc[page_idx]
                rect = page.rect

                result.pages.append(PageOutput(
                    page_no=page_idx,
                    width=rect.width,
                    height=rect.height,
                ))

                texts = extract_page_text(page, page_idx,
                                          header_sigs=header_sigs)
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

            # track_p3a QW#3 + stage 2 streaming:
            #   - QW#3: skip the PNG encode+decode round trip on the full
            #     page pixmap.  pix.samples is already RGB(A) uint8; we go
            #     straight to ``Image.fromarray`` and crop.  Saves ~4.6s
            #     on a 296-page book (3.6s encode + 1.0s decode) per P1.
            #   - stage 2: SLANet batching moves INSIDE the per-page loop.
            #     ``all_crops`` is page-scoped (one page at a time) instead
            #     of book-scoped (whole book accumulated).  Single-page
            #     crop list typically holds 0-5 PIL images vs ~330 in the
            #     book-wide accumulator.  The page-local list is released
            #     at the end of each iteration, so peak RSS drops by
            #     ~1.5GB on a 296-page book.
            # track_p3b stage 3: the per-page crop/gate/SLANet body now
            # lives in prepare_page_slanet_data (module-level, shared with
            # the v2 page-granular pipeline).  Same algorithm, same
            # page-scoped accumulation (QW#3 + stage 2 streaming kept).
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

                pairs = prepare_page_slanet_data(
                    page, pix, yolo_tables,
                    lambda img, _m=self.model: _m.is_table(
                        img, threshold=GATEKEEPER_THRESHOLD),
                    self.model.process_crops,
                )
                if pairs:
                    gpu_results.setdefault(page_idx, []).extend(pairs)

            # ── Phase 3: coordinate mapping + text extraction ────────────
            from assay.rulings import is_bordered_cell_table, is_partitioned_grid
            from assay.splitting import looks_text_table
            for page_idx in table_page_indices:
                page = doc[page_idx]
                slanet_data = gpu_results.get(page_idx, [])
                if not slanet_data:
                    continue

                scale_x, scale_y = _page_cache[page_idx]
                page_hdr_rects = _running_header_rects(page, header_sigs)

                for sres, table_bbox in slanet_data:
                    table_out = finalize_table_from_slanet(
                        sres, page, page_idx, scale_x, scale_y, table_bbox,
                        page_hdr_rects=page_hdr_rects
                    )
                    if table_out is not None:
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
