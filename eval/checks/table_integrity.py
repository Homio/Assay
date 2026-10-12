#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C4 整行/整表丢失校验器（离线、只读）。

根因 C4（整行/整表丢失）的离线校验：读管道产物（<stem>.json 的 tables 网格）
+ 原始 PDF，输出缺陷清单 JSON。不修改任何文件、不改管道运行时行为。

三类检查（对应 fix_recommendations C4 修法建议 1/2/3）：
  1. sum_identity   含「合计/总计」行的表：数据行求和 vs 合计（含小计/其中
                    变体、跨页 fragment 合并复算）。不等 →
                    potentially_missing_row（合并后仍不闭合，高严重）或
                    cross_page_split_identity（仅单 fragment 不闭合、合并后
                    闭合，页内视角恒等式断裂，中严重）。
  2. cross_page     跨页表 fragment 识别（同列数 + x 包络对齐 + 上页表未以
                    合计收尾 + 几何位置约束），对每个 fragment 做 PDF 视觉行
                    覆盖核对，中间行缺失 → cross_page_missing_row。
  3. row_coverage   每张表：网格行数 / PDF 视觉行数（表区横 ruling 线带数，
                    无 ruling 时退化为文本 y 聚类）< min_coverage →
                    row_coverage_low。

阈值全部相对化（×页宽/×页高/×行高/×求和行数），无书目/页码/绝对值硬编码。

用法：
  python3 eval/checks/table_integrity.py \
      --output-dir /home/homio/sda/1111/output \
      --pdf-dir /home/homio/sda/1111/original \
      [--book <stem子串> ...] [--min-coverage 0.9] \
      [--out /path/to/report.json]
"""

import argparse
import glob
import itertools
import json
import os
import re
import sys
from collections import defaultdict

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# 常量（全部为相对参数或通用词表，无书目/页码硬编码）
# ---------------------------------------------------------------------------

TOTAL_RE = re.compile(r'^[合总]\s*计$')
SUBTOTAL_RE = re.compile(r'^小\s*计$')
WHEREOF_RE = re.compile(r'^其中')

# 几何相对阈值
X_ENVELOPE_TOL = 0.02      # 跨页 fragment x 包络对齐容差 ×页宽
TOP_REGION = 0.35          # 续页 fragment 顶部区域 ×页高
BOTTOM_REGION = 0.65       # 上页 fragment 底部区域 ×页高
RULE_LINE_MIN_SPAN = 0.55  # 横 ruling 线最短跨度 ×表宽
RULE_CLUSTER_TOL = 2.0     # ruling 线 y 聚类容差（pt，防重复绘制线）
TEXT_CLUSTER_TOL_RATIO = 0.7  # 文本行聚类容差 ×行高（无 ruling 退化路径）
MIN_VISUAL_ROWS = 3        # 视觉行数低于此值不做覆盖检查（噪声区）
MIN_PARSE_COVERAGE = 0.6   # 数值解析成功率低于此值跳过该列恒等式检查
SIGN_FLIP_TAIL = 4        # 合计行前参与符号翻转候选的行数上限
SIGN_FLIP_MAX = 3         # 符号翻转候选的最大子集行数
COVERAGE_DEFAULT = 0.9     # MD 行数 / PDF 视觉行数 下限


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

_FW_TRANS = {ord(f): ord(t) for f, t in zip(
    '０１２３４５６７８９．，％－（）　',
    '0123456789.,%-() ')}


def fw_to_half(s):
    return s.translate(_FW_TRANS)


def clean_label(s):
    """行标签归一：全角转半角、去所有空白，用于合计/小计匹配。"""
    return re.sub(r'\s+', '', fw_to_half(s or ''))


def collapse_ws(s):
    return re.sub(r'[\s\u00a0]+', '', s or '')


AMOUNT_FULL_RE = re.compile(r'^-?\(?\d[\d,]*(?:\.\d+)?\)?$')


def parse_amount(text):
    """解析单元格数值。返回 (value, ok)。

    空串 / '-' / '--' / '---' 视为 0（表格空位）；解析失败返回 (None, False)。
    内部空白先剔除（C1 破碎形态 '18,335,694,335.\\n41' / '2 502' 需拼回）。
    """
    if text is None:
        return 0.0, True
    s = collapse_ws(fw_to_half(str(text)))
    if s in ('', '-', '--', '---', '/', '—'):
        return 0.0, True
    neg = False
    if s.endswith('%'):
        return None, False  # 百分比列不做金额恒等式
    if s.startswith('(') and s.endswith(')'):
        neg = True
        s = s[1:-1]
    elif s.startswith('-'):
        neg = True
        s = s.lstrip('-')
    if not re.fullmatch(r'\d[\d,]*(?:\.\d+)?', s):
        return None, False
    if ',' in s:
        ip = s.split('.')[0]
        if not re.fullmatch(r'\d{1,3}(?:,\d{3})+', ip):
            return None, False
    try:
        v = float(s.replace(',', ''))
    except ValueError:
        return None, False
    return (-v if neg else v), True


def has_percent(cells):
    return any('%' in (c.get('text') or '') for c in cells)


def cell_text(c):
    return (c.get('text') or '').strip()


def row_texts(row):
    return [cell_text(c) for c in row]


def row_label(row):
    for t in row_texts(row):
        t2 = clean_label(t)
        if t2:
            return t2
    return ''


def is_header_row(row):
    if any(c.get('column_header') for c in row):
        return True
    lab = row_label(row)
    return bool(re.fullmatch(r'项目|单位[：:].*|科目|项目[／/].*', lab))


def label_col_index(grid):
    """标签列 = CJK 字符数最多的列（无 CJK 时取第 0 列）。"""
    best, best_n = 0, -1
    if not grid:
        return 0
    ncols = max(len(r) for r in grid)
    for ci in range(ncols):
        n = 0
        for r in grid:
            if ci < len(r):
                n += len(re.findall(r'[\u4e00-\u9fff]', cell_text(r[ci])))
        if n > best_n:
            best, best_n = ci, n
    return best


# ---------------------------------------------------------------------------
# 检查 1：合计恒等式
# ---------------------------------------------------------------------------

def _tolerance(values):
    """求和容差：全整数表（元位舍入）按 0.5×n，带小数表（分位舍入）按
    0.006×n。相对化于参与求和的行数，非绝对常数。"""
    n = len(values)
    if n == 0:
        return 0.01
    if all(abs(v - round(v)) < 1e-9 for v in values if v is not None):
        return 0.5 * n + 0.5
    return 0.006 * n + 0.02


def check_sum_identity(rows, label_ci, table_ref):
    """对单张表（rows = 数据行列表，含合计行）做恒等式检查。

    返回 defects 列表（dict）。跨页合并复算由调用方传入合并后的 rows。
    """
    defects = []
    # 分组：每个合计行闭上自上一个合计行以来的组
    groups = []  # (total_idx, item_rows, subtotal_rows, whereof_rows)
    cur_items, cur_subtotals, cur_whereof = [], [], []
    in_whereof_block = False
    for ri, row in enumerate(rows):
        lab = row_label(row)
        if TOTAL_RE.fullmatch(lab):
            groups.append((ri, list(cur_items), list(cur_subtotals),
                           list(cur_whereof)))
            cur_items, cur_subtotals, cur_whereof = [], [], []
            in_whereof_block = False
        elif SUBTOTAL_RE.fullmatch(lab):
            cur_subtotals.append(ri)
            in_whereof_block = False
        elif re.fullmatch(r'其中[：:]?', lab):
            # 「其中：」标签行（值格为空）：其后的行是分解行
            cur_whereof.append(ri)
            in_whereof_block = True
        elif WHEREOF_RE.match(lab):
            cur_whereof.append(ri)
            in_whereof_block = True
        elif in_whereof_block:
            # 分解块延续：父行已含其值，归 whereof 不参与主求和
            cur_whereof.append(ri)
        elif is_header_row(row):
            continue
        else:
            cur_items.append(ri)

    ncols = max((len(r) for r in rows), default=0)
    for (tri, items, subtotals, whereof) in groups:
        total_row = rows[tri]
        for ci in range(ncols):
            if ci == label_ci:
                continue
            if ci >= len(total_row):
                continue
            t_txt = clean_label(cell_text(total_row[ci]))
            tv, tok = parse_amount(cell_text(total_row[ci]))
            if not tok or t_txt in ('', '-', '--', '---', '/', '—', '——'):
                # 合计格为空/占位符 → 该列无有效合计，跳过
                continue
            if has_percent(total_row):
                # 合计行含百分号 → 该行不做金额恒等式（比率行）
                if '%' in cell_text(total_row[ci]):
                    continue
            members = [(ri, rows[ri], ci) for ri in items + subtotals
                       + whereof if ci < len(rows[ri])]
            vals, excl, nonempty = [], 0, 0
            for ri, row, ci2 in members:
                txt = cell_text(row[ci2])
                if txt:
                    nonempty += 1
                v, ok = parse_amount(txt)
                if ok:
                    vals.append((ri, v))
                elif txt:
                    excl += 1
            n_all = len(items) + len(subtotals)
            if n_all == 0:
                continue
            if nonempty and excl / max(nonempty, 1) > (1 - MIN_PARSE_COVERAGE):
                defects.append(_mk('sum_identity', 'info',
                                   'low_parse_coverage', table_ref,
                                   col=ci, total_row=tri,
                                   note='数值解析成功率不足，恒等式检查跳过',
                                   excluded=excl, nonempty=nonempty))
                continue
            tol = _tolerance([v for _, v in vals])
            item_set = set(items)
            sub_set = set(subtotals)
            whereof_set = set(whereof)

            def s(filter_set=None, adj=False):
                tot = 0.0
                for ri, v in vals:
                    if filter_set is not None and ri not in filter_set:
                        continue
                    if adj:
                        # 「减：X」行对合计的贡献 = -X（按显示值翻转符号）；
                        # 「加：」行按显示值直接计入，无需调整
                        lab = row_label(rows[ri])
                        if lab.startswith('减'):
                            v = -v
                    tot += v
                return tot

            cands = [('items+subtotals', s(item_set | sub_set)),
                     ('items+subtotals+whereof', s()),
                     ('items+subtotals_flip_minus', s(item_set | sub_set, True)),
                     ('all_adj_minus', s(None, True))]
            if subtotals:
                cands.append(('items_only', s(item_set)))
                cands.append(('subtotals_only', s(sub_set)))
            matched = next(((nm, sv) for nm, sv in cands
                            if abs(sv - tv) <= tol), None)
            if matched is not None:
                # 候选闭合（items+subtotals / 含其中 / 减行翻转等）→ 无缺陷
                continue
            # 尾部符号翻转候选：合计行之前 ≤SIGN_FLIP_TAIL 行里选
            # ≤SIGN_FLIP_MAX 子集翻转显示值符号后闭合。
            # 语义依据：「减：X」行（含无前缀的伴随行）对合计贡献为 -X，
            # 与按显示值直接求和恰差 2×X。
            s_main = s(item_set | sub_set)
            resid = tv - s_main
            tail = sorted(item_set | sub_set)[-SIGN_FLIP_TAIL:]
            tvals = dict(vals)
            found = False
            for k in range(1, SIGN_FLIP_MAX + 1):
                if k > len(tail):
                    break
                for comb in itertools.combinations(tail, k):
                    if any(ri not in tvals for ri in comb):
                        continue
                    # 翻转后闭合：tv ≈ s_main ± 2×sum(comb)，两个方向都试
                    # （显示值为正/负的语义减行分别对应 -/+）
                    sv2 = 2.0 * sum(tvals[ri] for ri in comb)
                    if abs(resid - sv2) <= tol or abs(resid + sv2) <= tol:
                        found = True
                        break
                if found:
                    break
            if found:
                # 翻转候选闭合：语义减行按显示值入账导致 2×X 偏差，
                # 不是丢行，不报缺陷
                continue
            diff = tv - s_main
            defects.append(_mk(
                'sum_identity', 'high', 'potentially_missing_row', table_ref,
                col=ci, total_row=tri, sum=s_main, total=tv, diff=diff,
                note='SUM(数据行) != 合计',
                whereof_rows=len(whereof), excluded_rows=excl))
    return defects


def _mk(check, severity, dtype, table_ref, **kw):
    d = {'check': check, 'severity': severity, 'type': dtype,
         'book': table_ref.get('book'), 'table_index': table_ref.get('ti'),
         'page_no': table_ref.get('page'),
         'pdf_page_1based': (table_ref.get('page') + 1)
         if table_ref.get('page') is not None else None}
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
# 检查 2/3：PDF 视觉行 + 覆盖率 + 跨页 fragment
# ---------------------------------------------------------------------------

class PdfCache(object):
    def __init__(self, pdf_dir):
        self.pdf_dir = pdf_dir
        self.docs = {}

    def doc(self, stem):
        if stem not in self.docs:
            path = os.path.join(self.pdf_dir, stem + '.pdf')
            self.docs[stem] = fitz.open(path)
        return self.docs[stem]


def visual_rows_by_rules(drawings, words, bbox):
    """用表区横 ruling 线计数视觉行带。

    ruling 线在 PDF 中常被画成多段细 fill 矩形（单段不足表宽），因此按
    y 聚类后取段并集的 x 覆盖率判定（覆盖率 >= RULE_LINE_MIN_SPAN×表宽）。
    bbox: [x0, y0, x1, y1]。返回 (n_rows, method)。
    """
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    if w <= 0:
        return None, 'degenerate'
    segs = []  # (y, xa, xb)
    for d in drawings:
        r = d.get('rect')
        if r is not None:
            cy = (r.y0 + r.y1) / 2.0
            if y0 - RULE_CLUSTER_TOL <= cy <= y1 + RULE_CLUSTER_TOL \
                    and r.height <= RULE_CLUSTER_TOL and r.width > 0.5:
                segs.append((cy, r.x0, r.x1))
        for it in d.get('items', ()):
            if it and it[0] == 'l':
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) <= 0.5 and abs(p1.x - p2.x) > 0.5:
                    cy = (p1.y + p2.y) / 2.0
                    if y0 - RULE_CLUSTER_TOL <= cy <= y1 + RULE_CLUSTER_TOL:
                        segs.append((cy, min(p1.x, p2.x), max(p1.x, p2.x)))
    if not segs:
        return None, 'no_rules'
    segs.sort()
    clusters = []
    for y, xa, xb in segs:
        if clusters and y - clusters[-1][-1][0] <= RULE_CLUSTER_TOL:
            clusters[-1].append((y, xa, xb))
        else:
            clusters.append([(y, xa, xb)])
    centers = []
    for c in clusters:
        ivs = sorted((xa, xb) for _, xa, xb in c)
        merged = [list(ivs[0])]
        for a, b in ivs[1:]:
            if a <= merged[-1][1] + 1.0:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        cov = sum(b - a for a, b in merged)
        if cov >= RULE_LINE_MIN_SPAN * w:
            centers.append(sum(y for y, _, _ in c) / len(c))
    if len(centers) < 2:
        return None, 'no_rules'
    # 行带 = 相邻线之间；剔除完全无文本的带（表框外延/装饰线）
    n_rows = 0
    for i in range(len(centers) - 1):
        a, b = centers[i], centers[i + 1]
        has_text = False
        for wd in words:
            wy = (wd[1] + wd[3]) / 2.0
            if a <= wy <= b and wd[0] <= x1 + 2 and wd[2] >= x0 - 2:
                has_text = True
                break
        if has_text:
            n_rows += 1
    if n_rows == 0:
        return None, 'no_rules'
    return n_rows, 'rules'


def visual_rows_by_text(words, bbox):
    """无 ruling 线时的退化路径：表区内文本 y 聚类。

    行高取词高中位数的 1.0 倍做容差，避免把折行标签算成多行。
    """
    x0, y0, x1, y1 = bbox
    words = [w for w in words
             if x0 - 2 <= w[0] and w[2] <= x1 + 2
             and y0 - 2 <= (w[1] + w[3]) / 2.0 <= y1 + 2]
    if not words:
        return None, 'no_text'
    ys = sorted((w[1] + w[3]) / 2.0 for w in words)
    # 以词高中位数的 1.0 倍为聚类容差（相对化于字号）
    h = sorted(w[3] - w[1] for w in words)
    med_h = h[len(h) // 2] if h else 10.0
    tol = max(med_h * TEXT_CLUSTER_TOL_RATIO, 1.5)
    clusters = [[ys[0]]]
    for y in ys[1:]:
        if y - clusters[-1][-1] <= tol:
            clusters[-1].append(y)
        else:
            clusters.append([y])
    return len(clusters), 'text_clusters'


def visual_row_count(drawings, words, bbox):
    n, method = visual_rows_by_rules(drawings, words, bbox)
    if n is None:
        n, method = visual_rows_by_text(words, bbox)
    return n, method


def detect_cross_page_pairs(tables_meta, page_sizes):
    """识别跨页表 fragment 对。

    条件（全部相对化）：
      - 页码相邻；
      - 列数相同；
      - x 包络差 ≤ X_ENVELOPE_TOL×页宽；
      - 上页表未以「合计/总计」行收尾（已闭合的表不再延续）；
      - 几何：上表到达页底区域或为该页最底表；下表位于页顶区域或为该页最顶表。
    返回 [(ti_a, ti_b), ...]
    """
    by_page = defaultdict(list)
    for m in tables_meta:
        by_page[m['page']].append(m)
    for p in by_page:
        by_page[p].sort(key=lambda m: m['bbox'][1])
    pairs = []
    for p, ms in by_page.items():
        nxt = by_page.get(p + 1)
        if not nxt:
            continue
        pw, ph = page_sizes.get(p, (None, None))
        if not pw:
            continue
        a = ms[-1]
        b = nxt[0]
        if abs(a['cols'] - b['cols']) > 1:
            continue
        if abs(a['bbox'][0] - b['bbox'][0]) > X_ENVELOPE_TOL * pw or \
                abs(a['bbox'][2] - b['bbox'][2]) > X_ENVELOPE_TOL * pw:
            continue
        if TOTAL_RE.fullmatch(a['last_label']):
            continue
        b_first_is_header = bool(b['grid']) and is_header_row(b['grid'][0])
        if b_first_is_header and b['bbox'][1] > TOP_REGION * ph:
            # 表头起且不在页顶 → 更可能是新表（重复表头续表仍可经页顶命中）
            continue
        bottom_ok = a['bbox'][3] >= BOTTOM_REGION * ph or a is ms[-1]
        top_ok = b['bbox'][1] <= TOP_REGION * ph or b is nxt[0]
        if bottom_ok and top_ok:
            pairs.append((a['ti'], b['ti']))
    return pairs


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def process_book(stem, out_dir, pdf_cache, min_coverage):
    subdir = os.path.join(out_dir, stem)
    jpath = os.path.join(subdir, stem + '.json')
    if not os.path.exists(jpath):
        return None
    with open(jpath, encoding='utf-8') as f:
        data = json.load(f)
    doc = pdf_cache.doc(stem)
    pages = data.get('pages', {})
    page_sizes = {int(k): (v['size']['width'], v['size']['height'])
                  for k, v in pages.items()}
    tables = data.get('tables', [])
    meta = []
    for ti, t in enumerate(tables):
        prov = t.get('prov') or [{}]
        pno = prov[0].get('page_no')
        bbox = prov[0].get('bbox') or [0, 0, 0, 0]
        grid = t.get('data', {}).get('grid', [])
        meta.append({'ti': ti, 'page': pno, 'bbox': bbox,
                     'cols': t.get('data', {}).get('num_cols',
                                                   max((len(r) for r in grid),
                                                       default=0)),
                     'grid': grid,
                     'last_label': row_label(grid[-1]) if grid else ''})

    book_ref = {'book': stem}
    defects = []
    stats = {'tables': len(tables), 'sum_checked': 0, 'sum_groups': 0,
             'coverage_checked': 0, 'cross_page_pairs': 0}

    pairs = detect_cross_page_pairs(meta, page_sizes)
    # 链式合并（A→B→C 同一张跨三页表）
    merged_groups = {}
    for a_ti, b_ti in pairs:
        grp = merged_groups.get(a_ti)
        if grp is None:
            grp = [a_ti]
            merged_groups[a_ti] = grp
        grp.append(b_ti)
        merged_groups[b_ti] = grp
    uniq_groups = {id(g): g for g in merged_groups.values()}.values()
    group_starts = {g[0]: g for g in uniq_groups}
    frag_group = {}
    for g in group_starts.values():
        for t2 in g:
            frag_group[t2] = g

    stats['cross_page_pairs'] = len(pairs)

    page_ctx = {}

    def get_ctx(pno):
        if pno not in page_ctx:
            pg = doc[pno]
            page_ctx[pno] = (pg.get_drawings(), pg.get_text('words'))
        return page_ctx[pno]

    for ti, m in enumerate(meta):
        ref = dict(book_ref, ti=ti, page=m['page'])
        grid = m['grid']
        if not grid:
            continue
        label_ci = label_col_index(grid)

        # --- 检查 1a：单表恒等式 ---
        d1 = check_sum_identity(grid, label_ci, ref)
        # --- 检查 1b：跨页合并恒等式（组首触发一次，合并全部 fragment）---
        g = frag_group.get(ti)
        merged_defects = []
        if g and len(g) > 1 and group_starts.get(g[0]) and ti == g[0]:
            rows = []
            for t2 in g:
                rows.extend(meta[t2]['grid'])
            lc = label_col_index(rows)
            mref = dict(book_ref, ti=g[0], page=meta[g[0]]['page'],
                        merged_tables=g)
            merged_defects = check_sum_identity(rows, lc, mref)
            for dd in merged_defects:
                if dd['type'] == 'potentially_missing_row':
                    dd['note'] = '跨页合并后 SUM 仍 != 合计'
        if g and len(g) > 1:
            # 合并口径闭合（merged_defects 无 high）而单 fragment 不闭合
            # → 降级为 split_identity（页内视角恒等式断裂）
            merged_broken = any(dd['type'] == 'potentially_missing_row'
                                for dd in merged_defects)
            for dd in d1:
                if dd['type'] == 'potentially_missing_row' and not merged_broken:
                    dd['type'] = 'cross_page_split_identity'
                    dd['severity'] = 'medium'
                    dd['note'] = ('单页 fragment 恒等式不闭合，跨页合并后闭合'
                                  '（表在页界被切断，页内视角缺行）')
                    dd['merged_tables'] = g
                    dd['fragment_pages_1based'] = [
                        meta[t2]['page'] + 1 for t2 in g
                        if meta[t2]['page'] is not None]
        defects.extend(d1)
        defects.extend(dd for dd in merged_defects
                       if dd['type'] == 'potentially_missing_row')
        stats['sum_groups'] += sum(1 for d in d1 + merged_defects
                                   if d['type'] != 'low_parse_coverage')
        stats['sum_checked'] += 1

        # --- 检查 3：行覆盖兜底（含跨页 fragment 各自核对，即检查 2 的
        #     「第二页起内容须延续、中间行缺失告警」落地）---
        pno = m['page']
        if pno is None or pno >= doc.page_count:
            continue
        drawings, words = get_ctx(pno)
        n_md = len(grid)
        wrapped = any('\n' in (c.get('text') or '')
                      for r in grid for c in r)
        n_vis, method = visual_row_count(drawings, words, m['bbox'])
        if n_vis is None or n_vis < MIN_VISUAL_ROWS:
            continue
        if method != 'rules' and wrapped:
            # 无 ruling 线 + 网格含折行单元格：文本 y 聚类必然高估视觉行数，
            # 覆盖率兜底在该退化路径下不可信，跳过（记 info）
            defects.append(_mk('row_coverage', 'info',
                               'coverage_skip_wrapped', ref,
                               md_rows=n_md, note=(
                                   '无 ruling 线且网格含折行单元格，'
                                   'text 聚类路径不可信，跳过覆盖率检查')))
            continue
        cov = n_md / float(n_vis)
        stats['coverage_checked'] += 1
        if cov < min_coverage:
            extra = ''
            if g and len(g) > 1 and ti != g[-1]:
                extra = '（跨页表非末 fragment，视觉行含被切分行时比例偏低）'
            defects.append(_mk(
                'row_coverage', 'medium' if method == 'rules' else 'low',
                'row_coverage_low', ref,
                md_rows=n_md, visual_rows=n_vis, coverage=round(cov, 3),
                visual_method=method, note='MD行数/PDF视觉行数 < 阈值' + extra))
    return {'stem': stem, 'defects': defects, 'stats': stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--pdf-dir', required=True)
    ap.add_argument('--book', action='append', default=[],
                    help='只处理 stem 含该子串的书（可多次）')
    ap.add_argument('--min-coverage', type=float, default=COVERAGE_DEFAULT)
    ap.add_argument('--out', default=None, help='缺陷清单 JSON 输出路径')
    args = ap.parse_args()

    out_dir = args.output_dir
    subdirs = sorted(d for d in os.listdir(out_dir)
                     if os.path.isdir(os.path.join(out_dir, d)))
    if args.book:
        subdirs = [d for d in subdirs if any(b in d for b in args.book)]

    pdf_cache = PdfCache(args.pdf_dir)
    books = {}
    total_defects = []
    for stem in subdirs:
        try:
            r = process_book(stem, out_dir, pdf_cache, args.min_coverage)
        except Exception as e:  # 单本失败不拖垮整批
            books[stem] = {'error': repr(e)}
            continue
        if r is None:
            continue
        books[stem] = {'stats': r['stats'], 'defects': r['defects']}
        total_defects.extend(r['defects'])
        n = len(r['defects'])
        sev = defaultdict(int)
        for d in r['defects']:
            sev[d['severity']] += 1
        print('%-60s tables=%-4d defects=%-4d %s' % (
            stem[:60], r['stats']['tables'], n, dict(sev)), file=sys.stderr)

    sev_all = defaultdict(int)
    type_all = defaultdict(int)
    for d in total_defects:
        sev_all[d['severity']] += 1
        type_all[d['type']] += 1
    report = {
        'meta': {
            'output_dir': out_dir, 'pdf_dir': args.pdf_dir,
            'min_coverage': args.min_coverage,
            'n_books': len(books),
            'total_tables': sum(b['stats']['tables'] for b in books.values()
                                if 'stats' in b),
            'pipeline_untouched': True,
            'thresholds': {
                'x_envelope_tol_x_page_width': X_ENVELOPE_TOL,
                'top_region_x_page_height': TOP_REGION,
                'bottom_region_x_page_height': BOTTOM_REGION,
                'rule_line_min_span_x_table_width': RULE_LINE_MIN_SPAN,
                'min_visual_rows': MIN_VISUAL_ROWS,
                'min_parse_coverage': MIN_PARSE_COVERAGE,
                'rounding_tol_formula': '0.5*n+0.5 (整数表) / 0.006*n+0.02 (小数表)',
            },
        },
        'summary': {'defects_total': len(total_defects),
                    'by_severity': dict(sev_all), 'by_type': dict(type_all)},
        'defects': total_defects,
        'books': books,
    }
    out = args.out
    if out:
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print('report -> %s (%d defects)' % (out, len(total_defects)),
              file=sys.stderr)
    else:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
