# -*- coding: utf-8 -*-
"""assay/normalize.py — C7 抽取后处理规范化模块（可选接入）。

修复根因 C7（复选符号 / 单位 / 数值格式规范缺失）：
  1. 复选符号白名单标准化：PUA 私有区字形（Wingdings \uf052 等）、
     √ / ✓ / ✔ 统一映射为 ☑（勾选）；□ / ☐ 保留映射为 ☐（未勾选）；
     ☒（叉选）视为已勾选 → ☑。不允许乱码或丢字。
  2. 单位字符串标准化：`单位:元` / `单位： 元` / `单位 ：元` 等变体
     统一为 `单位：元`（全角冒号、无多余空格）。
  3. 数值格式规范化：
     - `- ( 687,860,833.00)` → `-687,860,833.00`（负号+空格+括号）
     - `(99,361.07)` → `-99,361.07`（会计括号负数统一为负号）
     - `--25,093,729.87` → `25,093,729.87`（重复负号按奇偶性归并：
       偶数个负号 = 正数，奇数个 = 负数）
     - 数字内部空格 `1, 234.56` → `1,234.56`
     - 全角数字 / 小数点 / 逗号 / 百分号转半角

纯后处理函数，不改管道运行时行为；由输出层选购用（见
analysis/fix_c1c8/INTERFACE_REQUESTS.md）。零硬编码阈值——全部为
白名单 / 字形映射 / 严格正则，无量纲参数。

用法（CLI）：
    python3 assay/normalize.py --md <产物.md> [--out <路径>] \
        [--report <json路径>] [--diff]
    cat x.md | python3 assay/normalize.py --stdin

用法（库）：
    from assay.normalize import normalize_text, normalize_markdown
    new, report = normalize_markdown(md_text)
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# ---------------------------------------------------------------------------
# 1. 复选符号白名单（字形映射，非阈值）
# ---------------------------------------------------------------------------
CHECKED = '\u2611'     # ☑
UNCHECKED = '\u2610'   # ☐

# 已勾选的等价字形 → ☑
CHECKED_ALIASES = {
    '\uf052',   # Wingdings checked box（一彬/诚志 PUA 私有区）
    '\uf0fe',   # Wingdings check variant
    '\u221a',   # √ 平方根号被用作勾选（ST中地）
    '\u2713',   # ✓
    '\u2714',   # ✔
    '\u2611',   # ☑ 自身（幂等）
    '\u2612',   # ☒ 叉选 = 已勾选
}
# 未勾选的等价字形 → ☐
UNCHECKED_ALIASES = {
    '\u25a1',   # □ 白方块
    '\u2610',   # ☐ 自身（幂等）
    '\uf0a8',   # Wingdings empty box
}

# ---------------------------------------------------------------------------
# 2. 数值格式正则（严格匹配，避免误伤正文）
# ---------------------------------------------------------------------------
_NUM_CORE = (r'\d{1,3}(?:,\d{3})+(?:\.\d+)?'   # 千分位：1,234 / 1,234.56
             r'|\d+\.\d+')                     # 小数：1234.56
             # 纯整数（如年份 2025）不匹配——括号/负号规则不误伤
# 负号 + 可选空格 + 括号包裹的正数：- ( 687,860,833.00 )
RE_MINUS_PAREN = re.compile(
    r'(?<![\w%.])-+\s*\(\s*(' + _NUM_CORE + r')\s*\)')
# 单纯括号负数（会计记法）：(99,361.07) —— 前面不能是数字/字母/./%
RE_PAREN_NEG = re.compile(
    r'(?<![\w.%])\(\s*(' + _NUM_CORE + r')\s*\)(?!\w)')
# 连续负号 + 数字（双负号按奇偶归并）
RE_MULTI_MINUS = re.compile(r'(--+)\s*(' + _NUM_CORE + r')')
# 数字内部的空格（千分位逗号后的空格粘连）
# 前瞻式（尾随数字不进匹配）：连续多处 `1, 234, 567` 单遍即可全部收敛
RE_NUM_INNER_SPACE = re.compile(
    r'\d,\s+(?=\d{3})|\d\s+(?=\.\d)')
# 表行级括号跨格切分，两种严格形态：
#   形态1：`| - ( | 687,860,833.00) |` —— A=[负号+]左括号，B=数字+右括号
#   形态2：`| - | (117,039,002.86) |`  —— A=裸负号，B=括号负数
RE_CELL_SPLIT_PAREN_A = re.compile(r'^\s*(-?\s*)\(\s*$')
RE_CELL_SPLIT_PAREN_B = re.compile(r'^\s*(' + _NUM_CORE + r')\s*\)\s*$')
RE_CELL_SPLIT_MINUS_A = re.compile(r'^\s*-\s*$')
RE_CELL_SPLIT_MINUS_B = re.compile(r'^\s*\(\s*(' + _NUM_CORE + r')\s*\)\s*$')
# 表行分隔（MD 管道表）
RE_MD_TABLE_ROW = re.compile(r'^\s*\|.*\|\s*$')
# 单位字符串变体：`单位:元` / `单位 ： 元` / `单位﹕人民币元` 等
RE_UNIT_LINE = re.compile(r'单位\s*[：:﹕]\s*')

# ---------------------------------------------------------------------------
# 3. 全角→半角（仅限数字 / 字母 / 百分号 / 数字上下文括号——不动正文标点）
# ---------------------------------------------------------------------------
_FW_DIGIT = {chr(0xFF10 + i): str(i) for i in range(10)}
_FW_UPPER = {chr(0xFF21 + i): chr(65 + i) for i in range(26)}
_FW_LOWER = {chr(0xFF41 + i): chr(97 + i) for i in range(26)}
_FW_MAP = {}
_FW_MAP.update(_FW_DIGIT)
_FW_MAP.update(_FW_UPPER)
_FW_MAP.update(_FW_LOWER)
_FW_MAP.update({'\uff05': '%', '\uff0e': '.', '\u3000': ' '})
# 全角括号且内部是数字 → 转半角括号（仅数字上下文，避免动正文）
RE_FW_PAREN_NUM = re.compile(
    r'\uff08\s*([0-9\uff10-\uff19,.\uff0e\s]+?)\s*\uff09')


def fw_to_half(text: str) -> str:
    """仅转换数字/字母/百分号的全角形态；正文标点不动。"""
    out = []
    for ch in text:
        out.append(_FW_MAP.get(ch, ch))
    return ''.join(out)


# ---------------------------------------------------------------------------
# 4. 单条文本规范化（复选符号 + 单位 + 数值格式）
# ---------------------------------------------------------------------------
class _Report(object):
    def __init__(self):
        self.counts = {}
        self.samples = {}

    def hit(self, rule: str, sample: str):
        self.counts[rule] = self.counts.get(rule, 0) + 1
        if rule not in self.samples:
            self.samples[rule] = []
        if len(self.samples[rule]) < 5:
            self.samples[rule].append(sample[:120])

    def as_dict(self):
        return {'counts': self.counts, 'samples': self.samples}


ALL_RULES = frozenset([
    'fw', 'checked', 'unchecked', 'unit', 'minus_paren', 'paren_neg',
    'multi_minus', 'num_space', 'cell_split'])


def normalize_text(text: str, report: '_Report | None' = None,
                   rules=None):
    """规范化一段文本，返回 (新文本, 是否有改动)。

    rules: 可选规则集（ALL_RULES 的子集），缺省全开。
    规则间幂等：重复调用不产生二次改动（除 report 计数外）。
    """
    rep = report if report is not None else _Report()
    on = ALL_RULES if rules is None else frozenset(rules)
    orig = text

    # -- 全角→半角（仅数字/字母/百分号；全角括号仅数字上下文）--
    if 'fw' in on:
        text = RE_FW_PAREN_NUM.sub(
            lambda m: '(' + m.group(1) + ')', text)
        half = fw_to_half(text)
        if half != text:
            rep.hit('fullwidth_to_halfwidth', text.strip()[:60])
            text = half

    # -- 复选符号白名单 --
    if 'checked' in on:
        for alias in CHECKED_ALIASES:
            if alias != CHECKED and alias in text:
                rep.hit('check_symbol_checked', alias + ' -> ' + CHECKED)
                text = text.replace(alias, CHECKED)
    if 'unchecked' in on:
        for alias in UNCHECKED_ALIASES:
            if alias != UNCHECKED and alias in text:
                rep.hit('check_symbol_unchecked', alias + ' -> ' + UNCHECKED)
                text = text.replace(alias, UNCHECKED)

    # -- 单位字符串标准化：`单位:元`/`单位 ： 元` → `单位：元` --
    if 'unit' in on:
        def _unit_sub(m):
            if m.group(0) != '单位：':
                rep.hit('unit_string', m.group(0))
            return '单位：'
        text = RE_UNIT_LINE.sub(_unit_sub, text)

    # -- 数值格式 --
    # 4d. 数字内部空格（先于括号负数规则：`（1, 234.56）` 须先收敛成
    #     (1,234.56)，paren_neg 才能在同一遍命中——否则二次应用才生效，
    #     破坏幂等）
    if 'num_space' in on:
        def _num_space_sub(m):
            rep.hit('number_inner_space', m.group(0))
            return re.sub(r'\s+', '', m.group(0))
        text = RE_NUM_INNER_SPACE.sub(_num_space_sub, text)

    # 4a. - ( 687,860,833.00 ) → -687,860,833.00
    if 'minus_paren' in on:
        def _minus_paren_sub(m):
            rep.hit('number_minus_paren', m.group(0))
            return '-' + m.group(1)
        text = RE_MINUS_PAREN.sub(_minus_paren_sub, text)

    # 4b. (99,361.07) → -99,361.07
    #     注意：该规则改变字符集（去括号加负号），会使 vs PDF 原生文本的
    #     trec 下降；接入门禁打分通道时须关闭（见 INTERFACE_REQUESTS.md）
    if 'paren_neg' in on:
        def _paren_neg_sub(m):
            rep.hit('number_paren_negative', m.group(0))
            return '-' + m.group(1)
        text = RE_PAREN_NEG.sub(_paren_neg_sub, text)

    # 4c. --25,093,729.87 → 奇偶归并（偶负号=正）
    if 'multi_minus' in on:
        def _multi_minus_sub(m):
            rep.hit('number_multi_minus', m.group(0))
            signs = len(m.group(1))
            return m.group(2) if signs % 2 == 0 else '-' + m.group(2)
        text = RE_MULTI_MINUS.sub(_multi_minus_sub, text)

    return text, text != orig


def normalize_table_row(line: str, rep: '_Report'):
    """MD 表行级修复：括号负数被切到相邻两格的合并。

    仅处理严格形态：A 格 = [负号]+左括号（别无他物），B 格 = 数字+右括号，
    合并后 A 格写入 `-数字`、B 格置空。零阈值、模式驱动。
    """
    if not RE_MD_TABLE_ROW.match(line):
        return line
    cells = line.split('|')
    # cells[0]/cells[-1] 为行首尾空段；逐对检查相邻内层格
    i = 1
    merged = 0
    while i < len(cells) - 2:
        a, b = cells[i], cells[i + 1]
        done = False
        # 形态1：A=[负号+]左括号，B=数字+右括号
        ma = RE_CELL_SPLIT_PAREN_A.match(a)
        mb = RE_CELL_SPLIT_PAREN_B.match(b)
        if ma and mb:
            sign = '-' if ma.group(1).strip().startswith('-') else ''
            rep.hit('cell_split_paren_merged', a.strip() + ' | ' + b.strip())
            cells[i] = ' ' + sign + mb.group(1) + ' '
            cells[i + 1] = ' '
            merged += 1
            done = True
        # 形态2：A=裸负号，B=括号负数
        if not done:
            ma2 = RE_CELL_SPLIT_MINUS_A.match(a)
            mb2 = RE_CELL_SPLIT_MINUS_B.match(b)
            if ma2 and mb2:
                rep.hit('cell_split_minus_merged', a.strip() + ' | ' + b.strip())
                cells[i] = ' -' + mb2.group(1) + ' '
                cells[i + 1] = ' '
                merged += 1
                done = True
        i += 2 if done else 1
    if merged:
        return '|'.join(cells)
    return line


# ---------------------------------------------------------------------------
# 5. 整份 Markdown 规范化
# ---------------------------------------------------------------------------
def normalize_markdown(md_text: str, rules=None):
    """对整份产物 MD 做规范化，返回 (新文本, report dict)。

    rules: 规则集（ALL_RULES 子集），缺省全开。
    """
    rep = _Report()
    lines = md_text.split('\n')
    out_lines = []
    n_changed_lines = 0
    in_code = False
    for line in lines:
        if line.lstrip().startswith('```'):
            in_code = not in_code
            out_lines.append(line)
            continue
        if in_code:
            out_lines.append(line)
            continue
        # 行级跨格切分合并先做（避免文本级括号负数转换造成双负号）
        if rules is None or 'cell_split' in rules:
            line = normalize_table_row(line, rep)
        new_line, changed = normalize_text(line, rep, rules)
        if changed:
            n_changed_lines += 1
        out_lines.append(new_line)
    report = rep.as_dict()
    report['changed_lines'] = n_changed_lines
    report['total_lines'] = len(lines)
    return '\n'.join(out_lines), report


def normalize_json_product(data: dict, rules=None):
    """对管道 JSON 产物（texts + tables grid cells）逐格规范化。

    rules: 规则集（ALL_RULES 子集），缺省全开。
    返回 (新 data(深拷贝), report dict)。原 data 不被修改。
    """
    import copy
    rep = _Report()
    new = copy.deepcopy(data)

    for item in new.get('texts', []) or []:
        if isinstance(item, dict) and 'text' in item:
            item['text'], _ = normalize_text(item['text'], rep, rules)

    for tbl in new.get('tables', []) or []:
        grid = ((tbl.get('data') or {}).get('grid'))
        if not grid:
            continue
        for row in grid:
            for cell in row:
                if isinstance(cell, dict) and cell.get('text'):
                    cell['text'], _ = normalize_text(cell['text'], rep, rules)

    report = rep.as_dict()
    report['texts'] = len(new.get('texts') or [])
    report['tables'] = len(new.get('tables') or [])
    return new, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description='C7 抽取后处理规范化（复选符号/单位/数值格式）')
    ap.add_argument('--md', help='输入 Markdown 产物路径')
    ap.add_argument('--json', help='输入 JSON 产物路径')
    ap.add_argument('--stdin', action='store_true', help='从 stdin 读 MD')
    ap.add_argument('--out', help='规范化结果输出路径（缺省 stdout）')
    ap.add_argument('--report', help='改动报告 JSON 输出路径')
    ap.add_argument('--diff', action='store_true',
                    help='报告里附带逐行前后对比（前 200 处）')
    args = ap.parse_args(argv)

    if not (args.md or args.json or args.stdin):
        ap.error('需要 --md / --json / --stdin 之一')

    diff_pairs = []

    if args.stdin:
        md = sys.stdin.read()
        new, report = normalize_markdown(md)
        if args.diff:
            _collect_diff(md.split('\n'), new.split('\n'), diff_pairs)
        result = new
    elif args.md:
        with open(args.md, encoding='utf-8') as f:
            md = f.read()
        new, report = normalize_markdown(md)
        if args.diff:
            _collect_diff(md.split('\n'), new.split('\n'), diff_pairs)
        result = new
    else:
        with open(args.json, encoding='utf-8') as f:
            data = json.load(f)
        new, report = normalize_json_product(data)
        result = json.dumps(new, ensure_ascii=False)

    if diff_pairs:
        report['diff'] = diff_pairs[:200]

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(result)
    else:
        sys.stdout.write(result)

    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    # 摘要恒写 stderr，避免污染管道输出
    sys.stderr.write('[normalize] %s\n' % json.dumps(
        report.get('counts', {}), ensure_ascii=False))
    return 0


def _collect_diff(old_lines, new_lines, out):
    for i, (a, b) in enumerate(zip(old_lines, new_lines)):
        if a != b:
            out.append({'line': i + 1, 'before': a[:200], 'after': b[:200]})


if __name__ == '__main__':
    sys.exit(main())
