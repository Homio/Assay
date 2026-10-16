"""
P19c（track_p5）：精确内容去重（SHA-256）—— v2 页粒度管道的生产去重层。

设计（对齐 P19C 交付四要素）：
  1. 内容身份 = 文件字节的 SHA-256（流式 1MiB 读，文件落盘稳定后计算；
     本管道输入为静态语料目录，无"边写边算"窗口）。
  2. 同 hash 只算一次：调度器预扫描阶段一次性分组，每组首个（排序序）
     为主本，其余为影子本；影子本不产生任何 calib/text/detect/build
     任务，主本完成后按原始提取结果各自装配输出（保留各自的公司/
     日期/来源/公告 ID 身份与输出目录）。单调度器进程 + 分组静态化，
     并发击穿在构造上不可能（仍以互斥锁保护落盘写入）。
  3. 缓存纪律：只缓存完整成功结果；条目经 tmp+fsync+rename 原子提交，
     失败/半成品绝不落盘；任何读取异常一律视为 miss（宁可重算，绝不
     复用可疑载荷）；载荷自校验（content_hash 回读比对）。
  4. 缓存键 = 内容 hash + 管道签名。管道签名 = 代码版本（code_rev）
     + 有效配置（pipeline_config.yaml 规范化全文）+ 输出格式签名
     （assay.output.EXPORT_FORMAT_SIG）——任一变化即整域失效，
     只会多算、绝不错复用。

零硬编码：本模块不含任何公司名/文档名/语料特有模式；全部阈值与路径
来自 pipeline_config.yaml（dedup 段）。
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

# 流式读取块大小（1MiB：NVMe 顺序读 + SHA-NI 的甜点位）
_CHUNK = 1 << 20


def content_sha256(path: str) -> str:
    """文件完整字节的 SHA-256（流式；大文件常驻内存恒为 chunk 大小）。"""
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(_CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_files(paths: List[str], workers: int = 8) -> Dict[str, str]:
    """批量并行哈希。path -> sha256；结果与调用序一致（map 保序）。

    单文件哈希失败（文件消失/不可读）直接抛出：预扫描阶段语料完整性
    出问题应中止运行，而不是静默把坏书当成"算过了"。
    """
    if workers <= 1 or len(paths) <= 1:
        return {p: content_sha256(p) for p in paths}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return {p: h for p, h in zip(paths, ex.map(content_sha256, paths))}


def group_by_hash(hashes: Dict[str, str]) -> Dict[str, List[str]]:
    """content_hash -> [路径...]（保持输入插入序 = 主本选择序）。"""
    groups: Dict[str, List[str]] = {}
    for p, h in hashes.items():
        groups.setdefault(h, []).append(p)
    return groups


def pipeline_signature(cfg: dict) -> str:
    """缓存命名空间签名：实现版本 + 有效配置 + 输出格式签名。

    任一分量变化 → 签名变化 → 旧条目全部 miss（fail-toward-recompute）。
    """
    import yaml

    from .output import EXPORT_FORMAT_SIG, code_rev

    canon = yaml.safe_dump(cfg or {}, sort_keys=True, allow_unicode=True)
    h = hashlib.sha256()
    h.update(code_rev().encode("utf-8"))
    h.update(b"\x00")
    h.update(canon.encode("utf-8"))
    h.update(b"\x00")
    h.update(EXPORT_FORMAT_SIG.encode("utf-8"))
    return h.hexdigest()


class DedupStore:
    """跨运行持久载荷缓存（调度器进程独占访问）。

    布局：root/<sig[:8]>/<hash[:2]>/<hash>.pkl —— 签名分域 + hash 分片，
    单目录条目数有界。put 原子：同目录 tmp 写入 → fsync → rename，
    崩溃只留 .tmp 孤儿（下次 put 同 key 时被 rename 覆盖，get 永远
    读不到半成品）。get 的任何异常（缺文件/坏 pickle/自校验不过）一律
    miss；坏条目顺手清掉（宁可重算）。
    """

    def __init__(self, root: str, signature: str):
        self.root = Path(root)
        self.sig = signature
        self._lock = threading.Lock()
        # 计账（调度器汇总进运行报告）
        self.n_get = 0
        self.n_hit = 0
        self.n_put = 0
        self.n_err = 0
        self.put_bytes = 0

    def _entry(self, content_hash: str) -> Path:
        return self.root / self.sig[:8] / content_hash[:2] / f"{content_hash}.pkl"

    def get(self, content_hash: str) -> Optional[dict]:
        """命中返回载荷 dict；未命中/任何异常返回 None（绝不抛出）。"""
        f = self._entry(content_hash)
        self.n_get += 1
        try:
            with f.open("rb") as fh:
                payload = pickle.load(fh)
        except FileNotFoundError:
            return None
        except Exception:  # noqa: BLE001 — 坏条目按 miss 处理并清除
            self.n_err += 1
            try:
                f.unlink()
            except OSError:
                pass
            return None
        # 载荷自校验：键位与内容哈希必须对得上——防错位/手改
        if (not isinstance(payload, dict)
                or payload.get("content_hash") != content_hash):
            self.n_err += 1
            try:
                f.unlink()
            except OSError:
                pass
            return None
        self.n_hit += 1
        return payload

    def put(self, content_hash: str, payload: dict) -> bool:
        """原子写入一条完整成功载荷。任何失败只记账、绝不抛出——
        缓存写失败不允许影响提取主流程（下次运行重算即可）。"""
        f = self._entry(content_hash)
        tmp = None
        with self._lock:
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=str(f.parent), suffix=".tmp")
                with os.fdopen(fd, "wb") as fh:
                    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, f)
                tmp = None
                self.n_put += 1
                self.put_bytes += f.stat().st_size
                return True
            except Exception:  # noqa: BLE001
                self.n_err += 1
                return False
            finally:
                if tmp is not None:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
