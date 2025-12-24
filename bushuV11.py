import streamlit as st
import PIL.Image
from google import genai
import os
import hashlib
import random
import re
from dataclasses import dataclass
from typing import Optional
from borax.calendars.lunardate import LunarDate
from datetime import date, datetime, time

from ailife_config import get_genai_client, get_google_api_key, pil_image_to_part
from PIL import ImageFilter, ImageOps, ImageStat


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _image_quality_metrics(image: PIL.Image.Image) -> dict:
    width, height = image.size
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_var = float(ImageStat.Stat(edges).var[0])
    return {"width": int(width), "height": int(height), "edge_var": edge_var}


def _clarity_score(metrics: dict) -> int:
    edge_var = float(metrics.get("edge_var") or 0.0)
    edge_norm = _clamp((edge_var - 50.0) / (220.0 - 50.0), 0.0, 1.0)

    width = int(metrics.get("width") or 0)
    height = int(metrics.get("height") or 0)
    min_side = float(min(width, height))
    res_norm = _clamp((min_side - 700.0) / (1600.0 - 700.0), 0.0, 1.0)

    score = (0.65 * edge_norm + 0.35 * res_norm) * 100.0
    return int(round(_clamp(score, 0.0, 100.0)))


def _clarity_grade(score: int) -> str:
    if score >= 85:
        return "A（高）"
    if score >= 70:
        return "B（中高）"
    if score >= 55:
        return "C（中）"
    return "D（偏低）"

def _life_open_close_keyword(life_rows: list[dict]) -> str | None:
    adult_rows = [r for r in (life_rows or []) if int(r.get("age", 0)) >= 18]
    if len(adult_rows) < 8:
        return None

    early = [r for r in adult_rows if 18 <= int(r.get("age", 0)) <= 30]
    late = [r for r in adult_rows if int(r.get("age", 0)) >= 31]
    if len(early) < 3 or len(late) < 3:
        return None

    early_avg = sum(float(r.get("close", 0.0)) for r in early) / float(len(early))
    late_avg = sum(float(r.get("close", 0.0)) for r in late) / float(len(late))
    trend = late_avg - early_avg

    closes = [float(r.get("close", 0.0)) for r in adult_rows]
    diffs = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    volatility = sum(abs(d) for d in diffs) / float(len(diffs) or 1)

    start_high = early_avg >= 55.0
    high_vol = volatility >= 8.0

    if start_high:
        if high_vol:
            return "高开疯走"
        if trend <= -4.0:
            return "高开低走"
        return "高开高走"

    if high_vol or trend < 0.0:
        return "低开疯走"
    return "低开高走"


def _enhance_for_lines(image: PIL.Image.Image) -> tuple[PIL.Image.Image, PIL.Image.Image]:
    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray)
    sharp = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=175, threshold=3))
    edges = sharp.filter(ImageFilter.FIND_EDGES)
    return sharp.convert("RGB"), edges.convert("RGB")


def _extract_text_from_genai_response(response) -> str | None:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        chunks: list[str] = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text:
                chunks.append(part_text)
        if chunks:
            joined = "".join(chunks).strip()
            if joined:
                return joined
    return None


def _stable_seed(*parts: object) -> int:
    payload = "|".join("" if p is None else str(p) for p in parts).encode("utf-8", errors="ignore")
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:4], "big", signed=False) & 0x7FFFFFFF  # int32
    return seed or 1


def _extract_future_keywords(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"未来三年关键词[：:]\s*(.+)", text)
    if not match:
        return None
    keywords = match.group(1).strip()
    keywords = keywords.strip("。.!！")
    return keywords or None


def _strip_footer_from_report(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    filtered: list[str] = []
    for line in lines:
        if "未来三年关键词" in line:
            continue
        if re.match(r"^\s*#{1,6}\s*收尾", line):
            continue
        filtered.append(line)

    while filtered and not filtered[-1].strip():
        filtered.pop()
    return "\n".join(filtered)


def _birth_datetime(local_date: date, local_time: time) -> datetime:
    return datetime.combine(local_date, local_time)


TIANGAN_WUXING = {
    "甲": "木",
    "乙": "木",
    "丙": "火",
    "丁": "火",
    "戊": "土",
    "己": "土",
    "庚": "金",
    "辛": "金",
    "壬": "水",
    "癸": "水",
}

DIZHI_WUXING = {
    "子": "水",
    "丑": "土",
    "寅": "木",
    "卯": "木",
    "辰": "土",
    "巳": "火",
    "午": "火",
    "未": "土",
    "申": "金",
    "酉": "金",
    "戌": "土",
    "亥": "水",
}

SHENG = {
    "木": "火",
    "火": "土",
    "土": "金",
    "金": "水",
    "水": "木",
}

KE = {
    "木": "土",
    "土": "水",
    "水": "火",
    "火": "金",
    "金": "木",
}

ZHI_CHONG = {
    "子": "午",
    "午": "子",
    "丑": "未",
    "未": "丑",
    "寅": "申",
    "申": "寅",
    "卯": "酉",
    "酉": "卯",
    "辰": "戌",
    "戌": "辰",
    "巳": "亥",
    "亥": "巳",
}


TIANYI_BY_DAYGAN = {
    "甲": ["丑", "未"],
    "戊": ["丑", "未"],
    "庚": ["丑", "未"],
    "乙": ["子", "申"],
    "己": ["子", "申"],
    "丙": ["亥", "酉"],
    "丁": ["亥", "酉"],
    "壬": ["卯", "巳"],
    "癸": ["卯", "巳"],
    "辛": ["寅", "午"],
}


PEACH_BY_GROUP = {
    frozenset(["申", "子", "辰"]): "酉",
    frozenset(["寅", "午", "戌"]): "卯",
    frozenset(["亥", "卯", "未"]): "子",
    frozenset(["巳", "酉", "丑"]): "午",
}

YIMA_BY_GROUP = {
    frozenset(["申", "子", "辰"]): "寅",
    frozenset(["寅", "午", "戌"]): "申",
    frozenset(["巳", "酉", "丑"]): "亥",
    frozenset(["亥", "卯", "未"]): "巳",
}

HUAGAI_BY_GROUP = {
    frozenset(["申", "子", "辰"]): "辰",
    frozenset(["寅", "午", "戌"]): "戌",
    frozenset(["巳", "酉", "丑"]): "丑",
    frozenset(["亥", "卯", "未"]): "未",
}


def _find_group_mapping(branch: str, mapping: dict) -> Optional[str]:
    for group, value in mapping.items():
        if branch in group:
            return value
    return None


def _bazi_markers(day_gan: str, year_gz: str, month_gz: str, day_gz: str, hour_gz: str) -> dict:
    pillars = {"年柱": year_gz, "月柱": month_gz, "日柱": day_gz, "时柱": hour_gz}
    branches = {k: _split_gz(v)[1] for k, v in pillars.items()}

    tianyi_targets = TIANYI_BY_DAYGAN.get(day_gan, [])
    tianyi_pos = [p for p, z in branches.items() if z in tianyi_targets and z]

    day_branch = branches.get("日柱", "")
    peach = _find_group_mapping(day_branch, PEACH_BY_GROUP) or ""
    peach_pos = [p for p, z in branches.items() if z == peach and z]

    yima = _find_group_mapping(day_branch, YIMA_BY_GROUP) or ""
    yima_pos = [p for p, z in branches.items() if z == yima and z]

    huagai = _find_group_mapping(day_branch, HUAGAI_BY_GROUP) or ""
    huagai_pos = [p for p, z in branches.items() if z == huagai and z]

    return {
        "天乙贵人": {"targets": tianyi_targets, "positions": tianyi_pos},
        "桃花": {"target": peach, "positions": peach_pos},
        "驿马": {"target": yima, "positions": yima_pos},
        "华盖": {"target": huagai, "positions": huagai_pos},
    }


def _element_relation_score(src: Optional[str], dst: Optional[str], weight: float) -> float:
    if not src or not dst:
        return 0.0
    if src == dst:
        return 1.0 * weight
    if SHENG.get(src) == dst:
        return 0.7 * weight
    if SHENG.get(dst) == src:
        return -0.45 * weight
    if KE.get(src) == dst:
        return 0.25 * weight
    if KE.get(dst) == src:
        return -1.0 * weight
    return 0.0


def _support_score_for_day_master(
    src: Optional[str],
    day_elem: Optional[str],
    weight: float,
    *,
    resource_buffer: float = 0.0,
) -> float:
    """
    粗粒度“经典派”取向：把五行关系映射成对日主的支持/消耗/压力。
    - 比劫（同我）：+1.0
    - 印（生我）：+0.85
    - 食伤（我生）：-0.20（消耗但可用）
    - 财（我克）：-0.10（消耗但可转化）
    - 官杀（克我）：-0.60（压力/规则/冲突）
    """
    if not src or not day_elem:
        return 0.0
    if src == day_elem:
        return 1.0 * weight
    if SHENG.get(src) == day_elem:
        return 0.85 * weight
    if SHENG.get(day_elem) == src:
        return -0.20 * weight
    if KE.get(day_elem) == src:
        return -0.10 * weight
    if KE.get(src) == day_elem:
        # “官杀”不一定坏：有印（资源）时更像“压力=成事的框架”，因此给一个缓冲项
        buf = float(_clamp(resource_buffer, 0.0, 1.0))
        return (-0.60 + 0.35 * buf) * weight
    return 0.0


def _resource_element_for(day_elem: Optional[str]) -> Optional[str]:
    if not day_elem:
        return None
    for k, v in SHENG.items():
        if v == day_elem:
            return k
    return None


def _output_element_for(day_elem: Optional[str]) -> Optional[str]:
    if not day_elem:
        return None
    return SHENG.get(day_elem)


def _wealth_element_for(day_elem: Optional[str]) -> Optional[str]:
    if not day_elem:
        return None
    return KE.get(day_elem)


def _kill_element_for(day_elem: Optional[str]) -> Optional[str]:
    if not day_elem:
        return None
    for k, v in KE.items():
        if v == day_elem:
            return k
    return None


def _count_elem_hits(elems: list[Optional[str]], target: Optional[str]) -> int:
    if not target:
        return 0
    return sum(1 for e in elems if e == target)


def _dimension_status_text(
    *,
    dimension: str,
    index: int,
    vol: float,
    conflict_tags: list[str],
) -> str:
    v = float(vol)
    i = int(index)
    has_turn = any(t in conflict_tags for t in ("极端转折", "内耗期", "困兽之斗", "情场劫财"))

    if dimension == "wealth":
        if "情场劫财" in conflict_tags:
            return "感情/人情牵扯进账与花销，钱容易被关系带节奏。"
        if i >= 70 and v >= 12:
            return "进账窗口明显，但伴随大起大落，守财比赚钱更难。"
        if 45 <= i <= 60 and (v >= 12 or has_turn):
            return "财来财去：账面不差，但会有反复与临时支出。"
        if i < 45 and v >= 10:
            return "破财/支出波动期：更像“先花钱后补救”，要设硬止损。"
        if i >= 65:
            return "资金面更顺，适合做长期规划与稳健累积。"
        if i <= 40:
            return "现金流偏紧，先保底盘与节奏，不宜冒进。"
        return "财务起伏不大，适合稳扎稳打。"

    if dimension == "career":
        if "困兽之斗" in conflict_tags:
            return "压力与表达欲同时拉满：想突围、也容易把自己逼到极限。"
        if 45 <= i <= 60 and (v >= 12 or has_turn):
            return "内耗期：方向在变、标准在变，做得多但不一定被看见。"
        if i >= 70 and v >= 12:
            return "冲刺窗口：项目/机会密集，能上台阶但需要强体力与取舍。"
        if i >= 65:
            return "更容易拿到平台与成果，适合要位置/要结果的打法。"
        if i <= 40:
            return "阻力偏大：优先修复基础能力与协作关系，再谈扩张。"
        return "推进节奏一般，适合打磨方法论与长期积累。"

    # romance
    if "情场劫财" in conflict_tags:
        return "吸引力很强但易起冲突：情绪与现实账本会互相拉扯。"
    if i >= 70 and v >= 12:
        return "热度很高、变化也大：容易一把上头，也容易快速降温。"
    if 45 <= i <= 60 and (v >= 12 or has_turn):
        return "暧昧与纠结并存：想要确定，又怕被绑定。"
    if i >= 65:
        return "更容易被看见/被喜欢，适合主动表达与建立边界。"
    if i <= 40:
        return "关系能量偏低：先把自己安顿好，关系才会变顺。"
    return "关系温度中等，重在沟通方式与节奏匹配。"


def _clamp_score(value: float) -> int:
    return int(round(_clamp(value, 0.0, 100.0)))


@dataclass(frozen=True)
class BaziPro:
    four_pillars: str
    day_master: str
    day_gz: str
    year_gz: str
    month_gz: str
    hour_gz: str
    dayun: list  # lunar_python objects
    birth_dt: datetime
    pillar_details: list[dict]
    markers: dict


def _split_gz(gz: str) -> tuple[str, str]:
    gz = (gz or "").strip()
    if len(gz) >= 2:
        return gz[0], gz[1]
    if len(gz) == 1:
        return gz[0], ""
    return "", ""


def _dayun_gz_for_year(dayun_list: list, year: int) -> Optional[str]:
    for dy in dayun_list or []:
        try:
            start_year = int(dy.getStartYear())
            end_year = int(dy.getEndYear())
            if start_year <= year < end_year:
                return str(dy.getGanZhi())
        except Exception:
            continue
    return None


def _dayun_transition_years(dayun_list: list) -> list[int]:
    years: list[int] = []
    for dy in dayun_list or []:
        try:
            years.append(int(dy.getStartYear()))
        except Exception:
            continue
    return sorted(set(years))


def _luck_index_for_year(
    *,
    day_gz: str,
    day_master: str,
    year_gz: str,
    dayun_gz: Optional[str],
    resource_buffer: float = 0.0,
) -> tuple[int, float, dict]:
    day_stem, day_branch = _split_gz(day_gz)
    year_stem, year_branch = _split_gz(year_gz)
    dayun_stem, dayun_branch = _split_gz(dayun_gz or "")

    day_elem = TIANGAN_WUXING.get(day_master)
    year_stem_elem = TIANGAN_WUXING.get(year_stem)
    year_branch_elem = DIZHI_WUXING.get(year_branch)
    dayun_stem_elem = TIANGAN_WUXING.get(dayun_stem)
    dayun_branch_elem = DIZHI_WUXING.get(dayun_branch)

    score = 50.0
    s1 = _support_score_for_day_master(year_stem_elem, day_elem, 12.0, resource_buffer=resource_buffer)
    s2 = _support_score_for_day_master(year_branch_elem, day_elem, 7.0, resource_buffer=resource_buffer)
    s3 = _support_score_for_day_master(dayun_stem_elem, day_elem, 9.0, resource_buffer=resource_buffer)
    s4 = _support_score_for_day_master(dayun_branch_elem, day_elem, 4.0, resource_buffer=resource_buffer)
    score += s1 + s2 + s3 + s4

    chong = ZHI_CHONG.get(day_branch) == year_branch and day_branch and year_branch
    if chong:
        # 冲=动：更像“变动/事件密度”，不等于坏；主要体现在波动上
        score -= 1.5

    if year_stem and day_stem and year_stem == day_stem:
        score += 2.0

    score_i = _clamp_score(score)

    # 张力：同一年里“支持与压力”越强，越容易呈现出转折与强波动，而不是平均脸
    tension = (abs(s1) + abs(s2) + abs(s3) + abs(s4)) / float(12.0 + 7.0 + 9.0 + 4.0)
    base_vol = 3.0 + float(_clamp(tension, 0.0, 1.4)) * 7.0 + (6.0 if chong else 0.0)

    # 随机扰动（极值规则）：特定日主遇到特定天干时，波动放大
    extreme_mult = 1.0
    extreme_map = {
        "壬": {"丙": 1.5},
        "癸": {"丁": 1.4},
    }
    try:
        extreme_mult = float(extreme_map.get(day_master, {}).get(year_stem, 1.0))
    except Exception:
        extreme_mult = 1.0

    volatility = float(_clamp(base_vol * extreme_mult, 2.0, 26.0))

    conflict_tags: list[str] = []
    if volatility >= 13.0 and 45 <= score_i <= 60:
        conflict_tags.append("极端转折")
    if volatility >= 13.0 and score_i < 45:
        conflict_tags.append("内耗期")
    if chong and score_i >= 55:
        conflict_tags.append("动中有利")
    if extreme_mult >= 1.35:
        conflict_tags.append("极值放大")
    meta = {
        "day_elem": day_elem,
        "year_gz": year_gz,
        "dayun_gz": dayun_gz,
        "chong": bool(chong),
        "conflict_tags": conflict_tags,
    }
    return score_i, volatility, meta


def _build_life_kline(
    *,
    bazi: BaziPro,
    max_age: int,
) -> dict:
    birth_year = int(bazi.birth_dt.year)
    years = [birth_year + age for age in range(0, max_age + 1)]

    elem_counts = {"木": 0, "火": 0, "土": 0, "金": 0, "水": 0}
    for row in bazi.pillar_details or []:
        for k in ("干五行", "支五行"):
            v = row.get(k)
            if v in elem_counts:
                elem_counts[v] += 1

    day_elem = TIANGAN_WUXING.get(bazi.day_master)
    resource_elem = None
    if day_elem:
        for k, v in SHENG.items():
            if v == day_elem:
                resource_elem = k
                break
    resource_count = int(elem_counts.get(resource_elem or "", 0))
    resource_buffer = _clamp((resource_count - 1) / 3.0, 0.0, 1.0)

    rows: list[dict] = []
    prev_close: Optional[float] = None
    change_abs: list[tuple[int, float]] = []
    dayun_transitions = _dayun_transition_years(bazi.dayun)

    for age, year in enumerate(years):
        year_gz = LunarDate.from_solar_date(year, 6, 1).gz_year
        dayun_gz = _dayun_gz_for_year(bazi.dayun, year)
        close_i, base_vol, meta = _luck_index_for_year(
            day_gz=bazi.day_gz,
            day_master=bazi.day_master,
            year_gz=str(year_gz),
            dayun_gz=dayun_gz,
            resource_buffer=resource_buffer,
        )

        close = float(close_i)
        open_ = close if prev_close is None else float(prev_close)
        delta = close - open_
        vol = base_vol + abs(delta) * 0.35
        if year in dayun_transitions:
            vol += 2.0

        high = _clamp(max(open_, close) + vol, 0.0, 100.0)
        low = _clamp(min(open_, close) - vol, 0.0, 100.0)

        label = f"{age}岁 ({year})"
        rows.append(
            {
                "x": label,
                "age": age,
                "year": year,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "year_gz": str(year_gz),
                "dayun_gz": dayun_gz or "",
                "chong": meta.get("chong", False),
                "conflict_tags": meta.get("conflict_tags", []) or [],
                "is_dayun_transition": year in dayun_transitions,
            }
        )

        if prev_close is not None:
            change_abs.append((year, abs(close - float(prev_close))))

        prev_close = close

    top_turns = sorted(change_abs, key=lambda x: x[1], reverse=True)[:8]

    macro_trend: dict = {}
    macro_trend_text = ""
    adult = [r for r in rows if int(r.get("age", 0)) >= 18]
    if len(adult) >= 6:
        best = None
        in_run = False
        run_start = 0
        thr = 0.6

        def _finalize(start_i: int, end_i: int) -> None:
            nonlocal best
            length = end_i - start_i + 1
            if length < 3:
                return
            drop = float(adult[start_i]["close"]) - float(adult[end_i]["close"])
            score = drop * 1.2 + float(length) * 0.8
            cand = (score, start_i, end_i, drop, length)
            if best is None or cand[0] > best[0]:
                best = cand

        for i in range(1, len(adult)):
            prev = float(adult[i - 1]["close"])
            cur = float(adult[i]["close"])
            is_down = cur < prev - thr
            if is_down:
                if not in_run:
                    in_run = True
                    run_start = i - 1
                continue
            if in_run:
                _finalize(run_start, i - 1)
                in_run = False

        if in_run:
            _finalize(run_start, len(adult) - 1)

        if best is not None:
            _, s, e, drop, length = best
            start_year = int(adult[s]["year"])
            end_year = int(adult[e]["year"])
            macro_trend["downtrend"] = {
                "start_year": start_year,
                "end_year": end_year,
                "years": int(length),
                "drop": float(round(drop, 2)),
            }
            macro_trend_text = f"{start_year}-{end_year} 连续阴跌（蛰伏期）：更适合稳住基本盘、修复系统，不建议在中段盲目冲刺。"

        div = None
        for i in range(2, len(adult)):
            c0 = float(adult[i - 2]["close"])
            c1 = float(adult[i - 1]["close"])
            c2 = float(adult[i]["close"])
            d1 = float(adult[i - 1]["close"]) - float(adult[i - 1]["open"])
            d2 = float(adult[i]["close"]) - float(adult[i]["open"])
            v1 = float(adult[i - 1]["high"]) - float(adult[i - 1]["low"])
            v2 = float(adult[i]["high"]) - float(adult[i]["low"])
            if c2 < min(c0, c1) - 1.0 and d2 > d1 and v2 < v1:
                div = {
                    "year": int(adult[i]["year"]),
                    "hint": "底背离候选：下跌放缓，可能开始筑底",
                }
        if div:
            macro_trend["divergence"] = div
            if macro_trend_text:
                macro_trend_text += f"；{div['year']} 附近 {div['hint']}。"
            else:
                macro_trend_text = f"{div['year']} 附近 {div['hint']}。"

    return {
        "rows": rows,
        "birth_year": birth_year,
        "dayun_transitions": dayun_transitions,
        "top_turns": top_turns,
        "macro_trend": macro_trend,
        "macro_trend_text": macro_trend_text,
    }


def _missing_element_talents(missing: list[str]) -> str:
    talent_map = {
        "木": "更重“生长/变化/开创”，不喜欢被固定路径绑住，适合走学习曲线和长期进化。",
        "火": "更偏冷静和延迟满足，不靠情绪热度推进，适合做耐心型、系统型的事。",
        "土": "更不容易被安稳与规则驯化，灵感和迁移力更强，但也要自建秩序与落地机制。",
        "金": "更少被外在评价牵着走，表达更柔，适合把锋芒变成方法论，而非硬碰硬。",
        "水": "更少靠直觉漂移，更偏确定性与执行；但要给自己留一点想象空间与缓冲。",
    }
    lines = []
    for e in missing:
        t = talent_map.get(e)
        if t:
            lines.append(f"- 缺{e}：{t}")
    return "\n".join(lines) if lines else "（未计算/无明显缺失）"


def _yongshen_profile(bazi: BaziPro) -> dict:
    day_elem = TIANGAN_WUXING.get(bazi.day_master)
    month_stem, month_zhi = _split_gz(bazi.month_gz)
    month_elem = DIZHI_WUXING.get(month_zhi) or TIANGAN_WUXING.get(month_stem)

    resource_elem = _resource_element_for(day_elem)
    output_elem = _output_element_for(day_elem)
    wealth_elem = _wealth_element_for(day_elem)
    kill_elem = _kill_element_for(day_elem)

    elem_counts = {"木": 0, "火": 0, "土": 0, "金": 0, "水": 0}
    for row in bazi.pillar_details or []:
        for k in ("干五行", "支五行"):
            v = row.get(k)
            if v in elem_counts:
                elem_counts[v] += 1

    missing = [k for k, v in elem_counts.items() if int(v) == 0]

    score = 0.0
    if month_elem and day_elem:
        if month_elem == day_elem:
            score += 1.4
        if resource_elem and month_elem == resource_elem:
            score += 1.0
        if output_elem and month_elem == output_elem:
            score -= 0.8
        if kill_elem and month_elem == kill_elem:
            score -= 1.0

    if resource_elem:
        score += (float(elem_counts.get(resource_elem, 0)) - 1.0) * 0.25
    if day_elem:
        score += (float(elem_counts.get(day_elem, 0)) - 1.0) * 0.15

    if score <= -1.0:
        strength = "偏弱"
        favored = [resource_elem, day_elem]
        avoid = [kill_elem, output_elem, wealth_elem]
    elif score >= 1.0:
        strength = "偏强"
        favored = [wealth_elem, output_elem, kill_elem]
        avoid = [resource_elem, day_elem]
    else:
        strength = "中和"
        favored = [wealth_elem, output_elem]
        avoid = [kill_elem]

    def _uniq(xs: list[Optional[str]]) -> list[str]:
        out = []
        for x in xs:
            if x and x not in out:
                out.append(x)
        return out

    favored_u = _uniq(favored)
    avoid_u = _uniq(avoid)

    summary = (
        f"- 日主：{bazi.day_master}（{day_elem or '未知'}）｜月令五行：{month_elem or '未知'}｜强弱：{strength}\n"
        f"- 用神倾向（模型化）：{('、'.join(favored_u) if favored_u else '未计算')}｜忌神倾向（模型化）：{('、'.join(avoid_u) if avoid_u else '未计算')}\n"
        f"- 五行缺失：{('、'.join(missing) if missing else '无明显缺失')}"
    )

    return {
        "strength": strength,
        "favored": favored_u,
        "avoid": avoid_u,
        "missing": missing,
        "summary": summary,
        "missing_talents": _missing_element_talents(missing),
    }


def _breakout_anchors_text(*, bazi: Optional[BaziPro], seed: Optional[int]) -> str:
    if not bazi:
        return "（未计算）"

    nayin_imagery = {
        "海中金": "深海矿脉：外冷内坚，价值要在压力里被锻出来。",
        "炉中火": "炉火：先受热、再成形，越是被限制越容易出成果。",
        "大林木": "原始森林：慢但强，靠长期迭代与根系积累。",
        "路旁土": "路基土：看似普通，但承重能力决定上限。",
        "剑锋金": "刀刃：锐利但要控方向，否则先伤自己。",
        "山头火": "山火：扩张很快，成败都在边界管理。",
        "涧下水": "涧水：路线多变，但总能找到出口。",
        "城头土": "城墙：规则感强，适合搭结构、做体系。",
        "白蜡金": "白蜡：可塑性强，成形需要温度与耐心。",
        "杨柳木": "柳木：柔中带韧，适合借势而不硬扛。",
        "泉中水": "泉眼：稳定供给，关键是别被杂质堵住。",
        "屋上土": "屋顶：擅长“收尾与定型”，不适合永远在开荒。",
        "霹雳火": "雷火：爆发式推进，代价是精力与关系磨损。",
        "松柏木": "松柏：慢热但抗压，越到后期越稳。",
        "长流水": "大江：势能来自路线与惯性，别逆势逞强。",
        "砂中金": "砂金：看起来散，聚拢后才显价值。",
        "山下火": "地火：藏在底层的野心，需要正确触发条件。",
        "平地木": "平原之木：更适合规模化与复制，而非孤勇。",
        "壁上土": "墙面：边界清晰，擅长隔离噪音、聚焦目标。",
        "金箔金": "金箔：要靠工艺与包装，粗暴推进反而掉价。",
        "覆灯火": "灯火：照亮一隅，靠持续稳定影响力取胜。",
        "天河水": "天河：想象力强，但要落到具体系统里。",
        "大驿土": "驿站：人生靠迁移与平台转换拿结果。",
        "钗钏金": "饰品：价值来自“被看见”，但别为认可透支。",
        "桑柘木": "桑柘：能在限制里生长，适合难局破题。",
        "大溪水": "溪谷：曲折但不断，适合从复杂中抽象方法。",
        "沙中土": "沙土：灵活但不稳，必须先建秩序再谈扩张。",
        "天上火": "日光：格局感强，容易一上来就想做大事。",
        "石榴木": "石榴：外壳硬，内里密，靠韧性与时间开花。",
        "大海水": "大海：边界模糊，能容万物，也容易被情绪淹没。",
    }

    candidates: list[str] = []
    for row in bazi.pillar_details or []:
        nayin = (row.get("纳音") or "").strip()
        if not nayin:
            continue
        imagery = nayin_imagery.get(nayin)
        if imagery:
            candidates.append(f"{row.get('柱','')} {row.get('干支','')} 的纳音“{nayin}”：{imagery}")

    mk = bazi.markers or {}
    huagai = mk.get("华盖", {}) or {}
    yima = mk.get("驿马", {}) or {}
    tianyi = mk.get("天乙贵人", {}) or {}
    if huagai.get("positions"):
        candidates.append(f"华盖落在 {','.join(huagai.get('positions') or [])}：独立审美/沉浸专注的代价是社交隔离。")
    if yima.get("positions"):
        candidates.append(f"驿马落在 {','.join(yima.get('positions') or [])}：动中求势，靠迁移/换赛道拿结果。")
    if tianyi.get("positions"):
        candidates.append(f"天乙贵人出现于 {','.join(tianyi.get('positions') or [])}：关键时刻更容易遇到“兜底资源”。")

    if not candidates:
        return "（未计算）"

    if seed is None:
        try:
            base_seed = int.from_bytes(os.urandom(4), "big")
        except Exception:
            base_seed = int(hashlib.md5((bazi.four_pillars or "").encode("utf-8")).hexdigest()[:8], 16)
    else:
        base_seed = int(seed)
    rnd = random.Random(base_seed)
    picks = []
    for _ in range(min(2, len(candidates))):
        choice = rnd.choice(candidates)
        candidates = [c for c in candidates if c != choice]
        picks.append(choice)

    if len(picks) < 2:
        picks.append("掌纹的一个“微小杂纹/岛纹/断续”作为第二锚点：你需要自己从图里指出它的存在。")

    return "\n".join([f"- 锚点{i+1}：{p}" for i, p in enumerate(picks[:2])])


def _inverse_mapping(mapping: dict[str, str]) -> dict[str, str]:
    return {v: k for k, v in mapping.items()}


def _dimension_scores_for_year(
    *,
    bazi: BaziPro,
    year: int,
) -> dict:
    year_gz = str(LunarDate.from_solar_date(year, 6, 1).gz_year)
    dayun_gz = _dayun_gz_for_year(bazi.dayun, year)

    _, day_branch = _split_gz(bazi.day_gz)
    _, year_branch = _split_gz(year_gz)
    chong = ZHI_CHONG.get(day_branch) == year_branch and day_branch and year_branch

    day_elem = TIANGAN_WUXING.get(bazi.day_master)
    wealth_elem = _wealth_element_for(day_elem)
    career_elem = _kill_element_for(day_elem)
    output_elem = _output_element_for(day_elem)

    year_stem, year_zhi = _split_gz(year_gz)
    dy_stem, dy_zhi = _split_gz(dayun_gz or "")

    def _elem_of_stem(stem: str) -> Optional[str]:
        return TIANGAN_WUXING.get(stem)

    def _elem_of_zhi(zhi: str) -> Optional[str]:
        return DIZHI_WUXING.get(zhi)

    y_stem_e = _elem_of_stem(year_stem)
    y_zhi_e = _elem_of_zhi(year_zhi)
    dy_stem_e = _elem_of_stem(dy_stem)
    dy_zhi_e = _elem_of_zhi(dy_zhi)
    elem_hits = [y_stem_e, y_zhi_e, dy_stem_e, dy_zhi_e]

    kill_hits = _count_elem_hits(elem_hits, career_elem)
    output_hits = _count_elem_hits(elem_hits, output_elem)
    wealth_hits = _count_elem_hits(elem_hits, wealth_elem)

    def _score_against(target_elem: Optional[str], base: float, w: tuple[float, float, float, float]) -> float:
        if not target_elem:
            return base
        s = base
        s += _element_relation_score(y_stem_e, target_elem, w[0])
        s += _element_relation_score(y_zhi_e, target_elem, w[1])
        s += _element_relation_score(dy_stem_e, target_elem, w[2])
        s += _element_relation_score(dy_zhi_e, target_elem, w[3])
        return s

    wealth_index = _clamp_score(_score_against(wealth_elem, 50.0, (14.0, 8.0, 10.0, 5.0)))
    career_index = _clamp_score(_score_against(career_elem, 50.0, (13.0, 7.0, 9.0, 4.0)))

    markers = bazi.markers or {}
    peach_target = ((markers.get("桃花") or {}).get("target")) or ""
    romance_base = 45.0
    if peach_target and year_branch == peach_target:
        romance_base += 16.0
    if peach_target and dy_zhi and dy_zhi == peach_target:
        romance_base += 9.0
    romance_index = _clamp_score(romance_base + _element_relation_score(y_stem_e, day_elem, 6.0))

    transition_years = set(_dayun_transition_years(bazi.dayun))
    transition = year in transition_years

    def _vol(idx: int) -> float:
        v = 5.0 + abs(idx - 50) * 0.12
        if chong:
            v += 4.0
        if transition:
            v += 2.0
        return float(_clamp(v, 2.0, 20.0))

    wealth_vol = _vol(wealth_index)
    career_vol = _vol(career_index)
    romance_vol = _vol(romance_index)

    conflict_tags: list[str] = []
    mid_range = lambda x: 45 <= int(x) <= 60

    if max(wealth_vol, career_vol, romance_vol) >= 14.0 and (45 <= int((wealth_index + career_index + romance_index) / 3) <= 60):
        conflict_tags.append("极端转折")

    if kill_hits >= 2 and output_hits >= 2 and max(career_vol, wealth_vol) >= 10.0:
        conflict_tags.append("困兽之斗")

    peach_on_year = bool(peach_target and year_branch == peach_target)
    peach_on_dayun = bool(peach_target and dy_zhi and dy_zhi == peach_target)
    if (peach_on_year or peach_on_dayun) and bool(chong):
        conflict_tags.append("情场劫财")

    if mid_range(career_index) and career_vol >= 12.0 and kill_hits >= 1:
        conflict_tags.append("内耗期")

    if mid_range(wealth_index) and wealth_vol >= 12.0 and wealth_hits >= 1:
        conflict_tags.append("财来财去")

    if romance_index >= 70 and romance_vol >= 12.0:
        conflict_tags.append("桃花风暴")

    wealth_status = _dimension_status_text(
        dimension="wealth",
        index=wealth_index,
        vol=wealth_vol,
        conflict_tags=conflict_tags,
    )
    career_status = _dimension_status_text(
        dimension="career",
        index=career_index,
        vol=career_vol,
        conflict_tags=conflict_tags,
    )
    romance_status = _dimension_status_text(
        dimension="romance",
        index=romance_index,
        vol=romance_vol,
        conflict_tags=conflict_tags,
    )

    return {
        "year": year,
        "year_gz": year_gz,
        "dayun_gz": dayun_gz or "",
        "wealth": {"index": wealth_index, "vol": wealth_vol, "prob": wealth_index, "status": wealth_status},
        "career": {"index": career_index, "vol": career_vol, "prob": career_index, "status": career_status},
        "romance": {"index": romance_index, "vol": romance_vol, "prob": romance_index, "status": romance_status},
        "conflict_tags": conflict_tags,
        "chong": bool(chong),
        "is_dayun_transition": transition,
    }

def _get_bazi_pro(
    birth_date: date,
    birth_time: time,
    gender_for_yun: Optional[str],
) -> Optional[BaziPro]:
    try:
        from lunar_python import Solar  # type: ignore
    except Exception:
        return None

    solar_dt = _birth_datetime(birth_date, birth_time)

    solar = Solar.fromYmdHms(
        solar_dt.year, solar_dt.month, solar_dt.day, solar_dt.hour, solar_dt.minute, solar_dt.second
    )
    lunar = solar.getLunar()
    bazi = lunar.getEightChar()

    def _call_first(obj, names: list[str]):
        for name in names:
            fn = getattr(obj, name, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    continue
        return None

    year_gz = _call_first(bazi, ["getYearGz", "getYear"])
    month_gz = _call_first(bazi, ["getMonthGz", "getMonth"])
    day_gz = _call_first(bazi, ["getDayGz", "getDay"])
    hour_gz = _call_first(bazi, ["getHourGz", "getTimeGz", "getTime"])

    day_gan = _call_first(bazi, ["getDayGan"])
    if not day_gan and day_gz:
        day_gan = str(day_gz)[0]

    gender_code = 1 if gender_for_yun == "男" else 0
    yun = bazi.getYun(gender_code)
    dayun = yun.getDaYun()

    year_gz_s = str(year_gz)
    month_gz_s = str(month_gz)
    day_gz_s = str(day_gz)
    hour_gz_s = str(hour_gz)
    day_master_s = str(day_gan or "")

    def _pillar_row(label: str, gz_value: str, gan_ss: Optional[str], zhi_ss: Optional[str], nayin: Optional[str]):
        gan, zhi = _split_gz(gz_value)
        return {
            "柱": label,
            "干支": gz_value,
            "天干": gan,
            "地支": zhi,
            "干五行": TIANGAN_WUXING.get(gan, ""),
            "支五行": DIZHI_WUXING.get(zhi, ""),
            "十神(干)": str(gan_ss or ""),
            "十神(支)": str(zhi_ss or ""),
            "纳音": str(nayin or ""),
        }

    details = [
        _pillar_row(
            "年柱",
            year_gz_s,
            _call_first(bazi, ["getYearShiShenGan"]),
            _call_first(bazi, ["getYearShiShenZhi"]),
            _call_first(bazi, ["getYearNaYin"]),
        ),
        _pillar_row(
            "月柱",
            month_gz_s,
            _call_first(bazi, ["getMonthShiShenGan"]),
            _call_first(bazi, ["getMonthShiShenZhi"]),
            _call_first(bazi, ["getMonthNaYin"]),
        ),
        _pillar_row(
            "日柱",
            day_gz_s,
            _call_first(bazi, ["getDayShiShenGan"]),
            _call_first(bazi, ["getDayShiShenZhi"]),
            _call_first(bazi, ["getDayNaYin"]),
        ),
        _pillar_row(
            "时柱",
            hour_gz_s,
            _call_first(bazi, ["getTimeShiShenGan"]),
            _call_first(bazi, ["getTimeShiShenZhi"]),
            _call_first(bazi, ["getTimeNaYin"]),
        ),
    ]

    markers = _bazi_markers(day_master_s, year_gz_s, month_gz_s, day_gz_s, hour_gz_s)

    return BaziPro(
        four_pillars=f"{year_gz_s} {month_gz_s} {day_gz_s} {hour_gz_s}",
        day_master=day_master_s,
        day_gz=day_gz_s,
        year_gz=year_gz_s,
        month_gz=month_gz_s,
        hour_gz=hour_gz_s,
        dayun=dayun,
        birth_dt=solar_dt,
        pillar_details=details,
        markers=markers,
    )

# ==========================================
# 0. 核心配置
# ==========================================
st.set_page_config(
    page_title="掌纹密码解读手册", 
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🕸️",
)

# ==========================================
# 1. 环境与 API 配置
# ==========================================
# ⚠️ API Key 建议放到环境变量/Streamlit secrets（或 local_secrets.py）
api_key = get_google_api_key()
if not api_key:
    st.error("未检测到 Google API Key：请在 `.streamlit/secrets.toml` 里设置 `GOOGLE_API_KEY`。")
    st.stop()

# ⚠️ 本地调试用代理，部署时请注释
# os.environ['http_proxy'] = "http://127.0.0.1:7897"
# os.environ['https_proxy'] = "http://127.0.0.1:7897"

try:
    client = get_genai_client(api_key)
except Exception as e:
    st.error(f"API 初始化失败: {e}")
    st.stop()

# ==========================================
# 2. 命理计算引擎
# ==========================================
def get_shizhu(day_gan, hour):
    """五鼠遁元法：根据日干和时辰推算时柱"""
    gan_map = {"甲": 0, "乙": 1, "丙": 2, "丁": 3, "戊": 4, "己": 0, "庚": 1, "辛": 2, "壬": 3, "癸": 4}
    zhi_list = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
    gan_list = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
    
    zhi_index = (hour + 1) // 2 % 12
    shizhi = zhi_list[zhi_index]
    
    start_gan_index = (gan_map.get(day_gan, 0) * 2) % 10
    shigan_index = (start_gan_index + zhi_index) % 10
    shigan = gan_list[shigan_index]
    
    return f"{shigan}{shizhi}"

def get_full_bazi_engine(date_obj, time_obj, gender="男"):
    if not date_obj: return None
    try:
        ld = LunarDate.from_solar_date(date_obj.year, date_obj.month, date_obj.day)
        
        gz_year = ld.gz_year   
        gz_month = ld.gz_month 
        gz_day = ld.gz_day     
        
        day_gan = gz_day[0]
        gz_hour = get_shizhu(day_gan, time_obj.hour)
        
        full_bazi = f"{gz_year} {gz_month} {gz_day} {gz_hour}"
        
        current_year = datetime.now().year
        current_ld = LunarDate.from_solar_date(current_year, 6, 1) 
        liu_nian = current_ld.gz_year 
        
        return {
            "四柱": full_bazi,
            "日主": f"{gz_day[0]}",
            "流年": f"{liu_nian} ({current_year})",
            "性别": gender
        }
    except Exception as e:
        return None


def _bazi_to_display_dict(pro: BaziPro) -> dict:
    current_year = datetime.now().year
    return {
        "四柱": pro.four_pillars,
        "日主": pro.day_master,
        "流年": f"{LunarDate.from_solar_date(current_year, 6, 1).gz_year} ({current_year})",
        "性别": "",
    }

# ==========================================
# 3. UI 界面
# ==========================================
st.markdown("""
    <style>
    .main { background-color: #f7f7f7; color: #1a2a3a; font-family: "PingFang SC", sans-serif; }
    .stButton>button { 
        width: 100%; border-radius: 0px; background-color: #000; color: white; 
        height: 3.5em; font-weight: bold; border: none; letter-spacing: 2px;
    }
    .stButton>button:hover { background-color: #333; }
    .report-box { 
        background-color: #fff; padding: 40px; border: 1px solid #000; 
        margin-top: 20px;
        box-shadow: 10px 10px 0px rgba(0,0,0,0.1);
        font-family: "Songti SC", "SimSun", serif;
    }
    .bazi-row {
        display: flex; justify-content: space-between; border-bottom: 2px solid #000;
        padding-bottom: 10px; margin-bottom: 20px; font-family: monospace;
    }
    h3 { border-left: 5px solid #000; padding-left: 10px; }
    .upload-header { font-weight: bold; margin-bottom: 10px; display: block;}
    @media (max-width: 600px) {
        .stButton>button { height: 3.2em; }
        .report-box { padding: 16px; }
        .bazi-row { flex-direction: column; align-items: flex-start; gap: 6px; }
        div[data-testid="stHorizontalBlock"] { flex-direction: column !important; }
        div[data-testid="stHorizontalBlock"] > div { width: 100% !important; flex: 1 1 100% !important; }
    }
    </style>
    """, unsafe_allow_html=True)

st.title("掌纹解读报告")
st.caption("填写生辰信息并上传左右手照片，生成一份更具体的解读（仅供参考）。")

with st.sidebar:
    st.header("设置")
    st.info("提示：解读内容基于你提供的信息与照片生成，仅供参考，不构成医疗/法律/投资建议。")
    rich_output_mode = st.checkbox("详细报告（更长更具体）", value=True)
    high_precision_mode = st.checkbox("更稳模式（更慢）", value=True)
    attach_enhanced_images = st.checkbox("启用纹路增强（更慢）", value=True)
    randomize_output = st.checkbox("每次输出略有不同", value=False)

    st.divider()
    show_life_kline = st.checkbox("展示人生K线图（模型化）", value=True)

with st.container(border=True):
    st.subheader("基本信息")
    cols = st.columns(2, gap="small")
    with cols[0]:
        birth_date = st.date_input(
            "出生日期（1960-2020）",
            value=None,
            min_value=date(1960, 1, 1),
            max_value=date(2020, 12, 31),
            format="YYYY-MM-DD",
        )
    with cols[1]:
        birth_time = st.time_input("出生时辰", value=time(8, 15))

    cols2 = st.columns(2, gap="small")
    with cols2[0]:
        gender = st.selectbox("性别（可选）", ["不填写", "男", "女", "非二元/其他"], index=0)
    with cols2[1]:
        relationship_preference = st.selectbox(
            "关系偏好（可选）",
            ["不填写", "不限定性别", "偏好男性", "偏好女性", "偏好多元/不设限"],
            index=1,
        )

    st.divider()
    xian_tian_method = st.selectbox(
        "左右手判定方式",
        ["不区分（只做左右手对比）", "左手为先天", "右手为先天", "按传统（男左女右）"],
        index=0,
        help="不想被传统规则限制，选“不区分”或手动指定先天手即可。",
    )

# 双列布局上传
st.markdown("请分别上传左手和右手的清晰照片（建议自然光、避免反光、掌心占画面大部分）。")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<span class="upload-header">🤚 左手样本 (Left)</span>', unsafe_allow_html=True)
    file_left = st.file_uploader("上传左手", type=["jpg", "png", "jpeg"], key="left")
    img_left = PIL.Image.open(file_left) if file_left else None
    if img_left: st.image(img_left, use_container_width=True)

with col2:
    st.markdown('<span class="upload-header">✋ 右手样本 (Right)</span>', unsafe_allow_html=True)
    file_right = st.file_uploader("上传右手", type=["jpg", "png", "jpeg"], key="right")
    img_right = PIL.Image.open(file_right) if file_right else None
    if img_right: st.image(img_right, use_container_width=True)

if img_left or img_right:
    with st.expander("照片清晰度检测", expanded=False):
        st.caption("清晰度评分仅用于判断照片是否适合识别纹路；分数越高，解读越稳定。")

        cols = st.columns(2)
        if img_left:
            qm = _image_quality_metrics(img_left)
            score = _clarity_score(qm)
            with cols[0]:
                st.subheader("左手")
                st.write(f"{qm['width']}×{qm['height']} | edge_var：{qm['edge_var']:.1f}")
                st.metric("清晰度评分", f"{score}/100", delta=_clarity_grade(score))
                st.progress(score / 100)
                if qm["width"] < 900 or qm["height"] < 900:
                    st.warning("分辨率偏低：建议至少 900×900 以上、掌心占画面 70% 左右。")
                if qm["edge_var"] < 80:
                    st.warning("可能偏糊/反光：建议补光、避免强反光、对焦更清晰。")

        if img_right:
            qm = _image_quality_metrics(img_right)
            score = _clarity_score(qm)
            with cols[1]:
                st.subheader("右手")
                st.write(f"{qm['width']}×{qm['height']} | edge_var：{qm['edge_var']:.1f}")
                st.metric("清晰度评分", f"{score}/100", delta=_clarity_grade(score))
                st.progress(score / 100)
                if qm["width"] < 900 or qm["height"] < 900:
                    st.warning("分辨率偏低：建议至少 900×900 以上、掌心占画面 70% 左右。")
                if qm["edge_var"] < 80:
                    st.warning("可能偏糊/反光：建议补光、避免强反光、对焦更清晰。")


if st.button("生成解读报告"):
    if not img_left or not img_right:
        st.error("请同时上传左手和右手的照片。")
    else:
        if not birth_date:
            st.error("请先选择出生日期（1960-2020）。")
            st.stop()

        gender_for_display = None if gender == "不填写" else gender
        relationship_for_display = None if relationship_preference == "不填写" else relationship_preference

        bazi_pro = _get_bazi_pro(
            birth_date=birth_date,
            birth_time=birth_time,
            gender_for_yun=gender_for_display if gender_for_display in ("男", "女") else "男",
        )

        if bazi_pro:
            bazi_res = _bazi_to_display_dict(bazi_pro)
        else:
            bazi_res = get_full_bazi_engine(birth_date, birth_time, gender_for_display or "男")
        
        if not bazi_res:
            st.error("八字计算失败（返回 None）：请确认出生日期/时辰有效，并检查 `borax` 依赖是否安装正常。")
            st.stop()

        if bazi_res:
            if xian_tian_method == "按传统（男左女右）" and gender_for_display not in ("男", "女"):
                st.error("你选择了“按传统（男左女右）”，但性别未填写为男/女；请改成手动指定或选择“不区分”。")
                st.stop()

            progress_slot = st.empty()
            status_slot = st.empty()
            progress_bar = progress_slot.progress(0)

            def _set_progress(pct: int, msg: str) -> None:
                progress_bar.progress(int(round(_clamp(float(pct), 0.0, 100.0))))
                status_slot.caption(msg)

            try:
                _set_progress(8, "步骤 1/6：整理信息与左右手口径…")
                 
                # 判定先天与后天（可选）
                xian_tian = None
                hou_tian = None
                if xian_tian_method == "左手为先天":
                    xian_tian, hou_tian = "左手", "右手"
                elif xian_tian_method == "右手为先天":
                    xian_tian, hou_tian = "右手", "左手"
                elif xian_tian_method == "按传统（男左女右）":
                    xian_tian, hou_tian = ("左手", "右手") if gender_for_display == "男" else ("右手", "左手")

                xian_tian_label = xian_tian or "不区分"
                hou_tian_label = hou_tian or "不区分"

                left_qm = _image_quality_metrics(img_left)
                right_qm = _image_quality_metrics(img_right)
                image_order_desc = "1) 左手原图；2) 右手原图。"
                if attach_enhanced_images:
                    image_order_desc += "\\n3) 左手增强；4) 右手增强；5) 左手边缘；6) 右手边缘。"

                left_score = _clarity_score(left_qm)
                right_score = _clarity_score(right_qm)
                _set_progress(22, "步骤 2/6：计算清晰度与报告长度…")

                target_length = "约 1800-2600 字" if rich_output_mode else "约 900-1400 字"
                seed = (
                    None
                    if randomize_output
                    else (int(_stable_seed(bazi_res["四柱"], birth_date, birth_time, gender_for_display, relationship_for_display, xian_tian_method, rich_output_mode)) % 2147483647)
                )

                bazi_pillars = (bazi_res.get("四柱") or "").split()
                bazi_pillars = bazi_pillars + [""] * (4 - len(bazi_pillars))
                bazi_table = [
                    {"柱": "年柱", "干": (bazi_pillars[0][:1] if bazi_pillars[0] else ""), "支": (bazi_pillars[0][1:2] if len(bazi_pillars[0]) >= 2 else "")},
                    {"柱": "月柱", "干": (bazi_pillars[1][:1] if bazi_pillars[1] else ""), "支": (bazi_pillars[1][1:2] if len(bazi_pillars[1]) >= 2 else "")},
                    {"柱": "日柱", "干": (bazi_pillars[2][:1] if bazi_pillars[2] else ""), "支": (bazi_pillars[2][1:2] if len(bazi_pillars[2]) >= 2 else "")},
                    {"柱": "时柱", "干": (bazi_pillars[3][:1] if bazi_pillars[3] else ""), "支": (bazi_pillars[3][1:2] if len(bazi_pillars[3]) >= 2 else "")},
                ]

                _set_progress(34, "步骤 3/6：计算盘面信息与人生节奏…")
                extra = ""
                dayun_lines: list[str] = []
                if bazi_pro:
                    markers = bazi_pro.markers or {}
                    tianyi = markers.get("天乙贵人", {})
                    peach = markers.get("桃花", {})
                    extra = (
                        f"｜天乙贵人：{','.join(tianyi.get('positions', []) or []) or '无'}"
                        f"｜桃花：{','.join(peach.get('positions', []) or []) or '无'}"
                    )

                    if bazi_pro.dayun:
                        for dy in bazi_pro.dayun:
                            try:
                                dayun_lines.append(f"{dy.getGanZhi()}（{dy.getStartYear()}-{dy.getEndYear()}）")
                            except Exception:
                                continue

                life = None
                kline_hint = ""
                highlight_years_hint = ""
                low_years_hint = ""

                if bazi_pro:
                    life = _build_life_kline(bazi=bazi_pro, max_age=100)
                    rows = life["rows"]

                    turns = life.get("top_turns") or []
                    top_years = [str(y) for y, _ in turns[:5]]
                    if top_years:
                        kline_hint = "、".join(top_years)

                    adult_rows = [r for r in rows if int(r["age"]) >= 18]

                    def _volatility(r: dict) -> float:
                        try:
                            return float(r.get("high", 0.0)) - float(r.get("low", 0.0))
                        except Exception:
                            return 0.0

                    def _opportunity_rank(r: dict) -> float:
                        close = float(r.get("close", 0.0))
                        vol = _volatility(r)
                        return close - 0.1 * vol + (2.0 if r.get("is_dayun_transition") else 0.0)

                    def _pressure_rank(r: dict) -> float:
                        close = float(r.get("close", 0.0))
                        vol = _volatility(r)
                        chong_bonus = 2.0 if r.get("chong") else 0.0
                        # 波折更多来自“波动+消耗”，但不等于坏
                        return (60.0 - close) * 0.55 + vol * 0.35 + chong_bonus + (2.0 if r.get("is_dayun_transition") else 0.0)

                    high_rows = sorted(adult_rows, key=_opportunity_rank, reverse=True)[:6]
                    pressure_rows = sorted(adult_rows, key=_pressure_rank, reverse=True)[:6]

                    if high_rows:
                        highlight_years_hint = "、".join(str(r["year"]) for r in high_rows[:5])
                    if pressure_rows:
                        low_years_hint = "、".join(str(r["year"]) for r in pressure_rows[:5])

                life_avg_score = None
                life_open_close_keyword = None
                macro_trend_text = ""
                if life:
                    rows = life.get("rows") or []
                    adult_rows = [r for r in rows if int(r.get("age", 0)) >= 18]
                    base_rows = adult_rows or rows
                    if base_rows:
                        life_avg_score = sum(float(r.get("close", 0.0)) for r in base_rows) / float(len(base_rows))
                    life_open_close_keyword = _life_open_close_keyword(rows)
                    macro_trend_text = str(life.get("macro_trend_text") or "").strip()

                _set_progress(46, "步骤 4/6：准备未来三年指标与写作锚点…")

                current_year = date.today().year
                future_years = [current_year + i for i in range(1, 4)]
                future_range = f"{future_years[0]}-{future_years[-1]}"
                future_metrics = []
                future_metrics_text = "（未计算）"
                markers_text = "（未计算）"
                plate_details_text = "（未计算）"
                if bazi_pro:
                    future_metrics = [_dimension_scores_for_year(bazi=bazi_pro, year=y) for y in future_years]
                    future_metrics_text = "\n".join(
                        [
                            (
                                f"- {m['year']}（{m['year_gz']}）："
                                f"财运 {m['wealth']['index']}±{m['wealth']['vol']:.1f}（{m['wealth']['prob']}%）｜{m['wealth'].get('status','')} "
                                f"｜事业 {m['career']['index']}±{m['career']['vol']:.1f}（{m['career']['prob']}%）｜{m['career'].get('status','')} "
                                f"｜关系 {m['romance']['index']}±{m['romance']['vol']:.1f}（{m['romance']['prob']}%）｜{m['romance'].get('status','')}"
                                f"{('｜冲突标签：' + '、'.join(m.get('conflict_tags') or [])) if (m.get('conflict_tags') or []) else ''}"
                            )
                            for m in future_metrics
                        ]
                    )

                    mk = bazi_pro.markers or {}
                    tianyi = mk.get("天乙贵人", {})
                    peach = mk.get("桃花", {})
                    yima = mk.get("驿马", {})
                    huagai = mk.get("华盖", {})
                    markers_text = "\n".join(
                        [
                            f"- 天乙贵人（日干 {bazi_pro.day_master}）：落支 {','.join(tianyi.get('targets', []) or []) or '无'}；出现于 {','.join(tianyi.get('positions', []) or []) or '无'}",
                            f"- 桃花：{(peach.get('target') or '无')}；出现于 {','.join(peach.get('positions', []) or []) or '无'}",
                            f"- 驿马：{(yima.get('target') or '无')}；出现于 {','.join(yima.get('positions', []) or []) or '无'}",
                            f"- 华盖：{(huagai.get('target') or '无')}；出现于 {','.join(huagai.get('positions', []) or []) or '无'}",
                        ]
                    )
                    plate_details_text = "\n".join(
                        [
                            f"- {row.get('柱','')} {row.get('干支','')}｜十神(干/支) {row.get('十神(干)','')}/{row.get('十神(支)','')}｜纳音 {row.get('纳音','')}｜五行(干/支) {row.get('干五行','')}/{row.get('支五行','')}"
                            for row in (bazi_pro.pillar_details or [])
                        ]
                    )

                yongshen_text = "（未计算）"
                missing_talents_text = "（未计算）"
                breakout_anchors = "（未计算）"
                if bazi_pro:
                    ys = _yongshen_profile(bazi_pro)
                    yongshen_text = ys.get("summary") or "（未计算）"
                    missing_talents_text = ys.get("missing_talents") or "（未计算）"
                breakout_anchors = _breakout_anchors_text(bazi=bazi_pro, seed=seed)

                # --- 核心：更自然、更具体的写作风格 ---
                final_prompt = f"""
                请你以“做了 20 年整合营销、又深研过子平命理的前辈”的口吻写一份中文报告：笃定、有分色感，有人味儿。
                场景：在私人会所里跟后辈交心，语气直接但不刻薄。

                [语言审美红线]（必须遵守）
                - 拒绝抽象：禁止出现这些词：维度、矩阵、机制、杠杆、优化。
                - 拒绝空洞：每一句分析必须“带数据上岗”。至少包含 1 个具体数字（年份/百分比/指数/波动/损耗率等）+ 1 个具体事实（左右手差异/纳音意象/神煞位置/冲突标签/大运流年信息）。
                - 禁止万金油：禁止“自古以来/每个人都有/总体来说/因人而异/可能/大概/或许”等敷衍开场。
                - 禁止自曝后台：不要提及“模型/提示词/系统/参数/Token/置信度”等字眼。

                [写作硬规则]（必须遵守）
                1) 先天 vs 后天必须写成“博弈感”：不要单独说左手怎样右手怎样，要写出“出厂配置 vs 后天改写”的拉扯。
                2) 对比分析法（强制）：每个核心结论必须同时提到左手（先天）与右手（后天）至少 1 处“具体可感知差异”（线条走向/深浅/断续/分叉/岛纹/掌丘饱满度等）。看不清就直说“看不清”，并给出重拍建议。
                3) 用神叙事（强制）：结合下方“用神/忌神（参考）”，判定命主更像“顺流起步，逆流操盘”还是“顺流躺平”，并写出这种进化的能量损耗率（__%）。
                4) 冲突检测（强制）：遇到“波动很大但指数中等”的年份，不许写“平稳”，必须使用并解释标签（极端转折/内耗期/困兽之斗/情场劫财/桃花风暴）。
                5) 时空连贯性（强制）：如果“宏观节奏”提示蛰伏/阴跌，你的战术建议必须体现“先稳住系统，再谈冲刺”。
                6) 五行缺失（强制）：不准写“补元素”；只能写“独特天赋 + 代价 + 管理方式”。
                7) 关系部分不限定性别，不用“婚姻/恋爱”字眼，只谈“深度关系里的能量交换与损耗”。

                [破局指令]
                - 本次报告必须围绕“破局锚点”里的 2 个小众特征展开（纳音意象/不起眼的神煞/掌纹微小杂纹）。
                - 报告的前三句话必须围绕这两个锚点展开，禁止任何空话起手。

                **【档案数据】**
                - 性别（可选）：{gender_for_display or "未填写"}
                - 关系偏好（可选）：{relationship_for_display or "不填写/不设限"}
                - 八字原局：{bazi_res['四柱']} (日主：{bazi_res['日主']})
                - 当前流年：{bazi_res['流年']}
                - 当前公历年份：{current_year}（明年={future_years[0]}）
                - 未来三年（固定）：{future_years[0]}、{future_years[1]}、{future_years[2]}（文中如果写“明年/后年/第三年”，必须严格对应这三年，不要写成其他年份）
                - 可能波动较大的年份（模型参考）：{kline_hint or "未计算/无"}
                - 可能的机会窗口年份（参考）：{highlight_years_hint or "未计算/无"}
                - 可能的波折窗口年份（参考）：{low_years_hint or "未计算/无"}（注意：波折=变动/消耗/选择压力，不等于“过得差”）
                - 宏观节奏（参考）：{macro_trend_text or "未计算/无"}

                **【用神/忌神（参考，不是定论）】**
                {yongshen_text}

                **【五行缺失的独特天赋（不要写成“补元素建议”）】**
                {missing_talents_text}

                **【破局锚点（本次随机抽取 2 个）】**
                {breakout_anchors}

                **【排盘细项（请融入解释，不要原样照抄）】**
                {plate_details_text}

                **【盘面标记（可用于更“有依据”的解释）】**
                {markers_text}

                解读提示（用于提高专业度）：
                - 天乙贵人：一般表示“遇事有人/有资源兜底”的倾向；落在年柱偏早年/长辈助力，月柱偏工作平台/贵人同事，日柱偏自带福气或伴侣助力，时柱偏后期机会/晚运资源。
                - 桃花：不要只解读成“异性缘”，更像“被看见/被喜欢/社交吸引力”的窗口期；有利也有风险（烂桃花/情绪牵扯）。
                - 驿马：更像“动”的信号（换城市、换赛道、出差奔波、迁移）；动得好是机会，动得乱是消耗。
                - 华盖：偏“独立/审美/学术/宗教感/孤高”，适合沉下去做事，但也要注意社交隔离。

                **【未来三年趋势（模型化，固定为 {future_range}）】**
                下面每行包含：指数±波动（发生概率%）+ 状态文案 + 冲突标签；请在事业/财运/关系的解释里引用（至少引用 2 个不同标签或状态）。
                {future_metrics_text}

                **【照片清晰度评分（仅供参考）】**
                - 左手：{left_score}/100（{left_qm['width']}×{left_qm['height']}）
                - 右手：{right_score}/100（{right_qm['width']}×{right_qm['height']}）

                **【样本定义】**
                你会收到图片（按传入顺序）：
                {image_order_desc}

                **【左右手解读口径】**
                - 如果用户选择“不区分”，只做左右手对比，不要强行定义先天/后天。
                - 如果用户指定了先天/后天：先天={xian_tian_label}，后天={hou_tian_label}。
                - 若为“不区分”，请把“先天 vs 后天”理解成“左手 vs 右手”的差异即可。

                **【输出结构 - 请逐一输出】**

                ### 轨迹：[先天局限] 与 [后天破局]
                - 用“八字（出厂配置）+ 左右手差异（改写痕迹）”写出一段有博弈感的开场：先天底座很强/很弱？后天是修正、妥协还是硬改？
                - 必须写出一句“分色句”：例如“顺流起步，逆流操盘 / 顺流躺平 / 逆流进化”等，且给出能量损耗率：__%（0-100）。
                - 必须点名 2 条“证据锚点”，且都来自“破局锚点”。

                ### 内核：[性格的明线] 与 [认知的暗线]
                - 明线（性格）：日主 {bazi_res['日主']} 在这个格局下的原始诉求是什么（求稳/求名/求自由/求掌控/求安全感），给一个明确结论。
                - 暗线（冲突）：引用至少 1 个“冲突标签”（例如极端转折/内耗期/困兽之斗/情场劫财/桃花风暴），说明“你为什么看起来稳，但内部在打架”。
                - 防御动作：用职场/心理学语境写出 1 个自动化防御（例如过度理性、完美主义、冷处理、抢先否定、讨好式控制），并写出机会成本（至少 1 个具体场景）。

                ### 势能：[财富与事业的生存策略]
                - 把“财运/事业”翻译成“生存策略”：你更像靠窗口期、靠平台、还是靠死磕把优势做出来？
                - 三个来源（必须输出强/中/弱）：
                  - 天（窗口期/溢价/突然的机会）= 【强/中/弱】
                  - 地（平台/行业红利/人脉/兜底）= 【强/中/弱】
                  - 人（执行/抗压/复盘/稳定产出）= 【强/中/弱】
                - 给一个具体职业画像：引用至少 3 个盘面事实（驿马/桃花/天乙贵人/华盖/纳音意象/冲突标签），说清“更适合做什么/不适合做什么/为什么”。
                - 横财倾向：输出 __% + 1 条资金止损规则（必须可执行）。

                ### 镜像：[亲密关系中的投射与边界]
                - 依恋光谱图（必须整数，总和=100）：安全型__% / 焦虑型__% / 回避型__% / 恐惧-回避型__%。
                - 触发器：结合盘面具体事实解释你更容易触发的应激反应（例如官杀压力、印星过旺、财星受制等）。
                - 能量损耗动作：写出你在深度关系里最常见的 1 个消耗动作，以及你用它换来的“短期好处”。
                - 边界建议：2 条（每条都要写“怎么做 + 不做会怎样”）。

                ### 应期：[未来三年的伏笔与高光]
                - 先判宏观：结合“宏观节奏（参考）”+ 未来三年数据，判定这是三年的“向上筑底”还是“高位减持”。
                - 关键节点（必须引用冲突标签）：
                  - 必须按下暂停键：{future_years[0]} / {future_years[1]} / {future_years[2]} 中选 2 个（写清对应冲突标签 + 停什么）
                  - 必须 ALL IN：{future_years[0]} / {future_years[1]} / {future_years[2]} 中选 2 个（写清对应冲突标签 + 冲什么）
                - 风控底线：给 1 条资金止损 + 1 条情绪止损（必须具体可执行）。

                ### 收尾（固定格式）
                - 最后一行必须输出：未来三年关键词：词1、词2、词3（3-6个关键词，用顿号/逗号分隔）

                **输出风格要求：**
                - 语气自然、清晰、有分寸；避免“万能句”。
                - 信息尽量具体，不要写大段空话。

                **长度建议：总输出 {target_length}**
                """


                try:
                    # 优先使用 Pro 模型以获得更好的图像识别能力，并传入两张图片
                    model_name = "gemini-2.5-pro" 
                    part_left = pil_image_to_part(img_left)
                    part_right = pil_image_to_part(img_right)
                    parts = [part_left, part_right]
                    if attach_enhanced_images:
                        left_enh, left_edges = _enhance_for_lines(img_left)
                        right_enh, right_edges = _enhance_for_lines(img_right)
                        parts.extend(
                            [
                                pil_image_to_part(left_enh),
                                pil_image_to_part(right_enh),
                                pil_image_to_part(left_edges),
                                pil_image_to_part(right_edges),
                            ]
                        )

                    _set_progress(72, "步骤 5/6：生成报告（这一步耗时较长）…")
                    gen_kwargs = dict(
                        temperature=0.2 if high_precision_mode else 0.35,
                        topP=0.9,
                        maxOutputTokens=8192 if rich_output_mode else 4096,
                    )
                    if seed is not None:
                        gen_kwargs["seed"] = int(seed)
                    gen_config = genai.types.GenerateContentConfig(**gen_kwargs)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[final_prompt, *parts],
                        config=gen_config,
                    )

                    report_text = _extract_text_from_genai_response(response)
                    candidates = getattr(response, "candidates", None) or []
                    finish_reason = None
                    if candidates:
                        finish_reason = getattr(candidates[0], "finish_reason", None)

                    if finish_reason == "MAX_TOKENS":
                        st.warning("输出太长被截断了：已尽量展示可获取的内容。可以关闭“详细报告”或关闭“启用纹路增强”再试。")

                    if not report_text:
                        st.error("生成失败：未返回可显示的正文。请稍后重试，或减少输入图片/关闭详细模式。")
                        try:
                            summary = []
                            for c in candidates[:3]:
                                summary.append(
                                    {
                                        "finish_reason": getattr(c, "finish_reason", None),
                                        "avg_logprobs": getattr(c, "avg_logprobs", None),
                                        "safety_ratings": getattr(c, "safety_ratings", None),
                                    }
                                )
                            if summary:
                                st.json({"candidates": summary})
                        except Exception:
                            pass
                        raise RuntimeError("生成失败：未返回可显示的正文。")
                      
                    _set_progress(92, "步骤 6/6：整理排版并展示结果…")
                    st.markdown("---")
                    with st.container(border=True):
                        st.subheader("解读报告")
                        meta_cols = st.columns(2, gap="small")
                        with meta_cols[0]:
                            gender_caption = gender_for_display or "未填写"
                            st.caption(f"📅 {birth_date} {birth_time}｜{gender_caption}")
                        with meta_cols[1]:
                            st.caption(f"🧬 {bazi_res['四柱']}｜先天：{xian_tian_label}｜后天：{hou_tian_label}")

                        st.caption(f"未来三年范围：{future_range}")

                        cleaned_report = _strip_footer_from_report(report_text)
                        st.markdown(cleaned_report)

                        keywords = _extract_future_keywords(report_text)
                        if life_avg_score is not None or life_open_close_keyword or keywords:
                            st.markdown("---")
                            if life_avg_score is not None:
                                st.caption(f"人生平均分：{life_avg_score:.0f}/100")
                            if life_open_close_keyword:
                                st.caption(f"你的先天/后天人生对比关键词属于：{life_open_close_keyword}")
                            if keywords:
                                st.markdown(
                                    f"<div style='font-size:1.15rem;font-weight:800;'>未来三年关键词：{keywords}</div>",
                                    unsafe_allow_html=True,
                                )

                    with st.expander("八字排盘（展开查看）", expanded=False):
                        if bazi_pro:
                            st.dataframe(bazi_pro.pillar_details, use_container_width=True, hide_index=True)
                        else:
                            st.table(bazi_table)

                        st.caption(
                            f"四柱：{bazi_res.get('四柱', '')}｜日主：{bazi_res.get('日主', '')}｜流年：{bazi_res.get('流年', '')}{extra}"
                        )

                        if dayun_lines:
                            with st.expander("大运列表", expanded=False):
                                st.write("；".join(dayun_lines))

                    if show_life_kline and bazi_pro and life:
                        rows = life["rows"]
                        with st.expander("人生K线图（模型化）", expanded=False):
                            st.caption(
                                "这是把“大运/流年 + 五行关系”等规则映射成 0-100 指数的可视化，用来观察人生节奏与波动；不是客观预测。"
                            )

                            try:
                                try:
                                    import plotly.graph_objects as go  # type: ignore

                                    fig = go.Figure(
                                        data=[
                                            go.Candlestick(
                                                x=[r["x"] for r in rows],
                                                open=[r["open"] for r in rows],
                                                high=[r["high"] for r in rows],
                                                low=[r["low"] for r in rows],
                                                close=[r["close"] for r in rows],
                                                increasing_line_color="#111111",
                                                decreasing_line_color="#999999",
                                                showlegend=False,
                                            )
                                        ]
                                    )

                                    dy_x = [r["x"] for r in rows if r["is_dayun_transition"]]
                                    for x in dy_x:
                                        fig.add_vline(x=x, line_width=1, line_dash="dot", line_color="#cccccc")

                                    fig.update_layout(
                                        height=420,
                                        margin=dict(l=10, r=10, t=10, b=10),
                                        xaxis=dict(
                                            rangeslider=dict(visible=False),
                                            tickmode="array",
                                            tickvals=[rows[i]["x"] for i in range(0, len(rows), 10)],
                                            tickangle=-35,
                                        ),
                                        yaxis=dict(range=[0, 100]),
                                    )

                                    st.plotly_chart(fig, use_container_width=True)
                                except ModuleNotFoundError:
                                    import pandas as pd  # type: ignore
                                    import altair as alt  # type: ignore

                                    df = pd.DataFrame(rows)
                                    df["direction"] = df.apply(
                                        lambda r: "up" if float(r["close"]) >= float(r["open"]) else "down",
                                        axis=1,
                                    )

                                    base = alt.Chart(df).encode(
                                        x=alt.X(
                                            "age:Q",
                                            axis=alt.Axis(title="年龄", tickCount=11, labelAngle=-35),
                                            scale=alt.Scale(domain=[0, 100]),
                                        )
                                    )

                                    wick = base.mark_rule(color="#777").encode(
                                        y=alt.Y("low:Q", scale=alt.Scale(domain=[0, 100]), title="指数"),
                                        y2="high:Q",
                                        tooltip=[
                                            alt.Tooltip("age:Q", title="年龄"),
                                            alt.Tooltip("year:Q", title="年份"),
                                            alt.Tooltip("year_gz:N", title="流年"),
                                            alt.Tooltip("dayun_gz:N", title="大运"),
                                            alt.Tooltip("open:Q", title="开", format=".1f"),
                                            alt.Tooltip("close:Q", title="收", format=".1f"),
                                            alt.Tooltip("high:Q", title="高", format=".1f"),
                                            alt.Tooltip("low:Q", title="低", format=".1f"),
                                        ],
                                    )

                                    body = base.mark_bar(size=6).encode(
                                        y="open:Q",
                                        y2="close:Q",
                                        color=alt.condition(
                                            "datum.direction == 'up'",
                                            alt.value("#111111"),
                                            alt.value("#999999"),
                                        ),
                                    )

                                    transitions = (
                                        base.transform_filter("datum.is_dayun_transition")
                                        .mark_rule(color="#cccccc", strokeDash=[2, 2])
                                        .encode(x="age:Q")
                                    )

                                    chart = alt.layer(wick, body, transitions).properties(height=360)
                                    st.altair_chart(chart, use_container_width=True)

                            except Exception as e:
                                st.warning(f"图表组件不可用：{e}")

                            adult_rows = [r for r in rows if int(r["age"]) >= 18]
                            def _volatility(r: dict) -> float:
                                try:
                                    return float(r.get("high", 0.0)) - float(r.get("low", 0.0))
                                except Exception:
                                    return 0.0

                            def _opportunity_rank(r: dict) -> float:
                                close = float(r.get("close", 0.0))
                                vol = _volatility(r)
                                return close - 0.1 * vol + (2.0 if r.get("is_dayun_transition") else 0.0)

                            def _pressure_rank(r: dict) -> float:
                                close = float(r.get("close", 0.0))
                                vol = _volatility(r)
                                chong_bonus = 2.0 if r.get("chong") else 0.0
                                return (60.0 - close) * 0.55 + vol * 0.35 + chong_bonus + (2.0 if r.get("is_dayun_transition") else 0.0)

                            opportunity_rows = sorted(adult_rows, key=_opportunity_rank, reverse=True)[:6]
                            pressure_rows = sorted(adult_rows, key=_pressure_rank, reverse=True)[:6]

                            if opportunity_rows:
                                with st.expander("机会窗口（模型参考）", expanded=False):
                                    st.table(
                                        [
                                            {"年份": r["year"], "年龄": r["age"], "指数": round(float(r["close"]), 1)}
                                            for r in opportunity_rows
                                        ]
                                    )
                            if pressure_rows:
                                with st.expander("波折窗口（模型参考）", expanded=False):
                                    st.caption("提示：波折=变动/消耗/选择压力，不等于“过得差”；很多人恰恰会在波折期完成跃迁。")
                                    st.table(
                                        [
                                            {"年份": r["year"], "年龄": r["age"], "指数": round(float(r["close"]), 1)}
                                            for r in pressure_rows
                                        ]
                                    )

                    elif show_life_kline and not bazi_pro:
                        with st.expander("人生K线图（模型化）", expanded=False):
                            st.info("当前环境未安装 `lunar_python`，暂无法生成大运/人生K线图；部署端安装依赖后即可使用。")
                    
                except Exception as e:
                    if isinstance(e, (RuntimeError, ValueError)):
                        st.error(str(e))
                    else:
                        st.error(f"分析中断: {str(e)}")
                        st.caption("提示：请检查网络连接与 API Key 配置。")
            except Exception as e:
                st.error(f"分析中断: {str(e)}")
                st.caption("提示：请检查出生信息/依赖库是否正常，或稍后重试。")
            finally:
                progress_slot.empty()
                status_slot.empty()
