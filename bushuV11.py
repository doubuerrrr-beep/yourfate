import streamlit as st
import PIL.Image
from google import genai
import os
import hashlib
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
    score += _element_relation_score(year_stem_elem, day_elem, 12.0)
    score += _element_relation_score(year_branch_elem, day_elem, 7.0)
    score += _element_relation_score(dayun_stem_elem, day_elem, 9.0)
    score += _element_relation_score(dayun_branch_elem, day_elem, 4.0)

    chong = ZHI_CHONG.get(day_branch) == year_branch and day_branch and year_branch
    if chong:
        score -= 10.0

    if year_stem and day_stem and year_stem == day_stem:
        score += 2.0

    score_i = _clamp_score(score)
    volatility = 3.0 + (6.0 if chong else 0.0)
    meta = {
        "day_elem": day_elem,
        "year_gz": year_gz,
        "dayun_gz": dayun_gz,
        "chong": bool(chong),
    }
    return score_i, volatility, meta


def _build_life_kline(
    *,
    bazi: BaziPro,
    max_age: int,
) -> dict:
    birth_year = int(bazi.birth_dt.year)
    years = [birth_year + age for age in range(0, max_age + 1)]

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
                "is_dayun_transition": year in dayun_transitions,
            }
        )

        if prev_close is not None:
            change_abs.append((year, abs(close - float(prev_close))))

        prev_close = close

    top_turns = sorted(change_abs, key=lambda x: x[1], reverse=True)[:8]
    return {
        "rows": rows,
        "birth_year": birth_year,
        "dayun_transitions": dayun_transitions,
        "top_turns": top_turns,
    }


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
    wealth_elem = KE.get(day_elem or "")
    controlled_by = _inverse_mapping(KE)
    career_elem = controlled_by.get(day_elem or "")

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

    def _score_against(target_elem: Optional[str], base: float, w: tuple[float, float, float, float]) -> float:
        if not target_elem:
            return base
        s = base
        s += _element_relation_score(y_stem_e, target_elem, w[0])
        s += _element_relation_score(y_zhi_e, target_elem, w[1])
        s += _element_relation_score(dy_stem_e, target_elem, w[2])
        s += _element_relation_score(dy_zhi_e, target_elem, w[3])
        if chong:
            s -= 6.0
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
    if chong:
        romance_base -= 4.0
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

    return {
        "year": year,
        "year_gz": year_gz,
        "dayun_gz": dayun_gz or "",
        "wealth": {"index": wealth_index, "vol": _vol(wealth_index), "prob": wealth_index},
        "career": {"index": career_index, "vol": _vol(career_index), "prob": career_index},
        "romance": {"index": romance_index, "vol": _vol(romance_index), "prob": romance_index},
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
                    high_rows = sorted(adult_rows, key=lambda r: float(r["close"]), reverse=True)[:6]
                    low_rows = sorted(adult_rows, key=lambda r: float(r["close"]))[:6]
                    if high_rows:
                        highlight_years_hint = "、".join(str(r["year"]) for r in high_rows[:5])
                    if low_rows:
                        low_years_hint = "、".join(str(r["year"]) for r in low_rows[:5])

                life_avg_score = None
                life_open_close_keyword = None
                if life:
                    rows = life.get("rows") or []
                    adult_rows = [r for r in rows if int(r.get("age", 0)) >= 18]
                    base_rows = adult_rows or rows
                    if base_rows:
                        life_avg_score = sum(float(r.get("close", 0.0)) for r in base_rows) / float(len(base_rows))
                    life_open_close_keyword = _life_open_close_keyword(rows)

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
                            f"- {m['year']}（{m['year_gz']}）：财运 {m['wealth']['index']}±{m['wealth']['vol']:.1f}（{m['wealth']['prob']}%）｜事业 {m['career']['index']}±{m['career']['vol']:.1f}（{m['career']['prob']}%）｜桃花 {m['romance']['index']}±{m['romance']['vol']:.1f}（{m['romance']['prob']}%）"
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

                # --- 核心：更自然、更具体的写作风格 ---
                final_prompt = f"""
                请你以“掌纹解读顾问”的口吻写一份中文报告：表达自然、克制、具体。
                目标：让读者读完能明白“我现在的状态是什么、为什么会这样、接下来怎么做”。

                写作要求：
                1) 不要提及“模型/提示词/系统/参数/Token/置信度”等字眼。
                2) 不要堆砌玄乎的形容词；尽量给可执行建议（例如作息/压力管理/沟通方式/理财习惯）。
                3) 如果照片不清晰导致某条线无法判断，请直接说明“看不清”，并给出重拍建议（补光、对焦、角度、掌心占比）。
                4) 用 Markdown 输出，标题使用 `###`，列表使用 `-`，避免大段连续长文本。
                5) 避免模板化：必须把“八字信息”里的干支细节融入分析（至少点到 4 个不同的干/支），不要给所有人写一样的套话。
                6) 情感与关系部分不要默认异性恋，不要用“男女固定搭配”的叙述；以“你偏好/你更容易被哪类人吸引”来表达即可。
                7) 可以有话直说：直接指出问题和代价，但不要人身攻击，不要恐吓。

                **【档案数据】**
                - 性别（可选）：{gender_for_display or "未填写"}
                - 关系偏好（可选）：{relationship_for_display or "不填写/不设限"}
                - 八字原局：{bazi_res['四柱']} (日主：{bazi_res['日主']})
                - 当前流年：{bazi_res['流年']}
                - 当前公历年份：{current_year}（明年={future_years[0]}）
                - 未来三年（固定）：{future_years[0]}、{future_years[1]}、{future_years[2]}（文中如果写“明年/后年/第三年”，必须严格对应这三年，不要写成其他年份）
                - 可能波动较大的年份（模型参考）：{kline_hint or "未计算/无"}
                - 可能的高光年份（模型参考）：{highlight_years_hint or "未计算/无"}
                - 可能的低谷年份（模型参考）：{low_years_hint or "未计算/无"}

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
                下面每行格式：指数±波动（发生概率%），请在事业/财运/关系的解释里用上。
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

                ### 整体概览（先天 vs 后天）
                - 用 5-8 句话概括你看到的“先天底色”和“后天变化”，以及这对当下状态的影响。
                - 给出 3 个最关键的“变化点”（不要写证据链，但要具体）。

                ### 性格重点
                - 结合八字日主 {bazi_res['日主']} + 掌纹给出的整体气质，指出你最核心的矛盾点。
                - 直接点出你最常见的一种“伪装/防御机制”，以及它的代价。

                ### 事业与财富（{future_range}）
                请按“天地人”给出标签与倾向判断：

                #### 财运：天才/地才/人才
                - 天才（运气/机会/贵人/窗口期）：用【强/中/弱】标注，并解释原因。
                - 地才（资源/平台/家底/人脉/行业红利）：用【强/中/弱】标注，并解释原因。
                - 人才（执行力/耐力/学习力/抗压/复盘）：用【强/中/弱】标注，并解释原因。
                - 给一个综合结论：你更偏向靠“天才/地才/人才”哪一个来起势？另两个是助推还是拖累？

                #### 事业运（请直接分析，不要用“天业/地业/人业”术语）
                - 用 6-10 句话给出“事业主线判断”：更适合稳定晋升/项目型成长/创业/自由职业/技术深耕/销售BD/内容传播 等哪类路径，并说明原因。
                - 必须回到盘面：引用至少 3 个具体锚点（例如十神/五行偏向、天乙贵人位置、驿马、华盖、大运与未来三年趋势中的某一年等）。
                - 给 2 个“更容易上升的年份”+ 2 个“更容易卡住的年份”（优先从上面提供的高光/低谷候选里选），并给对应的做法建议。

                #### 横财倾向
                - 用 0-100% 给出“横财机会倾向”的概率（模型化画像，不是保证）。
                - 说明它更可能来自：机会性副业/投机型收益/信息差/贵人机会/资产配置 哪一类；并给 1 条风控底线。

                ### 情感与关系（不设限）
                请输出“依恋类型概率画像”（仅作自我观察，不是心理诊断；不限定对方性别）：
                - 安全型：__%
                - 焦虑型：__%
                - 回避型：__%
                - 恐惧-回避型：__%
                要求：百分比必须为整数且总和=100，先给概率，再给解释。

                解释部分要包含：
                - 你判断概率的关键依据（结合八字与掌纹的整体气质，点名至少 2 个盘面细节/标记）。
                - 你在亲密关系里更容易触发的 1 个“自动反应”（例如追确认/冷处理/过度理性/拯救倾向等）。
                - 你更适合的关系类型（2 条）+ 最容易踩的雷区（2 条）。

                ### 人生亮点与低谷（年份提示）
                - 用 5-8 句话解释：为什么你可能在“高光年份候选”更容易出成果？需要抓住什么？
                - 用 5-8 句话解释：为什么你可能在“低谷年份候选”更容易消耗？需要提前避开什么？

                ### 一句话总结
                - 用一句话概括命主当前主线（直白但克制）。

                ### 接下来怎么做
                - 给 3 条具体、可执行的建议，尽量贴合上面分析（不要空话）。

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
                            high_rows = sorted(adult_rows, key=lambda r: float(r["close"]), reverse=True)[:6]
                            low_rows = sorted(adult_rows, key=lambda r: float(r["close"]))[:6]

                            if high_rows:
                                with st.expander("高光年份（模型参考）", expanded=False):
                                    st.table(
                                        [
                                            {"年份": r["year"], "年龄": r["age"], "指数": round(float(r["close"]), 1)}
                                            for r in high_rows
                                        ]
                                    )
                            if low_rows:
                                with st.expander("低谷年份（模型参考）", expanded=False):
                                    st.table(
                                        [
                                            {"年份": r["year"], "年龄": r["age"], "指数": round(float(r["close"]), 1)}
                                            for r in low_rows
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
