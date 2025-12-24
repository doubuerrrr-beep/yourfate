import streamlit as st
import PIL.Image
from google import genai
import os
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
    </style>
    """, unsafe_allow_html=True)

st.title("掌纹解读报告")
st.caption("填写生辰信息并上传左右手照片，生成一份更具体的解读（仅供参考）。")

with st.sidebar:
    st.header("档案录入")
    birth_date = st.date_input("出生日期", value=date(1991, 5, 21))
    birth_time = st.time_input("出生时辰", value=time(8, 15))
    gender = st.radio("生理性别", ("男", "女"))
    # 移除原本的单选，改为下方双上传
    
    st.markdown("---")
    st.info("提示：解读内容基于你提供的信息与照片生成，仅供参考，不构成医疗/法律/投资建议。")
    high_precision_mode = st.checkbox("更稳模式（更慢）", value=True)
    attach_enhanced_images = st.checkbox("启用纹路增强（更慢）", value=True)
    rich_output_mode = st.checkbox("详细报告（更长更具体）", value=True)

# 双列布局上传
st.markdown("请分别上传左手和右手的高清照片，系统将执行**【先天基因 vs 后天变数】**的差分比对。")

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
        bazi_res = get_full_bazi_engine(birth_date, birth_time, gender)
        
        if not bazi_res:
            st.error("八字计算失败（返回 None）：请确认出生日期/时辰有效，并检查 `borax` 依赖是否安装正常。")
            st.stop()

        if bazi_res:
            with st.spinner("正在生成解读..."):
                
                # 判定先天与后天
                if gender == "男":
                    xian_tian = "左手"
                    hou_tian = "右手"
                else:
                    xian_tian = "右手"
                    hou_tian = "左手"

                left_qm = _image_quality_metrics(img_left)
                right_qm = _image_quality_metrics(img_right)
                image_order_desc = "1) 左手原图；2) 右手原图。"
                if attach_enhanced_images:
                    image_order_desc += "\\n3) 左手增强；4) 右手增强；5) 左手边缘；6) 右手边缘。"

                left_score = _clarity_score(left_qm)
                right_score = _clarity_score(right_qm)

                target_length = "约 1800-2600 字" if rich_output_mode else "约 900-1400 字"

                # --- 核心：更自然、更具体的写作风格 ---
                final_prompt = f"""
                请你以“掌纹解读顾问”的口吻写一份中文报告：表达自然、克制、具体。
                目标：让读者读完能明白“我现在的状态是什么、为什么会这样、接下来怎么做”。

                写作要求：
                1) 不要提及“模型/提示词/系统/参数/Token/置信度”等字眼。
                2) 不要堆砌玄乎的形容词；尽量给可执行建议（例如作息/压力管理/沟通方式/理财习惯）。
                3) 如果照片不清晰导致某条线无法判断，请直接说明“看不清”，并给出重拍建议（补光、对焦、角度、掌心占比）。

                **【档案数据】**
                - 性别：{gender}
                - 八字原局：{bazi_res['四柱']} (日主：{bazi_res['日主']})
                - 当前流年：{bazi_res['流年']}

                **【照片清晰度评分（仅供参考）】**
                - 左手：{left_score}/100（{left_qm['width']}×{left_qm['height']}）
                - 右手：{right_score}/100（{right_qm['width']}×{right_qm['height']}）

                **【样本定义】**
                你会收到图片（按传入顺序）：
                {image_order_desc}

                根据“男左女右”为先天的定律：
                - 你的{xian_tian}代表【先天命格】（基因、祖荫、底牌）。
                - 你的{hou_tian}代表【后天运势】（作为、环境、变数）。

                **【输出结构 - 请逐一输出】**

                **第一部分：整体概览（先天 vs 后天）**
                - 用 5-8 句话概括你看到的“先天底色”和“后天变化”，以及这对当下状态的影响。
                - 给出 3 个最关键的“变化点”（不要写证据链，但要具体）。

                **第二部分：性格的核心矛盾与伪装**
                - 结合八字日主 {bazi_res['日主']} + 掌纹给出的整体气质，指出你最核心的矛盾点。
                - 直接点出你最常见的一种“伪装/防御机制”，以及它的代价。

                **第三部分：事业/财富病理与未来三年趋势（2026-2028）**
                - 给一个“赚钱方式画像”：靠拼命、靠资源、靠运气、靠认知差，哪一个更像你。
                - 提醒 1-2 个最可能的破财诱因（投资/关系/健康/冲动消费等），并给对应的对冲策略。

                **第四部分：情感刑侦（关系模式）**
                - 你在亲密关系里更像哪种模式（控制/回避/拯救/依赖/理性切割等）。
                - 给出“适合的关系类型”与“最不适合的雷区”。

                **第五部分：最终判决（3 句话）**
                - 现状一句话。
                - 未来三年必须斩断的一种关系/习惯。
                - 未来三年唯一的翻盘机会点（要具体、可执行）。

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

                    gen_config = genai.types.GenerateContentConfig(
                        temperature=0.2 if high_precision_mode else 0.35,
                        topP=0.9,
                        maxOutputTokens=8192 if rich_output_mode else 4096,
                        seed=42,
                    )
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
                        st.stop()
                     
                    st.markdown("---")
                    # 自定义显示的排盘信息
                    st.markdown(f"""
                    <div class="report-box">
                        <div class="bazi-row">
                            <span>📅 {birth_date}</span>
                            <span>🧬 {bazi_res['四柱']}</span>
                            <span>⚖️ 先天：{xian_tian} | 后天：{hou_tian}</span>
                        </div>
                        {report_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"分析中断: {str(e)}")
                    st.caption("提示：请检查网络连接与 API Key 配置。")
