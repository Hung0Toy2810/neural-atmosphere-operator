"""Build the Stage 1/Stage 2 report, including measured training histories."""

from __future__ import annotations

import csv
import math
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches

from review_report_slides import repair_and_enrich
from report_wording import correct_figure_terms

from build_report_slides import (
    BLUE,
    BLUE_2,
    GRAY,
    GREEN,
    GREEN_PALE,
    MID,
    NAVY,
    ORANGE,
    PALE,
    PALE_2,
    RED,
    RED_PALE,
    WHITE,
    add_bar,
    add_bullets,
    clear_slides,
    connector,
    metric_card,
    process_box,
    rgb,
    rounded_box,
    set_fill,
    text_box,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT.parent / "LuongNguyenThanhHung_final.pptx"
OUTPUT = ROOT / "reports" / "NeuralAtmosphereOperator_Full_Report.pptx"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


SUMMARY = {row["stage"]: row for row in load_rows(ROOT / "results/stage_smoke_tests/summary.csv")}
PRIORITY_ROWS = load_rows(ROOT / "results/stage_smoke_tests/priority_stage_metrics.csv")
ALL_ROWS = load_rows(ROOT / "results/stage_smoke_tests/all_stage_metrics.csv")


def add_brand(slide, logo: Path, title: str, number: int, subtitle: str | None = None) -> None:
    slide.shapes.add_picture(str(logo), Inches(0.35), Inches(0.2), height=Inches(0.48))
    line = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE,
        Inches(0.32),
        Inches(0.78),
        Inches(12.68),
        Inches(0.045),
    )
    set_fill(line, BLUE)
    line.line.fill.background()
    text_box(slide, title, 1.03, 0.2, 10.55, 0.4, size=23, color=NAVY, bold=True)
    text_box(slide, f"{number:02d}", 12.28, 0.28, 0.42, 0.2, size=9, color=BLUE, bold=True, align=PP_ALIGN.RIGHT)
    if subtitle:
        text_box(slide, subtitle, 1.05, 0.57, 10.8, 0.18, size=8.5, color=GRAY)
    text_box(slide, "Lương Nguyễn Thành Hưng · NeuralAtmosphereOperator", 0.4, 7.2, 5.2, 0.16, size=6.2, color="8794A1")
    text_box(slide, "Stage 1 → Stage 2", 11.3, 7.2, 1.45, 0.16, size=6.2, color="8794A1", align=PP_ALIGN.RIGHT)


def section_slide(prs, logo: Path, number: int, section: str, title: str, subtitle: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(NAVY)
    slide.shapes.add_picture(str(logo), Inches(0.62), Inches(0.55), height=Inches(0.72))
    line = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0.7), Inches(2.05), Inches(1.6), Inches(0.06))
    set_fill(line, BLUE_2)
    line.line.fill.background()
    text_box(slide, section, 0.72, 1.5, 3.2, 0.38, size=15, color=BLUE_2, bold=True)
    text_box(slide, title, 0.72, 2.35, 11.6, 0.8, size=34, color=WHITE, bold=True)
    text_box(slide, subtitle, 0.75, 3.38, 10.6, 0.56, size=17, color="D8E7F4")
    text_box(slide, f"{number:02d}", 11.8, 6.7, 0.75, 0.32, size=15, color=BLUE_2, bold=True, align=PP_ALIGN.RIGHT)
    return slide


def standard_slide(prs, logo: Path, number: int, title: str, subtitle: str | None = None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_brand(slide, logo, title, number, subtitle)
    return slide


def card(slide, title: str, body: str, x, y, w, h, *, accent=BLUE, fill=PALE_2):
    rounded_box(slide, x, y, w, h, fill=fill, line="D3DEE8")
    top = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.07))
    set_fill(top, accent)
    top.line.fill.background()
    text_box(slide, title, x + 0.2, y + 0.17, w - 0.4, 0.47, size=16, color=accent, bold=True)
    text_box(slide, body, x + 0.2, y + 0.85, w - 0.4, max(0.4,h - 0.97), size=15, color=NAVY, valign=MSO_ANCHOR.TOP)


def equation_png(folder: Path, name: str, latex: str, *, fontsize: int = 28, color=NAVY) -> Path:
    folder = ROOT / 'reports' / 'equations'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    with plt.rc_context({'mathtext.fontset': 'cm', 'svg.fonttype': 'path'}):
        fig = plt.figure(figsize=(11, 1.6), dpi=500)
        fig.patch.set_alpha(0)
        fig.text(0.5, 0.5, latex, ha="center", va="center", fontsize=fontsize, color=f"#{color}")
        fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.14)
        fig.savefig(path.with_suffix('.svg'), transparent=True, bbox_inches="tight", pad_inches=0.14)
        fig.savefig(path.with_suffix('.pdf'), transparent=True, bbox_inches="tight", pad_inches=0.14)
        plt.close(fig)
    return path


def save_chart(fig, folder: Path, name: str) -> Path:
    correct_figure_terms(fig)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A9B7C6")
    ax.tick_params(colors="#455564", labelsize=9)
    ax.grid(axis="y", color="#E5EBF1", linewidth=0.8)
    ax.set_axisbelow(True)


def charts(folder: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    s1, s2 = SUMMARY["stage1"], SUMMARY["stage2"]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    labels = ["Validation loss", "Objective loss", "Normalized RMSE"]
    a = [0.13444305187815098, float(s1["objective_loss"]), float(s1["normalized_rmse"])]
    b = [0.12538444922294145, float(s2["objective_loss"]), float(s2["normalized_rmse"])]
    x = np.arange(len(labels)); width = 0.33
    ax.bar(x - width / 2, a, width, color="#A9B7C6", label="Stage 1")
    ax.bar(x + width / 2, b, width, color="#2E8B57", label="Stage 2")
    for i, (v1, v2) in enumerate(zip(a, b)):
        ax.text(i - width / 2, v1 + 0.002, f"{v1:.4f}", ha="center", fontsize=9)
        ax.text(i + width / 2, v2 + 0.002, f"{v2:.4f}", ha="center", fontsize=9, color="#19663D", fontweight="bold")
    ax.set_xticks(x, labels); ax.set_ylim(0, 0.185); ax.legend(frameon=False, ncol=2)
    ax.set_title("Sai số tổng hợp tại lead 6 giờ", loc="left", fontsize=13, fontweight="bold", color="#20384F")
    style_axes(ax)
    out["aggregate"] = save_chart(fig, folder, "aggregate")

    priority = defaultdict(dict)
    for row in PRIORITY_ROWS:
        priority[row["variable"]][row["stage"]] = row
    names = ["Z500", "T850", "T2M", "U10", "V10", "MSLP", "Q850", "U850", "V850"]
    improvements = []
    for name in names:
        v1 = float(priority[name]["stage1"]["rmse"])
        v2 = float(priority[name]["stage2"]["rmse"])
        improvements.append(100 * (v1 - v2) / v1)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bars = ax.barh(names[::-1], improvements[::-1], color=["#2E8B57" if x >= 2 else "#1976D2" for x in improvements[::-1]])
    for bar, value in zip(bars, improvements[::-1]):
        ax.text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(improvements) * 1.22); ax.set_xlabel("RMSE reduction (%)")
    ax.set_title("Stage 2 cải thiện trên 9 biến ưu tiên", loc="left", fontsize=13, fontweight="bold", color="#20384F")
    style_axes(ax)
    out["priority"] = save_chart(fig, folder, "priority")

    paired: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in ALL_ROWS:
        paired[(row["variable"], row["lead_hours"])][row["stage"]] = row
    changes = []
    labels71 = []
    groups = []
    for (variable, _lead), stages in paired.items():
        if "stage1" not in stages or "stage2" not in stages:
            continue
        v1 = float(stages["stage1"]["rmse"]); v2 = float(stages["stage2"]["rmse"])
        changes.append(100 * (v1 - v2) / v1)
        labels71.append(variable)
        if "@" not in variable:
            groups.append("Surface")
        else:
            base = variable.split("@", 1)[0]
            groups.append({
                "u_component_of_wind": "U wind",
                "v_component_of_wind": "V wind",
                "geopotential": "Geopotential",
                "temperature": "Temperature",
                "specific_humidity": "Humidity",
            }.get(base, base))
    order = np.argsort(changes)
    sorted_changes = np.asarray(changes)[order]
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    colors = ["#D94C4C" if v < 0 else "#2E8B57" for v in sorted_changes]
    ax.bar(np.arange(len(sorted_changes)), sorted_changes, color=colors, width=0.82)
    ax.axhline(0, color="#20384F", linewidth=1)
    ax.set_xlabel("71 channels sorted by RMSE change"); ax.set_ylabel("Reduction (%)")
    ax.set_title("Phân bố thay đổi RMSE trên toàn bộ channel", loc="left", fontsize=13, fontweight="bold", color="#20384F")
    style_axes(ax)
    out["all_channels"] = save_chart(fig, folder, "all_channels")

    grouped: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, changes): grouped[group].append(value)
    group_names = ["Surface", "U wind", "V wind", "Geopotential", "Temperature", "Humidity"]
    medians = [float(np.median(grouped[name])) for name in group_names]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    bars = ax.bar(group_names, medians, color="#1976D2")
    for bar, value in zip(bars, medians):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.06, f"{value:.2f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Median RMSE reduction (%)"); ax.tick_params(axis="x", rotation=15)
    ax.set_title("Cải thiện trung vị theo nhóm biến", loc="left", fontsize=13, fontweight="bold", color="#20384F")
    style_axes(ax)
    out["groups"] = save_chart(fig, folder, "groups")

    acc_delta = []
    variance_distance_1, variance_distance_2 = [], []
    for name in names:
        r1, r2 = priority[name]["stage1"], priority[name]["stage2"]
        acc_delta.append(1000 * (float(r2["acc"]) - float(r1["acc"])))
        variance_distance_1.append(abs(float(r1["spatial_variance_ratio"]) - 1))
        variance_distance_2.append(abs(float(r2["spatial_variance_ratio"]) - 1))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].bar(names, acc_delta, color=["#2E8B57" if v >= 0 else "#D94C4C" for v in acc_delta])
    axes[0].axhline(0, color="#20384F", linewidth=1); axes[0].set_ylabel("ΔACC × 10³")
    axes[0].set_title("Thay đổi ACC", loc="left", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45); style_axes(axes[0])
    x = np.arange(len(names)); width=.35
    axes[1].bar(x-width/2, variance_distance_1, width, color="#A9B7C6", label="Stage 1")
    axes[1].bar(x+width/2, variance_distance_2, width, color="#2E8B57", label="Stage 2")
    axes[1].set_xticks(x, names, rotation=45); axes[1].set_ylabel("|variance ratio − 1|")
    axes[1].set_title("Độ lệch phương sai", loc="left", fontsize=12, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8); style_axes(axes[1])
    out["acc_variance"] = save_chart(fig, folder, "acc_variance")

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    k = np.arange(1, 65)
    energy = k ** -1.35
    retained = energy.copy(); retained[k > 30] = np.nan
    ax.loglog(k, energy, color="#A9B7C6", linewidth=2, label="Phổ tín hiệu minh họa")
    ax.fill_between(k, energy, 1e-4, where=(k <= 30).tolist(), color="#1976D2", alpha=.18, label="Mode được giữ")
    ax.fill_between(k, energy, 1e-4, where=(k > 30).tolist(), color="#D94C4C", alpha=.2, label="Chi tiết bị loại")
    ax.axvline(30, color="#D94C4C", linestyle="--", linewidth=1.5)
    ax.text(31, .02, "spectral cutoff", color="#D94C4C", fontsize=9)
    ax.set_xlabel("Spherical wavenumber / mode"); ax.set_ylabel("Energy (illustrative)")
    ax.set_ylim(1e-4, 1.2); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Minh họa nút thắt tần số cao", loc="left", fontsize=13, fontweight="bold", color="#20384F")
    style_axes(ax)
    out["spectrum"] = save_chart(fig, folder, "spectrum")
    return out


def add_picture(slide, path: Path, x, y, w, h=None):
    with Image.open(path) as source:
        ratio = source.width / source.height
        native_width = source.width / 500
    if path.name.startswith('eq_'):
        # Preserve the typesetter's 28–32 pt size; never enlarge to fill a box.
        allocation_width = w
        w = min(w, native_width)
        x += (allocation_width - w) / 2
        h = 1.48 if h is None else h
    if h is not None:
        actual_w = min(w, h * ratio)
        actual_h = actual_w / ratio
        x += (w - actual_w) / 2
        y += (h - actual_h) / 2
        w = actual_w
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(actual_h))
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))


def two_column_bullets(slide, left_title, left_items, right_title, right_items, *, left_color=BLUE, right_color=GREEN):
    preexisting = [shape for shape in slide.shapes if Inches(1.0) < shape.top < Inches(6.8)]
    y = 3.0 if preexisting else 1.45
    h = 3.48 if preexisting else 4.72
    for x,title,items,color in [(0.55,left_title,left_items,left_color),(6.82,right_title,right_items,right_color)]:
        rounded_box(slide,x,y,5.95,h,fill=PALE_2,line="D2DEE8")
        text_box(slide,title,x+0.32,y+0.25,5.28,0.48,size=20,color=color,bold=True)
        add_bullets(slide,items,x+0.32,y+0.96,5.1,h-1.1,size=16 if preexisting else 18,spacing=10 if preexisting else 15)


def build() -> Path:
    if not TEMPLATE.is_file(): raise FileNotFoundError(TEMPLATE)
    with tempfile.TemporaryDirectory(prefix="nao-full-slides-") as temp_name:
        temp = Path(temp_name)
        with zipfile.ZipFile(TEMPLATE) as archive:
            logo = temp / "hcmute.png"
            logo.write_bytes(archive.read("ppt/media/image1.png"))
        chart_paths = charts(temp)
        eq_operator = equation_png(temp, "eq_operator", r"$\mathcal{G}_{\theta}:\; x(t) \mapsto \hat{x}(t+\Delta t)$", fontsize=32)
        eq_sht = equation_png(temp, "eq_sht", r"$f(\theta,\phi)=\sum_{\ell=0}^{L}\;\;\sum_{m=-\ell}^{\ell}\hat{f}_{\ell m}\,Y_{\ell}^{m}(\theta,\phi)$", fontsize=29)
        eq_layer = equation_png(temp, "eq_layer", r"$v_{k+1}=\sigma\!\left(Wv_k+\mathcal{F}^{-1}_{S^2}\!\left(R_k\,\mathcal{F}_{S^2}(v_k)\right)\right)$", fontsize=27)
        eq_tendency = equation_png(temp, "eq_tendency", r"$\hat{x}_{t+1}=x_t+\frac{\sigma(\Delta x)}{\sigma(x)}\odot\hat{z}_{t+1}$", fontsize=29)
        eq_loss = equation_png(temp, "eq_loss", r"$\mathcal{L}=\sum_c w_c\;\mathbb{E}_{A}\!\left[\left(\frac{\hat{x}_c-x_c}{\sigma(\Delta x_c)/\sigma(x_c)}\right)^2\right]$", fontsize=25)
        eq_stage2 = equation_png(temp, "eq_stage2", r"$\hat{z}^{(2)}=\hat{z}^{(1)}+C_{\phi}(x_t,\,\Delta x_t,\,\mathrm{stopgrad}(\hat{z}^{(1)}),\,f_t)$", fontsize=25)

        prs = Presentation(str(TEMPLATE)); clear_slides(prs); blank = prs.slide_layouts[6]
        n = 0
        def std(title, subtitle=None):
            nonlocal n; n += 1; return standard_slide(prs, logo, n, title, subtitle)
        def sec(label, title, subtitle):
            nonlocal n; n += 1; return section_slide(prs, logo, n, label, title, subtitle)

        # 1 Cover
        n += 1; slide = prs.slides.add_slide(blank)
        accent = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(.25), Inches(0), Inches(.12), Inches(7.5)); set_fill(accent, BLUE); accent.line.fill.background()
        slide.shapes.add_picture(str(logo), Inches(.72), Inches(.52), height=Inches(.86))
        text_box(slide, "TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT TP. HỒ CHÍ MINH", 1.72, .66, 8.5, .24, size=10.5, color=GRAY, bold=True)
        text_box(slide, "NEURAL ATMOSPHERE OPERATOR", .72, 1.8, 11.7, .58, size=31, color=BLUE, bold=True)
        text_box(slide, "Dự báo khí quyển toàn cầu bằng Spherical Fourier Neural Operator", .72, 2.52, 11.2, .5, size=20, color=NAVY, bold=True)
        rounded_box(slide, .72, 3.4, 4.5, .56, fill=PALE, line="C8D7E4"); text_box(slide, "STAGE 1 · SFNO BASELINE", .9, 3.58, 4.1, .2, size=12.5, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
        rounded_box(slide, 5.5, 3.4, 5.5, .56, fill=GREEN_PALE, line="B8D8C5"); text_box(slide, "STAGE 2 · SPHERICAL RESIDUAL CORRECTOR", 5.68, 3.58, 5.1, .2, size=12.5, color=GREEN, bold=True, align=PP_ALIGN.CENTER)
        text_box(slide, "Sinh viên: Lương Nguyễn Thành Hưng · MSSV: 23110111", .76, 5.55, 8.5, .28, size=13, color=NAVY)
        text_box(slide, "TP. Hồ Chí Minh, 2026", .76, 6.05, 4.0, .2, size=10, color=GRAY)
        text_box(slide, "01", 12.15, 6.85, .48, .2, size=9, color=BLUE, bold=True, align=PP_ALIGN.RIGHT)

        # 2 Outline
        slide = std("Nội dung trình bày")
        sections = [("01", "Bài toán & dữ liệu", "ERA5 · 71 channels · protocol"), ("02", "Cơ sở phương pháp", "Neural operator · SHT · loss"), ("03", "Stage 1", "SFNO baseline · kết quả · bão hòa"), ("04", "Stage 2", "Forcing · corrector · fine-tuning"), ("05", "Đánh giá & thảo luận", "71 channels · giới hạn · hướng phát triển")]
        for i,(idx,title,sub) in enumerate(sections):
            y=1.22+i*1.05; rounded_box(slide,.7,y,11.9,.82,fill=WHITE,line="D2DEE8")
            rounded_box(slide,.9,y+.13,.72,.54,fill=BLUE if i<2 else GREEN,line=BLUE if i<2 else GREEN)
            text_box(slide,idx,1.02,y+.3,.48,.18,size=11,color=WHITE,bold=True,align=PP_ALIGN.CENTER)
            text_box(slide,title,1.9,y+.16,3.65,.24,size=15,color=NAVY,bold=True)
            text_box(slide,sub,5.55,y+.18,6.4,.22,size=12,color=GRAY)

        slide = std("Động lực nghiên cứu", "Dự báo khí quyển là bài toán động lực học đa thang trên mặt cầu")
        two_column_bullets(slide,"Nhu cầu",["Dự báo nhanh trên lưới toàn cầu","Thay thế một phần chi phí của mô hình số","Học trực tiếp ánh xạ giữa các trường khí quyển","Khai thác GPU cho suy luận song song"],"Thách thức",["71 biến với thang đo rất khác nhau","Động lực đa thang từ hành tinh đến cục bộ","Cực trị hiếm nhưng có ý nghĩa lớn","Lưới lat–lon biến dạng tại hai cực"],right_color=RED)

        slide = std("Phát biểu bài toán")
        add_picture(slide,eq_operator,1.15,1.35,11.0)
        for i,(title,body,color) in enumerate([("Input","ERA5 x(t), hoặc [x(t−6h), x(t)]",BLUE),("Output","71 trường dự báo tại t+6h",GREEN),("Miền","S² × channel space",ORANGE)]):
            card(slide,title,body,.75+i*4.15,3.1,3.7,1.45,accent=color)
        rounded_box(slide,.75,5.1,12.0,1.12,fill=PALE,line="CAD8E5")
        text_box(slide,"Mục tiêu tối ưu",1.0,5.39,1.7,.22,size=14,color=BLUE,bold=True)
        text_box(slide,"Sai số 6 giờ thấp · bảo toàn cấu trúc không gian · suy luận ổn định · có thể mở rộng autoregressive",2.62,5.32,9.55,.36,size=14,color=NAVY,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Mục tiêu nghiên cứu")
        goals=[("G1","Xây dựng SFNO baseline đáng tin cậy"),("G2","Chuẩn hóa loss theo tendency và diện tích"),("G3","Bổ sung history + forcing có ý nghĩa vật lý"),("G4","Học residual error bằng spherical corrector"),("G5","Đánh giá công bằng trên checkpoint validation-best"),("G6","Phân tích giới hạn tần số cao và cực trị")]
        for i,(tag,body) in enumerate(goals):
            x=.65+(i%2)*6.15; y=1.2+(i//2)*1.72
            rounded_box(slide,x,y,5.8,1.34,fill=PALE_2,line="D2DEE8")
            rounded_box(slide,x+.22,y+.25,.7,.7,fill=BLUE if i<3 else GREEN,line=BLUE if i<3 else GREEN)
            text_box(slide,tag,x+.3,y+.47,.54,.18,size=11,color=WHITE,bold=True,align=PP_ALIGN.CENTER)
            text_box(slide,body,x+1.15,y+.3,4.25,.58,size=15,color=NAVY,bold=True,valign=MSO_ANCHOR.MIDDLE)

        slide = std("Đóng góp của đồ án")
        contributions=[("Pipeline dữ liệu","ERA5 0,5° · 71 channels · split theo thời gian"),("Baseline SFNO","E192/L6 · standardized tendency prediction"),("Stage 2","Full-resolution spherical residual corrector"),("Fine-tuning","Freeze → progressive unfreeze với LR phân tầng"),("Evaluation","RMSE · ACC · bias · variance ratio · 71 channels"),("Reproducibility","Checkpoint, manifest, evaluator và audit độc lập")]
        for i,(title,body) in enumerate(contributions):
            x=.62+(i%3)*4.12; y=1.25+(i//3)*2.5; card(slide,title,body,x,y,3.78,1.95,accent=BLUE if i<3 else GREEN)

        slide = std("Pipeline end-to-end")
        labels=[("WeatherBench2","ERA5 0,25°"),("Preprocess","regrid + stats"),("Stage 1","SFNO baseline"),("Stage 2","residual corrector"),("Evaluation","July 2020")]
        for i,(a,b) in enumerate(labels):
            x=.45+i*2.57; process_box(slide,a,b,x,2.25,2.05,1.15,fill=GREEN_PALE if i==3 else PALE,line=GREEN if i==3 else BLUE)
            if i<4: connector(slide,x+2.08,2.82,x+2.5,2.82)
        rounded_box(slide,.7,4.25,11.9,1.45,fill=PALE_2,line="D2DEE8")
        text_box(slide,"Nguyên tắc kiểm soát leakage",1.0,4.58,2.85,.28,size=16,color=RED,bold=True)
        text_box(slide,"Statistics: train only  ·  Checkpoint: validation only  ·  July 2020: evaluation only",3.75,4.53,8.1,.34,size=15,color=NAVY,bold=True,align=PP_ALIGN.CENTER)

        sec("PHẦN I", "Dữ liệu và giao thức đánh giá", "Một so sánh model chỉ đáng tin khi dữ liệu, split và metric được khóa chặt.")

        slide = std("Nguồn dữ liệu ERA5 / WeatherBench2")
        cards=[("Nguồn","Public WB2 ERA5 archive"),("Lưới nguồn","0,25° · 721 × 1440"),("Lưới đích","0,5° · 361 × 720"),("Cadence","6 giờ"),("State","71 channels"),("Thời gian","1983–2020")]
        for i,(a,b) in enumerate(cards):
            x=.65+(i%3)*4.12; y=1.25+(i//3)*2.28; metric_card(slide,a,b,"",x,y,3.75,1.55,accent=BLUE if i<3 else GREEN)
        text_box(slide,"Dữ liệu test không được dùng để chọn kiến trúc hoặc checkpoint.",1.1,6.02,11.1,.3,size=15,color=RED,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Hợp đồng 71 channels")
        groups=[("Surface",6,"U10, V10, T2M, SP, MSLP, TCWV",BLUE),("U wind",13,"50 → 1000 hPa",GREEN),("V wind",13,"50 → 1000 hPa",GREEN),("Geopotential",13,"50 → 1000 hPa",ORANGE),("Temperature",13,"50 → 1000 hPa",RED),("Specific humidity",13,"50 → 1000 hPa",BLUE_2)]
        for i,(name,count,note,color) in enumerate(groups):
            x=.6+(i%3)*4.15; y=1.2+(i//3)*2.52
            rounded_box(slide,x,y,3.8,2.0,fill=PALE_2,line="D2DEE8")
            text_box(slide,str(count),x+.2,y+.32,1.0,.48,size=28,color=color,bold=True,align=PP_ALIGN.CENTER)
            text_box(slide,name,x+1.22,y+.35,2.2,.32,size=16,color=NAVY,bold=True)
            text_box(slide,note,x+.3,y+1.18,3.2,.3,size=11,color=GRAY,align=PP_ALIGN.CENTER)

        slide = std("Chia dữ liệu theo thời gian")
        for x,w,label,years,note,color,fill in [(.65,6.7,"TRAIN","1983–2018","36 năm · fit model + statistics",BLUE,PALE),(7.45,1.85,"VALID","2019","model selection",GREEN,GREEN_PALE),(9.42,3.18,"TEST","07/2020","held-out evaluation",RED,RED_PALE)]:
            rounded_box(slide,x,2.05,w,1.5,fill=fill,line=color); text_box(slide,label,x+.1,2.3,w-.2,.22,size=11,color=color,bold=True,align=PP_ALIGN.CENTER); text_box(slide,years,x+.1,2.72,w-.2,.35,size=22,color=NAVY,bold=True,align=PP_ALIGN.CENTER); text_box(slide,note,x+.1,3.17,w-.2,.18,size=9,color=GRAY,align=PP_ALIGN.CENTER)
        rounded_box(slide,.65,4.35,11.95,1.45,fill=PALE_2,line="D4DFE9")
        add_bullets(slide,["Validation-best checkpoint được giữ riêng cho từng stage","Test July 2020 cách validation boundary hơn 6 tháng","95 initializations hợp lệ cho cửa sổ đánh giá 7 ngày"],1.0,4.67,10.9,.8,size=13.5,spacing=5)

        slide = std("Regridding và kiểm soát chất lượng")
        steps=[("0,25° source","WeatherBench2"),("Area overlap","bảo toàn bậc nhất"),("Periodic longitude","không seam 0°/360°"),("0,5° target","361 × 720")]
        for i,(a,b) in enumerate(steps):
            x=.65+i*3.08; process_box(slide,a,b,x,1.55,2.45,1.1,fill=PALE,line=BLUE)
            if i<3: connector(slide,x+2.48,2.1,x+3.0,2.1)
        two_column_bullets(slide,"Kiểm tra bắt buộc",["Channel order bất biến","Tọa độ latitude north → south","Giá trị finite","Contract hash khi resume"],"Đánh đổi",["0,5° giảm compute đáng kể","Nhưng Nyquist mới thấp hơn","Cấu trúc dưới grid scale biến mất","Không thể phục hồi bằng model"],right_color=RED)

        slide = std("Chuẩn hóa state và tendency")
        add_picture(slide,eq_tendency,.9,1.25,11.5)
        card(slide,"State normalization","x̃ = (x − μx) / σ(x)",.75,3.15,3.75,1.52,accent=BLUE)
        card(slide,"Tendency scale","s = σ(Δx) / σ(x)",4.78,3.15,3.75,1.52,accent=GREEN)
        card(slide,"Model target","z = Δx / σ(Δx)",8.8,3.15,3.75,1.52,accent=ORANGE)
        rounded_box(slide,.75,5.18,11.8,1.05,fill=PALE_2,line="D2DEE8")
        text_box(slide,"Tác dụng",1.0,5.48,1.25,.22,size=14,color=GREEN,bold=True)
        text_box(slide,"Đưa các biến về thang gradient tương đương và biến residual update thành phép biến đổi có ý nghĩa vật lý.",2.1,5.42,9.75,.34,size=13.5,color=NAVY,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Giao thức đánh giá")
        metrics=[("RMSE","sai số biên độ"),("ACC","tương quan anomaly"),("Bias","sai lệch hệ thống"),("Variance ratio","độ sắc nét / smoothing")]
        for i,(a,b) in enumerate(metrics): metric_card(slide,b,a,"latitude-area weighted",.65+i*3.08,1.3,2.72,1.6,accent=[BLUE,GREEN,ORANGE,RED][i])
        rounded_box(slide,.65,3.45,12.0,2.58,fill=PALE_2,line="D2DEE8")
        add_bullets(slide,["Lead chính trong checkpoint smoke test: 6 giờ (K=1)","Metrics được hoàn nguyên về đơn vị vật lý cho từng channel","Z500 đổi sang gpm; MSLP sang hPa; humidity sang g/kg","Persistence và training climatology được tính trên cùng protocol"],1.0,3.8,11.1,1.75,size=14.5,spacing=8)

        sec("PHẦN II", "Cơ sở phương pháp", "Từ neural operator đến SFNO, standardized tendency loss và residual learning.")

        slide = std("Neural Operator là gì?")
        add_picture(slide,eq_operator,1.1,1.2,11.0)
        two_column_bullets(slide,"Khác neural network thông thường",["Học ánh xạ giữa hai không gian hàm","Tham số dùng chung trên toàn miền","Có thể biểu diễn tương tác xa","Phù hợp PDE và trường vật lý"],"Trong đồ án",["Input/output là trường trên mặt cầu","Operator tiến hóa một bước 6 giờ","Autoregressive cho lead dài hơn","SFNO là parameterization được chọn"])

        slide = std("Vì sao phải tôn trọng hình học mặt cầu?")
        for i,(title,body,color) in enumerate([("Longitude","Tuần hoàn tại 0°/360°",BLUE),("Poles","Điểm lưới hội tụ",RED),("Area","Ô lưới có diện tích khác nhau",ORANGE)]): card(slide,title,body,.7+i*4.15,1.35,3.75,1.5,accent=color)
        rounded_box(slide,.7,3.35,12.0,2.6,fill=PALE_2,line="D2DEE8")
        text_box(slide,"Convolution phẳng",1.0,3.76,2.45,.3,size=18,color=RED,bold=True)
        text_box(slide,"padding cực · seam kinh tuyến · receptive field phụ thuộc latitude",3.15,3.72,8.75,.38,size=15,color=NAVY,bold=True,align=PP_ALIGN.CENTER)
        text_box(slide,"SFNO",1.0,4.66,2.45,.3,size=18,color=GREEN,bold=True)
        text_box(slide,"biểu diễn bằng spherical harmonics · mixing toàn cục · geometry-aware",3.15,4.62,8.75,.38,size=15,color=NAVY,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Spherical Harmonic Transform")
        add_picture(slide,eq_sht,.75,1.2,11.8)
        for i,(a,b) in enumerate([("θ, φ","vĩ độ / kinh độ"),("Yℓᵐ","spherical harmonic basis"),("f̂ℓm","hệ số phổ"),("L","spectral cutoff")]): metric_card(slide,b,a,"",.7+i*3.05,3.08,2.68,1.42,accent=[BLUE,GREEN,ORANGE,RED][i])
        rounded_box(slide,.7,5.02,11.9,1.12,fill=RED_PALE,line="EDBDBD")
        text_box(slide,"Điểm mấu chốt",.95,5.36,1.75,.24,size=14,color=RED,bold=True)
        text_box(slide,"Giảm L giúp train nhanh hơn, nhưng cũng cắt trực tiếp cấu trúc không gian nhỏ.",2.58,5.31,9.3,.32,size=14,color=NAVY,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Một block SFNO")
        add_picture(slide,eq_layer,.7,1.15,11.9)
        blocks=[("Input vₖ","feature map"),("SHT","grid → spectrum"),("Rₖ","spectral mixing"),("iSHT","spectrum → grid"),("Skip + MLP","local refinement"),("Output vₖ₊₁","next features")]
        for i,(a,b) in enumerate(blocks):
            x=.35+i*2.14; process_box(slide,a,b,x,3.0,1.68,1.05,fill=PALE if i<5 else GREEN_PALE,line=BLUE if i<5 else GREEN)
            if i<5: connector(slide,x+1.72,3.53,x+2.07,3.53)
        add_bullets(slide,["Spectral branch cung cấp receptive field toàn cầu","Residual/MLP branch giữ biến đổi cục bộ và ổn định gradient","6 blocks trong Stage 1; 4 blocks trong Stage 2 corrector"],1.05,4.65,11.0,1.1,size=14,spacing=7)

        slide = std("Residual tendency prediction")
        add_picture(slide,eq_tendency,.9,1.25,11.4)
        two_column_bullets(slide,"Lợi ích",["Dự báo thay đổi thay vì absolute state","Identity path bảo toàn trạng thái hiện tại","Scale theo σ(Δx) ổn định gradient","Hợp với bước thời gian ngắn 6 giờ"],"Rủi ro",["Sai tendency tích lũy khi rollout","Bias nhỏ lặp lại thành drift","Smooth tendency làm mất cực trị","K=1 không tối ưu trực tiếp lead dài"],right_color=RED)

        slide = std("Hàm loss huấn luyện")
        add_picture(slide,eq_loss,.65,1.15,12.0)
        card(slide,"Area weighting","Bù mật độ lat–lon tại hai cực",.65,3.25,3.75,1.55,accent=BLUE)
        card(slide,"Channel weighting","Cân bằng biến và mức áp suất",4.78,3.25,3.75,1.55,accent=GREEN)
        card(slide,"Tendency scaling","Inverse one-step variance",8.9,3.25,3.75,1.55,accent=ORANGE)
        rounded_box(slide,.65,5.25,12.0,.9,fill=RED_PALE,line="EDBDBD"); text_box(slide,"MSE dễ tối ưu nhưng xem cực trị hiếm như số ít pixel — động lực làm trơn vẫn còn.",1.0,5.54,11.2,.25,size=14,color=RED,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Trọng số biến và diện tích")
        two_column_bullets(slide,"GraphCast-style channel weights",["Surface variables có trọng số định trước","Pressure-level weight ∝ pressure","Mỗi nhóm biến nhận tổng trọng số cân bằng","Vector cuối được chuẩn hóa tổng bằng 1"],"Latitude-cell-area weights",["Dựa trên biên ô latitude thực","Không dùng cos(latitude) tại tâm một cách xấp xỉ","RMSE/Bias pooling theo diện tích","ACC trừ training climatology"])
        rounded_box(slide,1.45,5.75,10.4,.58,fill=PALE,line="CAD8E5"); text_box(slide,"Cùng objective ở Stage 1 và Stage 2 → loss giảm không phải do đổi thước đo.",1.7,5.93,9.9,.22,size=13,color=NAVY,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Training stack và reproducibility")
        stack=[("PyTorch 2.8","AMP + AdamW"),("torch-harmonics 0.7.4","SHT backend"),("RTX PRO 4500","31,37 GiB VRAM"),("Checkpoint v3","model + optimizer + RNG"),("SHA-256 manifest","artifact integrity")]
        for i,(a,b) in enumerate(stack):
            x=.45+i*2.55; process_box(slide,a,b,x,1.55,2.08,1.12,fill=PALE,line=BLUE)
        rounded_box(slide,.65,3.35,12.0,2.6,fill=PALE_2,line="D2DEE8")
        add_bullets(slide,["Deterministic seed 42 và runtime contract","Resume kiểm tra data/source/config signature","Validation-best checkpoint, không chọn bằng test","Submission audit: clean tree, no secrets, 2 retained checkpoints"],1.0,3.77,11.0,1.65,size=15,spacing=9)

        sec("PHẦN III", "Stage 1 — SFNO baseline", "Thiết lập một baseline mạnh, ổn định và có thể kiểm chứng trước khi thêm độ phức tạp.")

        slide = std("Kiến trúc Stage 1")
        blocks=[("x(t)","71 × 361 × 720"),("Encoder","71 → 192"),("SFNO × 6","scale factor 2"),("Decoder","192 → 71"),("Residual","x + s·ẑ"),("x(t+6h)","71 channels")]
        for i,(a,b) in enumerate(blocks):
            x=.35+i*2.13; process_box(slide,a,b,x,1.65,1.7,1.12,fill=GREEN_PALE if i>=4 else PALE,line=GREEN if i>=4 else BLUE)
            if i<5: connector(slide,x+1.73,2.21,x+2.06,2.21)
        rounded_box(slide,.65,3.5,12.0,2.55,fill=PALE_2,line="D2DEE8")
        add_bullets(slide,["Backbone chỉ nhìn trạng thái hiện tại","Spectral transforms làm việc trên internal grid giảm resolution","Hard-threshold fraction = 1 trong bandwidth nội bộ, nhưng scale factor vẫn loại mode cao","Output là standardized tendency, sau đó residual update về normalized state"],1.0,3.88,11.0,1.65,size=14.5,spacing=9)

        slide = std("Kích thước tensor trong Stage 1")
        dims=[("Input","B × 71 × 361 × 720"),("Encoded","B × 192 × 361 × 720"),("Internal spectral grid","scale factor = 2"),("Tendency","B × 71 × 361 × 720"),("Forecast","B × 71 × 361 × 720")]
        for i,(a,b) in enumerate(dims):
            y=1.18+i*1.03; rounded_box(slide,.78,y,11.75,.78,fill=WHITE,line="D2DEE8")
            text_box(slide,a,1.08,y+.2,2.65,.24,size=14,color=BLUE,bold=True)
            text_box(slide,b,3.72,y+.18,7.9,.28,size=15,color=NAVY,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Cấu hình huấn luyện Stage 1")
        vals=[("Embedding","192"),("SFNO blocks","6"),("Batch size","7"),("Learning rate","5 × 10⁻⁴"),("Weight decay","1 × 10⁻⁴"),("Scheduler","warmup + cosine"),("History","0"),("Rollout","K = 1")]
        for i,(a,b) in enumerate(vals): metric_card(slide,a,b,"",.58+(i%4)*3.12,1.2+(i//4)*2.25,2.78,1.55,accent=BLUE)
        rounded_box(slide,.7,5.82,11.9,.54,fill=PALE,line="CAD8E5"); text_box(slide,"Best checkpoint: epoch 24 · 187.850 completed updates · validation loss 0,134443",.95,5.98,11.4,.22,size=13,color=NAVY,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Chiến lược tối ưu Stage 1")
        two_column_bullets(slide,"Ổn định số học",["AMP trên CUDA","Gradient clipping = 1,0","SHT constants được stabilize","Finite-loss checks mỗi batch"],"Model selection",["Validation theo chronological split","Selection score = weighted mean loss","Giữ best.pt độc lập với last.pt","Không dùng test để dừng train"])
        rounded_box(slide,.85,5.72,11.6,.7,fill=GREEN_PALE,line="B8D8C5"); text_box(slide,"Kết quả: backbone hội tụ ổn định, đủ mạnh để làm điểm khởi tạo cho Stage 2.",1.1,5.94,11.1,.24,size=14,color=GREEN,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Kết quả tổng hợp Stage 1")
        s1=SUMMARY["stage1"]
        metric_card(slide,"Best validation loss","0,134443","epoch 24",.62,1.2,2.75,1.55,accent=BLUE)
        metric_card(slide,"Normalized RMSE",f"{float(s1['normalized_rmse']):.4f}".replace('.',','),"persistence 0,3312",3.7,1.2,2.75,1.55,accent=BLUE)
        metric_card(slide,"Normalized ACC",f"{float(s1['normalized_acc']):.4f}".replace('.',','),"persistence 0,9233",6.78,1.2,2.75,1.55,accent=GREEN)
        metric_card(slide,"Objective loss",f"{float(s1['objective_loss']):.4f}".replace('.',','),"12 samples",9.86,1.2,2.75,1.55,accent=BLUE)
        rounded_box(slide,.62,3.25,12.0,2.65,fill=PALE_2,line="D2DEE8")
        add_bar(slide,"Persistence",float(s1['persistence_rmse']),.35,1.0,3.78,8.5,color=MID)
        add_bar(slide,"Stage 1",float(s1['normalized_rmse']),.35,1.0,4.62,8.5,color=BLUE)
        text_box(slide,"Stage 1 giảm normalized RMSE khoảng 52,5% so với persistence.",1.35,5.35,10.6,.3,size=16,color=GREEN,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Stage 1 trên các biến ưu tiên")
        table=[("Z500","3,493 gpm","0,9997"),("T850","0,580 K","0,9966"),("T2M","0,644 K","0,9963"),("U10","0,751 m/s","0,9837"),("V10","0,802 m/s","0,9810"),("MSLP","0,456 hPa","0,9982"),("Q850","0,603 g/kg","0,9742"),("U850","1,141 m/s","0,9836"),("V850","1,200 m/s","0,9766")]
        for c,(head,x) in enumerate([("Variable",.75),("RMSE",3.6),("ACC",7.0)]): text_box(slide,head,x,1.2,2.5,.28,size=14,color=WHITE,bold=True,align=PP_ALIGN.CENTER); box=slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE,Inches(x-.1),Inches(1.12),Inches(2.7),Inches(.5)); set_fill(box,NAVY); box.line.fill.background(); slide.shapes._spTree.remove(box._element); slide.shapes._spTree.insert(2,box._element)
        for i,(a,b,c) in enumerate(table):
            y=1.77+i*.52; fill=PALE_2 if i%2==0 else WHITE
            rounded_box(slide,.65,y,11.9,.43,fill=fill,line="E1E8EE",radius=False)
            text_box(slide,a,.9,y+.1,2.1,.18,size=10.5,color=NAVY,bold=True,align=PP_ALIGN.CENTER); text_box(slide,b,3.7,y+.1,2.5,.18,size=10.5,color=NAVY,align=PP_ALIGN.CENTER); text_box(slide,c,7.1,y+.1,2.3,.18,size=10.5,color=NAVY,align=PP_ALIGN.CENTER)

        slide = std("Điểm mạnh của Stage 1")
        two_column_bullets(slide,"Đã làm tốt",["Học trường quy mô lớn ổn định","ACC cao trên geopotential và pressure","Vượt persistence ở lead 6 giờ","Checkpoint đủ mạnh cho transfer learning"],"Giá trị kỹ thuật",["Pipeline worker-safe cho Zarr","Resume contract chặt chẽ","Metrics hoàn nguyên vật lý","Reproducible artifact packaging"])

        slide = std("Nút thắt phổ tần số cao", "Biểu đồ mang tính minh họa cơ chế, không phải spectrum đo trực tiếp từ checkpoint")
        add_picture(slide,chart_paths["spectrum"],.75,1.15,7.85,4.55)
        rounded_box(slide,8.88,1.35,3.75,4.25,fill=RED_PALE,line="EDBDBD")
        text_box(slide,"Thông tin bị ảnh hưởng",9.18,1.7,3.15,.3,size=17,color=RED,bold=True)
        add_bullets(slide,["Weather fronts","Xoáy quy mô nhỏ","Gradient ven biển","Cực trị gió/nhiệt","Chi tiết địa hình"],9.2,2.3,2.95,2.3,size=14.5,spacing=9)
        text_box(slide,"Mode đã cắt không thể được optimizer học lại.",9.25,4.95,2.85,.38,size=12.5,color=RED,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Vì sao Stage 1 bị bão hòa?")
        causes=[("Information bottleneck","Internal resolution hạn chế mode cao",RED),("One-state input","Không quan sát xu hướng gần nhất",ORANGE),("Missing forcing","Không biết địa hình và chu kỳ thời gian",BLUE),("Average MSE","Ưu tiên nghiệm trơn và phổ biến",RED),("Rare extremes","Đóng góp gradient quá nhỏ",ORANGE),("K = 1","Không tối ưu drift dài hạn",BLUE)]
        for i,(a,b,c) in enumerate(causes): card(slide,a,b,.62+(i%3)*4.15,1.2+(i//3)*2.42,3.78,1.82,accent=c,fill=RED_PALE if c==RED else PALE_2)

        sec("PHẦN IV", "Stage 2 — Spherical residual corrector", "Không thay backbone một cách mạo hiểm; học phần sai số còn lại với history và forcing.")

        slide = std("Nguyên lý thiết kế Stage 2")
        add_picture(slide,eq_stage2,.55,1.15,12.2)
        two_column_bullets(slide,"Giữ lại từ Stage 1",["Backbone E192/L6","Tendency scale","Residual state update","Loss và channel weighting"],"Bổ sung",["x(t−6h) và recent tendency","7 analytic/static forcing channels","Full-resolution SFNO corrector","Progressive unfreezing"])

        slide = std("Forcing encoder có ý nghĩa vật lý")
        forcing=[("Orography","địa hình chuẩn hóa",BLUE),("Land–sea mask","khác biệt bề mặt",GREEN),("Solar zenith","bức xạ tức thời",ORANGE),("Local time","sin/cos chu kỳ ngày",BLUE_2),("Year progress","sin/cos mùa",RED)]
        for i,(a,b,c) in enumerate(forcing):
            x=.48+i*2.52; card(slide,a,b,x,1.35,2.18,1.55,accent=c)
        rounded_box(slide,.7,3.55,11.9,2.1,fill=PALE_2,line="D2DEE8")
        add_bullets(slide,["Ba scalar phase được chuyển thành field trực tiếp trên accelerator","Static forcing được kiểm SHA-256 và tọa độ trước khi dùng","Backbone không nhận forcing; forcing chỉ condition corrector","Tổng forcing channels = 7"],1.05,3.92,11.0,1.3,size=14.5,spacing=8)

        slide = std("Kiến trúc spherical corrector")
        process_box(slide,"Covariates","220 channels",.6,1.6,2.0,1.12)
        process_box(slide,"Latitude geometry","sin φ, cos φ",3.05,1.6,2.0,1.12,fill=PALE,line=ORANGE)
        process_box(slide,"Encoder","222 → 84",5.5,1.6,2.0,1.12)
        process_box(slide,"SFNO × 4","native grid",7.95,1.6,2.0,1.12,fill=GREEN_PALE,line=GREEN)
        process_box(slide,"Decoder","84 → 71",10.4,1.6,2.0,1.12,fill=GREEN_PALE,line=GREEN)
        for x in [2.64,5.09,7.54,9.99]: connector(slide,x,2.16,x+.35,2.16)
        rounded_box(slide,.65,3.48,12.0,2.45,fill=PALE_2,line="D2DEE8")
        add_bullets(slide,["Scale factor = 1: corrector hoạt động trên 361 × 720","Không planar padding, pooling hoặc bilinear upsampling","Hai geometry channels giúp phân biệt latitude trên lưới equiangular","Projection cuối zero-initialized để prediction ban đầu đúng bằng Stage 1"],1.0,3.87,11.0,1.55,size=14.5,spacing=9)

        slide = std("Luồng dữ liệu đầy đủ của Stage 2")
        process_box(slide,"x(t−6h), x(t)","142 channels",.45,1.35,2.2,1.0)
        process_box(slide,"Stage 1 backbone","chỉ nhận x(t)",3.1,1.35,2.25,1.0)
        process_box(slide,"Baseline ẑ₁","71 channels",5.8,1.35,2.0,1.0)
        process_box(slide,"Add correction","ẑ₁ + Cφ",10.45,1.35,2.25,1.0,fill=GREEN_PALE,line=GREEN)
        connector(slide,2.7,1.85,3.04,1.85); connector(slide,5.4,1.85,5.74,1.85); connector(slide,7.85,1.85,10.38,1.85)
        inputs=[("x(t)",71),("Δx / σΔ",71),("stopgrad(ẑ₁)",71),("forcing",7),("geometry",2)]
        for i,(a,b) in enumerate(inputs):
            x=.6+i*2.47; rounded_box(slide,x,3.25,2.05,.75,fill=PALE_2,line="D2DEE8"); text_box(slide,a,x+.12,3.43,1.38,.2,size=11,color=NAVY,bold=True); text_box(slide,str(b),x+1.55,3.42,.3,.2,size=11,color=BLUE,bold=True,align=PP_ALIGN.RIGHT)
            connector(slide,x+1.02,4.04,6.65,4.92,GREEN,1.2)
        process_box(slide,"Corrector SFNO","222 → 84 → 71",5.32,4.92,2.72,1.0,fill=GREEN_PALE,line=GREEN)
        connector(slide,8.08,5.42,11.55,2.4,GREEN,1.8)

        slide = std("Transfer learning và zero initialization")
        two_column_bullets(slide,"Sao chép chính xác",["Toàn bộ backbone tensors","Tendency scale 71 channels","Kiểm tra shape và config contract","Không chuyển optimizer state"],"Khởi tạo an toàn",["Corrector decoder = 0","Cφ(·) = 0 tại update đầu","Stage 2 bắt đầu đúng bằng Stage 1","Tránh correction ngẫu nhiên phá forecast"])
        rounded_box(slide,1.15,5.7,10.9,.66,fill=GREEN_PALE,line="B8D8C5"); text_box(slide,"Đây là residual learning theo nghĩa chặt: chỉ học sai số mà baseline chưa giải thích.",1.45,5.91,10.3,.24,size=14,color=GREEN,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Progressive fine-tuning")
        bar=slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE,Inches(1.1),Inches(2.18),Inches(10.9),Inches(.1)); set_fill(bar,MID); bar.line.fill.background()
        for x,label,sub,color in [(1.15,"Khởi tạo","Stage 1 exact",BLUE),(4.5,"Epoch 0–1","corrector only",BLUE),(8.0,"Epoch 2+","corrector + tail",GREEN),(11.45,"Best","epoch 3",GREEN)]:
            dot=slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL,Inches(x),Inches(1.88),Inches(.62),Inches(.62)); set_fill(dot,color); dot.line.fill.background(); text_box(slide,label,x-.42,2.72,1.45,.24,size=12,color=color,bold=True,align=PP_ALIGN.CENTER); text_box(slide,sub,x-.55,3.08,1.75,.2,size=10,color=NAVY,align=PP_ALIGN.CENTER)
        card(slide,"Corrector","LR = 1×10⁻⁴ · WD = 1×10⁻⁴",1.0,4.15,3.45,1.32,accent=BLUE)
        card(slide,"Backbone tail","LR = 1×10⁻⁵ · WD = 0",4.92,4.15,3.45,1.32,accent=GREEN)
        card(slide,"Unfreeze scope","final block + decoder",8.83,4.15,3.45,1.32,accent=ORANGE)
        text_box(slide,"LR backbone nhỏ hơn 10× để hạn chế catastrophic forgetting.",1.2,5.9,10.8,.26,size=14,color=NAVY,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Cấu hình huấn luyện Stage 2")
        vals=[("History","1 → 2 states"),("Corrector","E84 / L4"),("Batch size","6"),("Corrector LR","1 × 10⁻⁴"),("Backbone LR","1 × 10⁻⁵"),("Corrector-only","2 epochs"),("Unfreeze blocks","1 + decoder"),("Rollout","K = 1")]
        for i,(a,b) in enumerate(vals): metric_card(slide,a,b,"",.58+(i%4)*3.12,1.2+(i//4)*2.25,2.78,1.55,accent=GREEN if i in {1,4,5,6} else BLUE)
        rounded_box(slide,.7,5.82,11.9,.54,fill=GREEN_PALE,line="B8D8C5"); text_box(slide,"Best checkpoint: epoch 3 · 78.892 completed updates · validation loss 0,125384",.95,5.98,11.4,.22,size=13,color=GREEN,bold=True,align=PP_ALIGN.CENTER)

        sec("PHẦN V", "Kết quả và thảo luận", "So sánh aggregate, 9 biến ưu tiên và toàn bộ 71 channels; sau đó đánh giá giới hạn khoa học.")

        slide = std("So sánh kết quả tổng hợp")
        add_picture(slide,chart_paths["aggregate"],.58,1.1,8.2,4.75)
        gains=[("Validation loss","−6,74%",GREEN),("Objective loss","−3,49%",GREEN),("Normalized RMSE","−1,48%",GREEN),("ACC","+0,000536",BLUE)]
        for i,(a,b,c) in enumerate(gains): metric_card(slide,a,b,"Stage 2 vs Stage 1",9.05,1.12+i*1.3,3.35,1.05,accent=c)
        text_box(slide,"Cải thiện nhất quán nhưng có biên nhỏ — đúng kỳ vọng của một residual corrector.",.9,6.13,11.5,.26,size=14,color=NAVY,bold=True,align=PP_ALIGN.CENTER)

        slide = std("RMSE trên 9 biến ưu tiên")
        add_picture(slide,chart_paths["priority"],.72,1.08,8.45,4.95)
        rounded_box(slide,9.43,1.3,3.12,4.45,fill=GREEN_PALE,line="B8D8C5")
        text_box(slide,"Quan sát",9.75,1.66,2.45,.3,size=18,color=GREEN,bold=True)
        add_bullets(slide,["9/9 biến đều giảm RMSE","Q850 cải thiện ≈ 4,1%","T2M cải thiện ≈ 3,7%","Gió cải thiện 2–3%","MSLP chỉ cải thiện nhẹ"],9.72,2.25,2.35,2.6,size=13.5,spacing=9)

        slide = std("Phân bố cải thiện trên 71 channels")
        add_picture(slide,chart_paths["all_channels"],.55,1.12,8.55,4.8)
        paired=defaultdict(dict)
        for row in ALL_ROWS: paired[row["variable"]][row["stage"]]=row
        changes=[]
        for variable,stages in paired.items():
            if 'stage1' in stages and 'stage2' in stages:
                a=float(stages['stage1']['rmse']); b=float(stages['stage2']['rmse']); changes.append(100*(a-b)/a)
        improved=sum(v>0 for v in changes); worsened=sum(v<0 for v in changes)
        metric_card(slide,"Channels improved",f"{improved} / {len(changes)}","RMSE Stage 2 thấp hơn",9.42,1.45,3.1,1.4,accent=GREEN)
        metric_card(slide,"Channels worsened",str(worsened),"cần xem theo biến/mức",9.42,3.05,3.1,1.4,accent=RED if worsened else BLUE)
        metric_card(slide,"Median change",f"{np.median(changes):.2f}%","RMSE reduction",9.42,4.65,3.1,1.4,accent=BLUE)

        slide = std("Cải thiện theo nhóm biến")
        add_picture(slide,chart_paths["groups"],.72,1.12,8.65,4.85)
        rounded_box(slide,9.62,1.35,2.9,4.45,fill=PALE_2,line="D2DEE8")
        text_box(slide,"Ý nghĩa",9.93,1.7,2.25,.3,size=18,color=BLUE,bold=True)
        add_bullets(slide,["Không đồng đều giữa nhóm biến","Surface hưởng lợi từ forcing","Humidity nhạy với smoothing","Gió hưởng lợi từ recent tendency","Cần ablation theo pressure level"],9.9,2.25,2.2,2.7,size=13,spacing=8)

        slide = std("ACC và spatial variance ratio")
        add_picture(slide,chart_paths["acc_variance"],.52,1.08,12.0,4.75)
        rounded_box(slide,.75,6.0,11.8,.52,fill=PALE,line="CAD8E5")
        text_box(slide,"Variance ratio gần 1 hơn không phải ở mọi biến; giảm RMSE chưa đồng nghĩa phổ biên độ được phục hồi hoàn toàn.",1.0,6.16,11.25,.22,size=12.5,color=NAVY,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Bias: cải thiện hay đổi dấu?")
        priority=defaultdict(dict)
        for row in PRIORITY_ROWS: priority[row['variable']][row['stage']]=row
        variables=["Z500","T850","T2M","U10","V10","MSLP","Q850","U850","V850"]
        for i,name in enumerate(variables):
            r1=priority[name]['stage1']; r2=priority[name]['stage2']; b1=float(r1['bias']); b2=float(r2['bias'])
            y=1.13+i*.57; rounded_box(slide,.7,y,11.9,.46,fill=PALE_2 if i%2==0 else WHITE,line="E0E8EF",radius=False)
            text_box(slide,name,.95,y+.11,1.2,.18,size=10.5,color=NAVY,bold=True)
            text_box(slide,f"S1 {b1:+.4f}",3.1,y+.11,2.0,.18,size=10.5,color=GRAY,align=PP_ALIGN.CENTER)
            text_box(slide,f"S2 {b2:+.4f}",6.0,y+.11,2.0,.18,size=10.5,color=GREEN if abs(b2)<abs(b1) else RED,bold=True,align=PP_ALIGN.CENTER)
            text_box(slide,"|bias| ↓" if abs(b2)<abs(b1) else "|bias| ↑",9.35,y+.11,1.5,.18,size=10.5,color=GREEN if abs(b2)<abs(b1) else RED,bold=True,align=PP_ALIGN.CENTER)
        text_box(slide,"Bias phải được đọc cùng RMSE và variance ratio; một metric riêng lẻ không đủ kết luận.",1.0,6.55,11.2,.24,size=12.5,color=NAVY,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Điều gì thực sự làm loss giảm?")
        factors=[("1","Residual learning","Học phần lỗi còn lại",GREEN),("2","Stage 1 initialization","Bắt đầu từ nghiệm tốt",BLUE),("3","History / tendency","Quan sát hướng chuyển động",ORANGE),("4","Full-resolution corrector","Bổ sung capacity phổ cao",GREEN),("5","Physical forcing","Địa hình + thời gian",BLUE),("6","Progressive unfreezing","Thích nghi có kiểm soát",ORANGE)]
        for i,(rank,a,b,c) in enumerate(factors):
            x=.62+(i%2)*6.15; y=1.15+(i//2)*1.72; rounded_box(slide,x,y,5.8,1.35,fill=PALE_2,line="D2DEE8")
            rounded_box(slide,x+.2,y+.25,.68,.68,fill=c,line=c); text_box(slide,rank,x+.32,y+.46,.42,.2,size=13,color=WHITE,bold=True,align=PP_ALIGN.CENTER)
            text_box(slide,a,x+1.08,y+.22,4.2,.25,size=15,color=NAVY,bold=True); text_box(slide,b,x+1.08,y+.68,4.2,.22,size=11.5,color=GRAY)
        text_box(slide,"Xếp hạng là suy luận từ thiết kế; chưa có ablation để xác lập đóng góp nhân quả.",1.0,6.42,11.2,.24,size=12.5,color=RED,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Các mối đe dọa đến tính hợp lệ")
        threats=[("Internal validity","Chưa có ablation từng thành phần"),("Statistical validity","Smoke test chỉ 12 samples"),("Evaluation scope","K=1 không đại diện 3–7 ngày"),("Sampling","Extreme events quá hiếm"),("Representation","0,5° và spectral cutoff mất chi tiết"),("Metric","Average MSE che khuất tail error")]
        for i,(a,b) in enumerate(threats): card(slide,a,b,.62+(i%3)*4.15,1.2+(i//3)*2.42,3.78,1.82,accent=RED if i>=3 else ORANGE,fill=RED_PALE if i>=3 else PALE_2)

        slide = std("Ablation còn thiếu nhưng quan trọng")
        experiments=[("A1","No forcing","đo đóng góp forcing"),("A2","No history","đo recent tendency"),("A3","Corrector scale factor 2","đo full resolution"),("A4","Frozen backbone only","đo progressive unfreeze"),("A5","Vary spectral modes","tìm compute–accuracy frontier"),("A6","Extreme-aware loss","đo tail skill")]
        for i,(tag,a,b) in enumerate(experiments):
            y=1.12+i*.91; rounded_box(slide,.75,y,11.75,.7,fill=WHITE,line="D2DEE8")
            rounded_box(slide,.95,y+.1,.62,.48,fill=BLUE if i<4 else GREEN,line=BLUE if i<4 else GREEN); text_box(slide,tag,1.03,y+.25,.46,.16,size=9.5,color=WHITE,bold=True,align=PP_ALIGN.CENTER)
            text_box(slide,a,1.85,y+.17,3.55,.22,size=13,color=NAVY,bold=True)
            text_box(slide,b,5.35,y+.17,6.25,.22,size=12,color=GRAY)

        slide = std("Hướng cải thiện loss và sampling")
        two_column_bullets(slide,"Objective",["MSE + spectral loss theo band","Gradient/front loss","Quantile hoặc tail-weighted loss","Multi-scale wavelet loss","Calibration/probabilistic objective"],"Sampling & curriculum",["Oversample cyclone/front/extreme windows","Curriculum K=1 → multi-step","Hard-example replay","Balance theo regime và mùa","Event-based validation split"])
        rounded_box(slide,1.0,5.92,11.2,.54,fill=GREEN_PALE,line="B8D8C5"); text_box(slide,"Mục tiêu không phải bỏ MSE, mà bổ sung tín hiệu gradient cho cấu trúc hiếm và dải tần cao.",1.25,6.08,10.7,.22,size=13,color=GREEN,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Hướng cải thiện kiến trúc")
        ideas=[("Multi-resolution operator","Backbone phổ thấp + high-frequency branch",BLUE),("Wavelet neural operator","Định vị tốt hơn spectral Fourier",GREEN),("Local-global hybrid","SFNO + convolution/attention cục bộ",ORANGE),("Adaptive spectral gating","Học mode cần giữ theo channel",BLUE_2),("Event-conditioned expert","Mixture-of-experts cho regime cực đoan",RED),("Probabilistic decoder","Dự báo ensemble / uncertainty",GREEN)]
        for i,(a,b,c) in enumerate(ideas): card(slide,a,b,.62+(i%3)*4.15,1.2+(i//3)*2.42,3.78,1.82,accent=c)

        slide = std("Bài học rút ra")
        lessons=[("01","High frequency không đồng nghĩa noise"),("02","Loss thấp chưa đủ cho extreme skill"),("03","Ablation đáng giá hơn thêm stage tùy ý"),("04","Data contract quan trọng ngang model"),("05","Transfer learning tiết kiệm compute"),("06","Metric phải bám câu hỏi khoa học")]
        for i,(tag,body) in enumerate(lessons):
            x=.62+(i%2)*6.15; y=1.15+(i//2)*1.72
            rounded_box(slide,x,y,5.8,1.34,fill=PALE_2,line="D2DEE8")
            text_box(slide,tag,x+.22,y+.28,.75,.35,size=19,color=BLUE if i<3 else GREEN,bold=True)
            text_box(slide,body,x+1.15,y+.3,4.25,.56,size=15,color=NAVY,bold=True,valign=MSO_ANCHOR.MIDDLE)

        slide = std("Tính tái lập của submission")
        artifacts=[("checkpoints/stage1/best.pt","Stage 1 validation-best"),("checkpoints/stage2/best.pt","Final retained model"),("data/stats","Training-only normalization"),("eval/","Independent July 2020 evaluator"),("results/stage_smoke_tests","71-channel result tables"),("MANIFEST.sha256","Binary integrity")]
        for i,(a,b) in enumerate(artifacts):
            x=.65+(i%2)*6.08; y=1.15+(i//2)*1.65; card(slide,a,b,x,y,5.7,1.28,accent=BLUE if i%2==0 else GREEN)
        text_box(slide,"Submission audit: passed · Pyright: 0 errors / 0 warnings",1.15,6.27,11.0,.28,size=15,color=GREEN,bold=True,align=PP_ALIGN.CENTER)

        slide = std("Kết luận")
        rounded_box(slide,.58,1.12,5.95,5.65,fill=PALE,line="CAD8E5")
        text_box(slide,"Kết quả chính",.95,1.48,2.5,.34,size=21,color=BLUE,bold=True)
        add_bullets(slide,["Stage 1 thiết lập SFNO baseline ổn định","Stage 2 giảm validation loss 6,74%","9/9 biến ưu tiên giảm RMSE","Residual corrector là cải tiến hiệu quả nhất","Pipeline có checkpoint và evaluator tái lập"],.95,2.08,5.0,3.05,size=15,spacing=11)
        rounded_box(slide,6.82,1.12,5.95,5.65,fill=GREEN_PALE,line="B8D8C5")
        text_box(slide,"Kết luận khoa học",7.18,1.48,3.1,.34,size=21,color=GREEN,bold=True)
        add_bullets(slide,["Cải thiện average skill là thật nhưng còn nhỏ","Không thể suy ra extreme-event skill","Spectral cutoff tạo information bottleneck","K=1 chưa chứng minh forecast dài hạn","Bước tiếp theo phải tập trung vào phổ cao và tail error"],7.18,2.08,5.0,3.05,size=15,spacing=11)
        text_box(slide,"Stage 2 là điểm dừng hợp lý cho submission; giá trị lớn nhất là bài học thiết kế thí nghiệm.",1.15,6.35,11.0,.28,size=14,color=NAVY,bold=True,align=PP_ALIGN.CENTER)

        # Final Q&A
        n += 1; slide=prs.slides.add_slide(blank); bg=slide.background.fill; bg.solid(); bg.fore_color.rgb=rgb(NAVY)
        slide.shapes.add_picture(str(logo),Inches(.75),Inches(.62),height=Inches(.85))
        line=slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE,Inches(5.25),Inches(2.2),Inches(2.8),Inches(.06)); set_fill(line,BLUE_2); line.line.fill.background()
        text_box(slide,"CẢM ƠN THẦY/CÔ VÀ CÁC BẠN ĐÃ THEO DÕI",1.1,2.65,11.1,.55,size=26,color=WHITE,bold=True,align=PP_ALIGN.CENTER)
        text_box(slide,"Q & A",4.9,3.75,3.5,.55,size=28,color=BLUE_2,bold=True,align=PP_ALIGN.CENTER)
        text_box(slide,"NeuralAtmosphereOperator · Stage 1 & Stage 2",3.1,5.2,7.1,.28,size=13,color="D8E7F4",align=PP_ALIGN.CENTER)
        text_box(slide,f"{n:02d}",12.0,6.8,.55,.25,size=10,color=BLUE_2,bold=True,align=PP_ALIGN.RIGHT)

        if n != 57:
            raise RuntimeError(f"Expected 57 base slides, built {n}")
        repair_and_enrich(prs, logo, standard_slide)
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        prs.save(str(OUTPUT))
    return OUTPUT


if __name__ == "__main__":
    print(build())
