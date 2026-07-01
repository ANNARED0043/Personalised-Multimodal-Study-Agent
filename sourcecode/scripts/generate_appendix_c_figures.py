from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)


W, H = 1800, 1050
BG = "#f7f8f5"
PANEL = "#ffffff"
INK = "#111b18"
MUTED = "#5e6f68"
LINE = "#d8dfd9"
GREEN = "#0d7f69"
GREEN_SOFT = "#e2f2ed"
BLACK = "#18201d"
WARN = "#8f5b00"
WARN_SOFT = "#fff2d2"
BLUE = "#1b5f9e"
BLUE_SOFT = "#e7f0fb"
RED = "#9d2f2f"
RED_SOFT = "#fde9e9"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


F_TITLE = font(44, True)
F_SUB = font(24, False)
F_H = font(29, True)
F_BODY = font(24, False)
F_BODY_B = font(24, True)
F_SMALL = font(19, False)
F_SMALL_B = font(19, True)


def rounded(draw: ImageDraw.ImageDraw, box, fill=PANEL, outline=LINE, r=22, width=2):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def pill(draw: ImageDraw.ImageDraw, xy, text, fill=GREEN_SOFT, color=GREEN, pad_x=22, pad_y=10, f=F_SMALL_B):
    x, y = xy
    b = draw.textbbox((0, 0), text, font=f)
    w, h = b[2] - b[0], b[3] - b[1]
    draw.rounded_rectangle((x, y, x + w + pad_x * 2, y + h + pad_y * 2), radius=22, fill=fill)
    draw.text((x + pad_x, y + pad_y - 2), text, font=f, fill=color)
    return x + w + pad_x * 2


def text(draw, xy, s, f=F_BODY, fill=INK, max_width=64, line_gap=8):
    x, y = xy
    for para in s.split("\n"):
        if not para:
            y += f.size
            continue
        lines = wrap(para, width=max_width)
        for line in lines:
            draw.text((x, y), line, font=f, fill=fill)
            y += f.size + line_gap
    return y


def header(draw, title, subtitle):
    draw.text((70, 48), title, font=F_TITLE, fill=INK)
    draw.text((72, 110), subtitle, font=F_SUB, fill=MUTED)
    pill(draw, (1320, 55), "9 docs", "#ffffff", INK, f=F_BODY_B)
    pill(draw, (1450, 55), "1017 evidence", "#ffffff", INK, f=F_BODY_B)
    pill(draw, (1650, 55), "LangGraph", "#ffffff", INK, f=F_BODY_B)


def draw_table(draw, x, y, col_widths, row_h, headers, rows, highlights=None):
    highlights = highlights or {}
    total_w = sum(col_widths)
    draw.rounded_rectangle((x, y, x + total_w, y + row_h * (len(rows) + 1)), radius=18, fill=PANEL, outline=LINE, width=2)
    draw.rounded_rectangle((x, y, x + total_w, y + row_h), radius=18, fill=GREEN_SOFT)
    cx = x
    for i, h in enumerate(headers):
        draw.text((cx + 18, y + 18), h, font=F_SMALL_B, fill=GREEN)
        cx += col_widths[i]
        if i:
            draw.line((cx, y, cx, y + row_h * (len(rows) + 1)), fill=LINE, width=1)
    for r, row in enumerate(rows):
        yy = y + row_h * (r + 1)
        if r in highlights:
            draw.rectangle((x + 2, yy, x + total_w - 2, yy + row_h), fill=highlights[r])
        draw.line((x, yy, x + total_w, yy), fill=LINE, width=1)
        cx = x
        for i, cell in enumerate(row):
            fill = INK
            ff = F_SMALL_B if i == 0 else F_SMALL
            wrapped = wrap(str(cell), width=max(12, col_widths[i] // 12))
            ty = yy + 12
            for line in wrapped[:3]:
                draw.text((cx + 18, ty), line, font=ff, fill=fill)
                ty += 23
            cx += col_widths[i]


def figure_c1():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    header(d, "Appendix C1. Benchmark Coverage and Ablation Dashboard", "Required query families, reproducible metrics, and final-agent comparison.")

    rounded(d, (70, 170, 1730, 950), fill=PANEL, r=28)
    d.text((110, 205), "Evaluation snapshot", font=F_H, fill=INK)
    d.text((110, 250), "The benchmark covers all four required query families and compares the final agent against targeted ablations.", font=F_BODY, fill=MUTED)

    rows = [
        ("Factual retrieval", "Naive RAG failures", "p.8 text + image", "Fully grounded"),
        ("Cross-modal retrieval", "Find RRF / reranker slide image", "p.25-p.29 images", "Slide evidence shown"),
        ("Analytical synthesis", "Retriever vs reranker comparison", "pipeline + RRF + roadmap", "Direct answer first"),
        ("Personalised follow-up", "Day 2 reranker confusion", "plan state + p.28/p.42", "Branch-aware"),
    ]
    draw_table(
        d,
        110,
        315,
        [330, 420, 350, 310],
        95,
        ["Query family", "Representative query", "Expected evidence", "Observed behavior"],
        rows,
        {2: BLUE_SOFT, 3: GREEN_SOFT},
    )

    d.text((110, 730), "Ablation result summary", font=F_H, fill=INK)
    metrics = [
        ("plain_llm", 0.000, RED_SOFT),
        ("text_only", 0.627, WARN_SOFT),
        ("caption_only", 0.723, BLUE_SOFT),
        ("no_rerank", 0.610, WARN_SOFT),
        ("final_agent", 0.748, GREEN_SOFT),
    ]
    bx, by = 112, 790
    max_w = 1130
    for name, score, color in metrics:
        d.text((bx, by + 10), name, font=F_SMALL_B, fill=INK)
        d.rounded_rectangle((bx + 180, by, bx + 180 + max_w, by + 42), radius=16, fill="#eef1ee")
        d.rounded_rectangle((bx + 180, by, bx + 180 + int(max_w * score), by + 42), radius=16, fill=color)
        d.text((bx + 180 + max_w + 28, by + 6), f"{score:.3f}", font=F_SMALL_B, fill=INK)
        by += 54
    pill(d, (1260, 810), "Final agent = hybrid retrieval + RRF + routing + memory", GREEN_SOFT, GREEN, f=F_SMALL_B)
    text(d, (1260, 875), "This figure can be replaced by a real screenshot of the Eval tab after running /api/evaluate.", F_SMALL, MUTED, 38)
    img.save(OUT / "appendix_c1_evaluation_dashboard.png")


def figure_c2():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    header(d, "Appendix C2. Failure Diagnosis with Evidence Panel", "Example of tracing whether a weak answer is caused by retrieval, representation, or orchestration.")

    rounded(d, (70, 165, 1190, 955), fill=PANEL, r=28)
    rounded(d, (1220, 165, 1730, 955), fill="#fbfcfa", r=28)
    d.text((110, 205), "Chat answer area", font=F_H, fill=INK)
    pill(d, (110, 255), "Analytical / multi-step synthesis", BLUE_SOFT, BLUE)
    user_q = "Query: Compare retriever and reranker, and explain why Week 8 says ranking matters."
    text(d, (110, 315), user_q, F_BODY_B, INK, 74)
    d.rounded_rectangle((110, 405, 1135, 570), radius=24, fill=RED_SOFT, outline="#f0b0b0")
    d.text((145, 435), "Failure pattern before fix", font=F_BODY_B, fill=RED)
    text(d, (145, 480), "The answer only points the learner to slides and does not first explain the conceptual difference.", F_BODY, INK, 78)
    d.rounded_rectangle((110, 610, 1135, 835), radius=24, fill=GREEN_SOFT, outline="#b8ddd2")
    d.text((145, 640), "Expected final-agent behavior", font=F_BODY_B, fill=GREEN)
    text(
        d,
        (145, 685),
        "First answer directly: the retriever finds candidate chunks/slides; the reranker reorders those candidates so the best evidence reaches the LLM. Then cite Week 8 RRF and roadmap slides.",
        F_BODY,
        INK,
        78,
    )

    d.text((1260, 205), "Evidence panel", font=F_H, fill=INK)
    cards = [
        ("p.28", "Reranker in RAG", "RRF combines ranked lists from multiple retrievers."),
        ("p.29", "Reranker in RAG", "Rank positions are used instead of raw retrieval scores."),
        ("p.42", "Multimodal RAG Roadmap", "Hybrid retrieval performs best; text and visual retrievers differ."),
    ]
    cy = 270
    for page, title, body in cards:
        rounded(d, (1260, cy, 1690, cy + 170), fill=PANEL, r=20)
        pill(d, (1285, cy + 20), "slide_text", "#f2f3f0", MUTED, f=F_SMALL_B)
        pill(d, (1415, cy + 20), page, GREEN_SOFT, GREEN, f=F_SMALL_B)
        d.text((1285, cy + 72), title, font=F_BODY_B, fill=INK)
        text(d, (1285, cy + 110), body, F_SMALL, MUTED, 40)
        cy += 195
    pill(d, (1260, 870), "Diagnosis: orchestration failure, not missing evidence", WARN_SOFT, WARN, f=F_SMALL_B)
    img.save(OUT / "appendix_c2_failure_diagnosis.png")


def figure_c3():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    header(d, "Appendix C3. Branch Follow-up and Personalised Context", "The branch question answers a local confusion while preserving the original revision plan.")

    rounded(d, (70, 165, 1115, 955), fill=PANEL, r=28)
    rounded(d, (1145, 165, 1730, 955), fill="#fbfcfa", r=28)

    d.text((110, 205), "Main revision plan remains stable", font=F_H, fill=INK)
    plan = [
        ("Day 1", "Review why RAG matters and where naive RAG fails."),
        ("Day 2", "Study reranker, RRF, hybrid retrieval, and evidence quality."),
        ("Day 3", "Practice multimodal RAG and agentic RAG exam answers."),
    ]
    y = 280
    for day, desc in plan:
        fill = BLUE_SOFT if day == "Day 2" else "#f3f5f2"
        rounded(d, (115, y, 1065, y + 120), fill=fill, r=22)
        d.text((150, y + 25), day, font=F_BODY_B, fill=BLUE if day == "Day 2" else INK)
        text(d, (270, y + 25), desc, F_BODY, INK, 60)
        if day == "Day 2":
            pill(d, (865, y + 34), "+ branch", GREEN_SOFT, GREEN, f=F_SMALL_B)
        y += 145

    d.line((610, 625, 610, 760), fill=LINE, width=5)
    d.polygon([(610, 790), (590, 755), (630, 755)], fill=LINE)
    rounded(d, (115, 790, 1065, 910), fill=GREEN_SOFT, r=22)
    d.text((150, 815), "Branch anchor", font=F_BODY_B, fill=GREEN)
    text(d, (330, 815), "The learner's Day 2 confusion is stored as a lightweight branch reference, not as a rewritten main plan.", F_BODY, INK, 58)

    d.text((1185, 205), "Branch panel", font=F_H, fill=INK)
    q = "Branch query:\nI do not understand Day 2 reranker. How is a retriever different from a reranker?"
    rounded(d, (1185, 275, 1690, 430), fill=PANEL, r=22)
    text(d, (1215, 305), q, F_SMALL_B, INK, 45)
    rounded(d, (1185, 465, 1690, 735), fill=GREEN_SOFT, r=22)
    d.text((1215, 495), "Branch answer", font=F_BODY_B, fill=GREEN)
    text(
        d,
        (1215, 545),
        "Retriever = finds candidate evidence. Reranker = reorders those candidates. Week 8 uses RRF to show why ranking quality matters after retrieval.",
        F_BODY,
        INK,
        42,
    )
    rounded(d, (1185, 770, 1690, 910), fill=PANEL, r=22)
    pill(d, (1215, 800), "references", "#f2f3f0", MUTED)
    text(d, (1215, 850), "p.28, p.29, p.42 are attached to this branch answer and can be reopened later.", F_SMALL, MUTED, 45)
    img.save(OUT / "appendix_c3_branch_followup.png")


if __name__ == "__main__":
    figure_c1()
    figure_c2()
    figure_c3()
    print("Generated:")
    print(OUT / "appendix_c1_evaluation_dashboard.png")
    print(OUT / "appendix_c2_failure_diagnosis.png")
    print(OUT / "appendix_c3_branch_followup.png")
