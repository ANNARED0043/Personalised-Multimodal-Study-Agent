from __future__ import annotations

import csv
import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "49085266 report.docx"
FIG_DIR = ROOT / "data" / "evaluation"
SUMMARY_CSV = FIG_DIR / "evaluation_summary.csv"
ARCH_IMG = FIG_DIR / "system_architecture.png"
UI_IMG = FIG_DIR / "system_ui_screenshot.png"
CHART_IMG = FIG_DIR / "ablation_comparison.png"


def font(size=22, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def wrap_text(text, fnt, max_width):
    words = str(text).split()
    lines, cur = [], ""
    for word in words:
        test = (cur + " " + word).strip()
        if not cur or fnt.getlength(test) <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_centered(draw, xy, text, fnt, fill="#1f2320"):
    x1, y1, x2, y2 = xy
    lines = wrap_text(text, fnt, x2 - x1 - 24)
    line_h = fnt.size + 4
    total_h = line_h * len(lines)
    y = y1 + (y2 - y1 - total_h) / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        draw.text((x1 + (x2 - x1 - bbox[2] + bbox[0]) / 2, y), line, font=fnt, fill=fill)
        y += line_h


def box(draw, xy, text, fill="#ffffff", outline="#d7d7d0", fnt=None, radius=14):
    fnt = fnt or font(20, True)
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=2)
    draw_centered(draw, xy, text, fnt)


def arrow(draw, start, end, color="#147766"):
    draw.line([start, end], fill=color, width=4)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 12
    p1 = (end[0] - size * math.cos(angle - 0.45), end[1] - size * math.sin(angle - 0.45))
    p2 = (end[0] - size * math.cos(angle + 0.45), end[1] - size * math.sin(angle + 0.45))
    draw.polygon([end, p1, p2], fill=color)


def read_eval_rows():
    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    order = ["plain_llm", "text_only", "caption_only", "no_visual", "no_router", "no_rerank", "no_memory", "final_agent"]
    return sorted(rows, key=lambda r: order.index(r["mode"]) if r["mode"] in order else 99)


def create_architecture_image():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1500, 850), "#f7f7f5")
    d = ImageDraw.Draw(img)
    title = font(34, True)
    label = font(20, True)
    small = font(17)
    d.text((40, 24), "Figure 2. Multimodal Retrieval + LangGraph Agent Architecture", font=title, fill="#1f2320")

    top = [
        ((50, 110, 300, 235), "Personal KB\nPDF / MD / images\nweak topics", "#e5f2ee"),
        ((355, 110, 635, 235), "Indexing\nslide text + page images\nOCR/caption metadata", "#ffffff"),
        ((690, 110, 990, 235), "Hybrid Retrieval\nBM25 + text vector\nvisual vector + RRF", "#ffffff"),
        ((1045, 110, 1420, 235), "LangGraph Agent\nroute / plan / verify\nmemory + session", "#e5f2ee"),
    ]
    for xy, text, fill in top:
        box(d, xy, text, fill, fnt=label)
    arrow(d, (300, 172), (355, 172))
    arrow(d, (635, 172), (690, 172))
    arrow(d, (990, 172), (1045, 172))

    lower = [
        ((70, 340, 255, 430), "load profile", "#ffffff"),
        ((285, 340, 470, 430), "resolve follow-up", "#ffffff"),
        ((500, 340, 685, 430), "classify intent", "#ffffff"),
        ((715, 340, 900, 430), "retrieve evidence", "#ffffff"),
        ((930, 340, 1115, 430), "verify grounding", "#ffffff"),
        ((1145, 340, 1330, 430), "generate answer", "#ffffff"),
    ]
    d.text((62, 285), "Agent workflow used for every query", font=font(27, True), fill="#1f2320")
    for i, (xy, text, fill) in enumerate(lower):
        box(d, xy, text, fill, fnt=small, radius=12)
        if i < len(lower) - 1:
            arrow(d, (xy[2], 385), (lower[i + 1][0][0], 385))

    box(d, (150, 555, 455, 700), "User-visible answer\nstreaming response\npage-cited slide cards", "#e5f2ee", fnt=label)
    box(d, (600, 555, 905, 700), "Right evidence panel\nbranch Q&A\nmodal page explanation", "#ffffff", fnt=label)
    box(d, (1050, 555, 1355, 700), "Evaluation artifacts\nbenchmark JSON\nCSV + failure report", "#ffffff", fnt=label)
    arrow(d, (1235, 235), (755, 555))
    arrow(d, (1235, 430), (755, 555))
    d.text((60, 770), "Design choice: the UI hides internal trace by default, but stores trace/evidence for reproducible evaluation and report artifacts.", font=small, fill="#56615a")
    img.save(ARCH_IMG)


def create_ui_image():
    img = Image.new("RGB", (1500, 850), "#f7f7f5")
    d = ImageDraw.Draw(img)
    title = font(32, True)
    h = font(22, True)
    small = font(18)
    d.text((38, 26), "Figure 1. System Screenshot: GPT-style Study Chat with Page-grounded Evidence", font=title, fill="#1f2320")
    d.rounded_rectangle((35, 90, 1450, 805), radius=22, fill="#ffffff", outline="#deded8", width=2)
    d.rounded_rectangle((55, 110, 160, 785), radius=18, fill="#f0f0ed")
    for y, txt in [(145, "Chat"), (215, "History"), (285, "Library"), (355, "Eval")]:
        fill = "#ffffff" if txt == "Chat" else "#f7f7f5"
        d.rounded_rectangle((70, y, 145, y + 45), radius=12, fill=fill)
        d.text((82, y + 12), txt, font=font(16, True), fill="#1f2320")

    d.text((190, 120), "Personalised Multimodal Study Agent", font=font(30, True), fill="#1f2320")
    d.rounded_rectangle((190, 185, 900, 250), radius=28, fill="#e9e9e4")
    d.text((220, 205), "retriever 和 reranker 到底怎么区分？", font=h, fill="#111")

    d.text((210, 305), "Study Agent", font=h, fill="#147766")
    answer = [
        "Retriever first recalls candidate chunks/slides from the whole KB.",
        "Reranker then re-orders these candidates; Week 8 p.25-p.29 uses RRF",
        "as an example of combining ranked lists from multiple retrievers.",
    ]
    y = 350
    for line in answer:
        d.text((210, y), line, font=small, fill="#1f2320")
        y += 35

    d.rounded_rectangle((210, 480, 770, 645), radius=16, fill="#fbfbf8", outline="#deded8")
    d.text((235, 500), "Reference slides under the paragraph: p.25, p.28, p.29", font=font(17, True), fill="#147766")
    for i, page in enumerate(["p.25", "p.28", "p.29"]):
        x = 245 + i * 150
        d.rounded_rectangle((x, 545, x + 190, 625), radius=10, fill="#57237a")
        d.text((x + 12, 575), page + " Reranker in RAG", font=font(15, True), fill="#ffffff")

    d.rounded_rectangle((960, 115, 1418, 785), radius=18, fill="#fbfbf8", outline="#deded8")
    d.text((995, 145), "Evidence / Branch", font=h, fill="#1f2320")
    for i, (page, desc) in enumerate([
        ("p.25", "Reranker in RAG: improves hit rate and MRR."),
        ("p.28", "RRF combines ranked lists using rank positions."),
        ("p.29", "Same concept, repeated for reinforcement."),
    ]):
        y = 200 + i * 165
        d.rounded_rectangle((995, y, 1385, y + 130), radius=14, fill="#ffffff", outline="#deded8")
        d.rounded_rectangle((1015, y + 15, 1100, y + 45), radius=15, fill="#e5f2ee")
        d.text((1035, y + 21), page, font=font(15, True), fill="#147766")
        for j, line in enumerate(wrap_text(desc, small, 320)):
            d.text((1015, y + 60 + j * 24), line, font=small, fill="#56615a")
    img.save(UI_IMG)


def create_chart_image():
    rows = read_eval_rows()
    img = Image.new("RGB", (1500, 760), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((40, 25), "Figure 3. Ablation Comparison: Retrieval Quality and Answer Success", font=font(34, True), fill="#1f2320")
    x0, y0 = 120, 610
    chart_w, chart_h = 1250, 470
    d.line((x0, y0, x0 + chart_w, y0), fill="#1f2320", width=2)
    d.line((x0, y0, x0, y0 - chart_h), fill="#1f2320", width=2)
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y0 - chart_h * t
        d.line((x0 - 6, y, x0 + chart_w, y), fill="#eeeeea", width=1)
        d.text((45, y - 10), f"{t:.2f}", font=font(15), fill="#56615a")
    bar_w = 48
    gap = 28
    group_gap = 18
    x = x0 + 30
    colors = ("#147766", "#5a67d8")
    for row in rows:
        recall = float(row["mean_recall_at_k"])
        answer = float(row["mean_answer_success_proxy"])
        vals = [recall, answer]
        for j, val in enumerate(vals):
            h = chart_h * val
            d.rectangle((x + j * (bar_w + 4), y0 - h, x + j * (bar_w + 4) + bar_w, y0), fill=colors[j])
        label = row["mode"].replace("_", "\n")
        d.multiline_text((x - 10, y0 + 12), label, font=font(13, True), fill="#1f2320", spacing=1)
        x += 2 * bar_w + gap + group_gap
    d.rectangle((1010, 90, 1370, 160), fill="#f7f7f5", outline="#deded8")
    d.rectangle((1030, 112, 1060, 137), fill=colors[0])
    d.text((1070, 110), "Recall@k", font=font(17, True), fill="#1f2320")
    d.rectangle((1160, 112, 1190, 137), fill=colors[1])
    d.text((1200, 110), "Answer proxy", font=font(17, True), fill="#1f2320")
    img.save(CHART_IMG)


def esc(text):
    return escape(str(text), {"'": "&apos;", '"': "&quot;"})


def r(text, bold=False, size=16):
    b = "<w:b/>" if bold else ""
    return f'<w:r><w:rPr>{b}<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def p(text="", bold=False, size=17, after=28):
    return f'<w:p><w:pPr><w:spacing w:after="{after}" w:line="205" w:lineRule="auto"/></w:pPr>{r(text, bold, size)}</w:p>'


def heading(text):
    return p(text, bold=True, size=20, after=20)


def bullet(text):
    return f'<w:p><w:pPr><w:spacing w:after="10" w:line="200" w:lineRule="auto"/><w:ind w:left="260" w:hanging="160"/></w:pPr>{r("• " + text, size=16)}</w:p>'


def page_break():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def table(rows, widths, size=13):
    xml = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/><w:tblLook w:firstRow="1" w:noVBand="1"/></w:tblPr><w:tblGrid>']
    for w in widths:
        xml.append(f'<w:gridCol w:w="{w}"/>')
    xml.append("</w:tblGrid>")
    for i, row in enumerate(rows):
        xml.append("<w:tr>")
        for j, cell in enumerate(row):
            fill = '<w:shd w:fill="E5F2EE"/>' if i == 0 else ""
            xml.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[j]}" w:type="dxa"/>{fill}</w:tcPr>{p(str(cell), bold=(i == 0), size=size, after=8)}</w:tc>')
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def img_paragraph(rid, cx, cy):
    return f"""
<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="10"/></w:pPr><w:r><w:drawing>
<wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="1" name="{rid}"/>
<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="0" name="{rid}.png"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rid}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>
</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>
"""


def eval_table_rows():
    rows = [["Mode", "R@k", "Page", "MRR", "Ans", "ms"]]
    for row in read_eval_rows():
        rows.append([
            row["mode"], row["mean_recall_at_k"], row["mean_page_recall_at_k"],
            row["mean_mrr"], row["mean_answer_success_proxy"], row["mean_latency_ms"]
        ])
    return rows


def document_xml():
    body = []
    body += [
        p("49085266 - Personalised Multimodal Study Agent", True, 23, 10),
        p("INFS4205/7205 A3 Project Report | 4-page main report, appendix artifacts in data/evaluation", True, 14, 20),
        heading("1. Introduction: problem, interest, hypothesis"),
        p("Problem. Generic chatbots are fluent but weakly grounded in a student's own lecture PDFs, slide diagrams, notes and weak topics. In revision, a useful answer must cite exact pages, expose the original slide image, and support follow-up questions without rewriting the whole plan. This makes the project a systems question: can retrieval, representation, state and orchestration improve study usefulness?", size=16),
        p("Research question and hypothesis. Does a personalised multimodal agent with slide-image grounding, separate indexes, hybrid retrieval/RRF, LangGraph routing, lightweight memory and branch Q&A improve grounded study answers over plain LLM and text-only RAG? I hypothesise that the final agent improves Recall@k, Page Recall@k and answer-success proxy, especially for cross-modal slide search and branch-style follow-up.", size=16),
        heading("2. Knowledge Base Construction"),
        bullet("Data: uploaded lecture PDFs, Markdown/text notes, images/screenshots and CSV tables, organised under data/courses/INFS4205 with configurable SQLite DB path."),
        bullet("Personalisation: profile, weak topics, conversation sessions, branch references and course-specific files rather than a public corpus."),
        bullet("Modalities: slide text chunks + rendered slide images + OCR/caption anchors + metadata (week, source, page, topics, visual type, index name)."),
        bullet("Granularity: each PDF page is represented as both text evidence and image evidence, so cited ranges such as p.25-p.29 can show the original slides under the relevant answer paragraph."),
        heading("Concrete system case"),
        p("For the query 'retriever 和 reranker 到底怎么区分?', the system retrieves Week 8 Reranker slides p.25-p.29, inserts those slide cards under the paragraph that cites them, and lets the user open each page for a page-specific explanation.", size=15),
        img_paragraph("rId6", 5750000, 3250000),
        p("Figure 1. User-facing screenshot. Replace this generated mock screenshot with a real screenshot showing: a Chinese query about retriever vs reranker; the agent answer; p.25/p.28/p.29 slide cards under the answer paragraph; and the right Evidence/Branch panel. This figure supports the KB, multimodal evidence and UX requirements.", True, 13, 0),
        page_break(),
    ]

    body += [
        heading("3. Retrieval / Representation Design"),
        p("Representation. Each evidence item stores text tokens, a text hash vector, a lightweight visual-signature vector, OCR/caption text and structured metadata. The logical indexes are text_index, visual_caption_index and table_index. Retrieval combines BM25-style keyword scoring, text-vector similarity, visual-vector similarity and metadata boosts, then applies Reciprocal Rank Fusion over keyword/text/visual/metadata rankings.", size=16),
        p("Design alternatives. plain_llm tests no retrieval; text_only tests lecture text; caption_only tests slide-image OCR/caption evidence; no_visual removes image evidence; no_router removes task-specific routing; no_rerank removes RRF; no_memory removes personal state; final_agent uses all components. This directly matches the assignment's requirement to compare the final system against baselines and ablations, not just screenshots.", size=16),
        bullet("Vector DB/metadata: SQLite stores vectors and metadata; FAISS IndexFlatIP is used for vector ranking when available."),
        bullet("Multimodal representation: text chunks answer definitions; slide images preserve diagrams/tables/layout; OCR/caption bridges text queries to visual pages."),
        bullet("Ranking/fusion: RRF makes the system robust when BM25, vector score and visual/caption evidence disagree."),
        heading("4. Agent Workflow"),
        p("The LangGraph pipeline is: load profile -> resolve follow-up -> classify intent -> retrieve evidence -> verify grounding -> generate answer -> update memory -> persist session. Routing matters because document overview needs reading-order coverage, slide search needs image evidence, comparison needs focused concept retrieval, and branch questions need local context with low memory impact. The agent therefore coordinates tools, states and verification steps that a one-shot RAG chain cannot represent cleanly.", size=16),
        img_paragraph("rId5", 5750000, 3260000),
        p("Figure 2. Architecture and workflow. The top row maps uploaded personalised data to indexing, hybrid retrieval and the LangGraph agent. The bottom row shows the actual query-processing states. This figure is intended to demonstrate technical requirements: vector/metadata retrieval, multimodal representation, separate indexes, OCR/caption indexing, routing, memory/state and verification.", True, 13, 0),
        heading("Rubric-to-system mapping"),
        table([
            ["Requirement", "Implemented evidence"],
            ["Retrieval component", "vector DB/metadata, multimodal representation, separate indexes, OCR/caption, hybrid retrieval, RRF"],
            ["Agent framework", "LangGraph routing, retrieval planning, memory/state, tool selection, task decomposition, verification"],
            ["User experience", "GPT-style streaming chat, history sessions, branch Q&A, slide cards under cited paragraphs"],
        ], [2200, 5600], 12),
        page_break(),
    ]

    body += [
        heading("5. Experimental Setup"),
        p("Benchmark. Eight queries cover four families required for a systems evaluation: factual retrieval (e.g., Week 8 Naive RAG failures p.8), cross-modal retrieval (e.g., Reranker/RRF slide images p.25-p.29), analytical multi-hop synthesis (retriever vs reranker; multimodal vs text-only RAG), and personalised follow-up/revision planning. Expected keywords, pages, source hints and modality requirements are stored in data/evaluation/benchmark_cases.json.", size=16),
        p("Metrics. Recall@k measures keyword coverage; Page Recall@k checks exact slide-page grounding; MRR measures ranking quality; answer-success proxy measures whether the generated/grounded answer covers expected concepts; latency and tool calls measure efficiency. The evaluation endpoint also writes evaluation_results.json, evaluation_summary.csv and failure_cases.md, satisfying the requirement for reproducible quantitative evidence.", size=16),
        heading("6. Results and Analysis"),
        table(eval_table_rows(), [1450, 760, 760, 760, 760, 760], 12),
        img_paragraph("rId7", 5750000, 2920000),
        p("Figure 3. Ablation plot. final_agent has the strongest Recall@k/answer proxy; plain_llm has no grounded retrieval; no_rerank lowers MRR, showing the value of fusion/ranking. Caption-only is competitive because lecture slides contain rich OCR/caption anchors, but the final agent is more general across factual, visual, analytical and personalised query families.", True, 13, 0),
        p("Key finding. final_agent reaches Recall@k=0.748 and Answer Proxy=0.748, outperforming text_only (0.627), no_router (0.627), no_rerank (0.610) and no_memory (0.627). The gap between no_router and final_agent shows that retrieval planning matters; the gap between no_rerank and final_agent shows RRF improves ranking stability; the gap between plain_llm and all retrieval modes shows why grounding is essential.", size=16),
        p("Trade-off. The final agent is slower than the fastest caption-only mode, but still practical in the retrieval benchmark (30.6 ms without optional LLaVA generation). The extra cost buys page evidence, personalised state, branch handling and verification, which are exactly the study-agent behaviours the assignment rewards.", size=16),
        page_break(),
    ]

    body += [
        heading("7. Failure Cases"),
        p("Failure analysis is recorded in data/evaluation/failure_cases.md. The weakest cases are personalised_revision_plan (Recall@k=0.4) and personalised_branch_followup (Recall@k=0.6 but Page Recall=1.0). These are not simple retrieval failures: planning queries require broad cross-lecture coverage, while branch questions retrieve the correct Reranker pages but require concise local context. Dense visual slides may also fail if OCR/caption anchors are insufficient.", size=16),
        bullet("Retrieval failure: ambiguous week/lecture references can retrieve nearby but not exact pages."),
        bullet("Representation failure: current visual vector is lightweight; CLIP/OpenCLIP/ColPali would improve true image semantics."),
        bullet("Orchestration failure: broad revision plans need coverage-aware retrieval, not only top-k similarity."),
        bullet("Reasoning/display failure: if the generated answer cites a page range but the evidence list lacks one page, the UI cannot show every slide; the fix is to add coverage-aware page expansion after retrieval."),
        heading("8. Conclusion and submission compliance"),
        p("The system satisfies the assignment as a mini systems research project: it states a design hypothesis, builds a personalised multimodal KB, implements separate indexes and hybrid retrieval, uses LangGraph routing/memory/verification, compares baselines and ablations quantitatively, and analyses trade-offs/failures. The main lesson is that study-agent quality comes from retrieval design and evidence presentation, not from LLM generation alone.", size=16),
        p("Appendix allowance. The main report is four pages. Supporting artifacts are intentionally kept outside the page limit: benchmark_cases.json, evaluation_results.json, evaluation_summary.csv, failure_cases.md, generated figures, and the runnable Flask/LangGraph code.", size=16),
        table([
            ["Deliverable/rubric item", "Where it is satisfied"],
            ["Technical requirements", "app.py: PDF/MD/image KB, text/image indexes, hybrid multimodal retrieval, LangGraph agent"],
            ["Evaluation requirements", "data/evaluation: benchmark_cases.json, evaluation_results.json, evaluation_summary.csv, failure_cases.md"],
            ["Report requirements", "This 4-page report follows Introduction, KB, Retrieval, Agent, Experiments, Results, Failures, Conclusion"],
            ["Reproducibility", "Run python app.py; run /api/evaluate or Evaluation tab; artifacts are regenerated automatically"],
        ], [2300, 5400], 12),
    ]

    sect = '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="500" w:right="500" w:bottom="500" w:left="500" w:header="300" w:footer="300" w:gutter="0"/></w:sectPr>'
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
<w:body>{''.join(body)}{sect}</w:body></w:document>'''


def styles_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="16"/></w:rPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:color="DEDED8"/><w:left w:val="single" w:sz="4" w:color="DEDED8"/><w:bottom w:val="single" w:sz="4" w:color="DEDED8"/><w:right w:val="single" w:sz="4" w:color="DEDED8"/><w:insideH w:val="single" w:sz="4" w:color="DEDED8"/><w:insideV w:val="single" w:sz="4" w:color="DEDED8"/></w:tblBorders></w:tblPr></w:style>
</w:styles>'''


def create_docx():
    create_architecture_image()
    create_ui_image()
    create_chart_image()
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
    doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/system_architecture.png"/>
<Relationship Id="rId6" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/system_ui_screenshot.png"/>
<Relationship Id="rId7" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/ablation_comparison.png"/>
</Relationships>'''
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document_xml())
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.write(ARCH_IMG, "word/media/system_architecture.png")
        z.write(UI_IMG, "word/media/system_ui_screenshot.png")
        z.write(CHART_IMG, "word/media/ablation_comparison.png")
    print(OUT)


if __name__ == "__main__":
    create_docx()
