# -*- coding: utf-8 -*-
"""Render the 6 key figures of the DreamerV3 paper directly from the PDF (vector-faithful).

Strategy: for each target figure, locate its caption ("Figure N:"), then take the union of
all vector-drawing rects + raster-image rects that sit ABOVE the caption on that page.
Each target page carries exactly one figure, so that union is the figure body.
Render the clip at high DPI to a PNG. No content is invented - pure rasterization.
"""
import os
import fitz  # PyMuPDF

ROOT = r"d:\研究生\项目内容\杂事\光源智能体设计大赛\项目主体"
PDF = os.path.join(ROOT, "DreamerV3_Mastering_Diverse_Domains_through_World_Models.pdf")
OUT = os.path.join(ROOT, "_figs_stage")
os.makedirs(OUT, exist_ok=True)

# (figure_number, 0-based page index)
TARGETS = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 7), (6, 8)]
ZOOM = 3.5  # ~252 DPI


def figure_bbox(page, fig):
    hits = page.search_for(f"Figure {fig}:")
    if not hits:
        hits = page.search_for(f"Figure {fig}")
    if not hits:
        return None, None
    cap_top = min(h.y0 for h in hits)
    cap_bottom = max(h.y1 for h in hits)
    rects = []
    pw = page.rect.width
    for d in page.get_drawings():
        r = d["rect"]
        # drop margin artifacts (e.g. the page-1 vertical arXiv watermark strip)
        if r.x1 <= 55 or r.x0 >= pw - 55:
            continue
        if r.y1 <= cap_top + 2 and r.height > 0.5 and r.width > 0.5:
            rects.append(r)
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"])
        if r.x1 <= 55 or r.x0 >= pw - 55:
            continue
        if r.y1 <= cap_top + 2:
            rects.append(r)
    if not rects:
        return None, (cap_top, cap_bottom)
    x0 = min(r.x0 for r in rects)
    y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects)
    union = fitz.Rect(x0, y0, x1, cap_top)
    return union, (cap_top, cap_bottom)


def main():
    doc = fitz.open(PDF)
    print("pages:", doc.page_count)
    mat = fitz.Matrix(ZOOM, ZOOM)
    for fig, pno in TARGETS:
        page = doc[pno]
        pw, ph = page.rect.width, page.rect.height
        bbox, cap = figure_bbox(page, fig)
        if bbox is None:
            print(f"[WARN] Figure {fig} p{pno+1}: bbox not found (cap={cap})")
            continue
        pad = 5
        clip = fitz.Rect(
            max(58, bbox.x0 - pad), max(0, bbox.y0 - pad),
            min(pw, bbox.x1 + pad), min(ph, bbox.y1 + pad),
        )
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        outp = os.path.join(OUT, f"fig{fig}.png")
        pix.save(outp)
        print(f"[OK] Figure {fig} p{pno+1}: clip={tuple(round(v) for v in clip)} "
              f"-> {pix.width}x{pix.height}px  ({outp})")
    doc.close()


if __name__ == "__main__":
    main()
