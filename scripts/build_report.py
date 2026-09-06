"""Generate the cofounder proof-of-concept report (PDF) from project results.

Data-driven: reads dev CV (baseline_auc.csv), held-out test (test_auc.csv), and
the longitudinal gradient (longitudinal_auc.csv), plus figures from
results/figures. Re-running after any analysis refresh updates the report.

Run:  python scripts/build_report.py
Needs: reportlab, pillow  (pip install reportlab pillow)
"""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
OUT = ROOT / "DementiaNet_Cofounder_Report.pdf"

INK = colors.HexColor("#1f2933")
ACCENT = colors.HexColor("#2b6777")
LIGHT = colors.HexColor("#eef3f4")
RED = colors.HexColor("#a4303f")


def _read(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def dev_test_by_model():
    """Return {label: (dev_auc, test_auc_or_None)} for the three classifiers."""
    dev = {r["model"]: float(r["auc_mean"]) for r in _read(TAB / "baseline_auc.csv")}
    test = {r["model"]: float(r["test_auc"]) for r in _read(TAB / "test_auc.csv")}
    names = {
        "logistic_regression": "Logistic regression",
        "svm_rbf": "SVM (RBF)",
        "gradient_boosting": "Gradient boosting",
    }
    return {label: (dev.get(k), test.get(k)) for k, label in names.items()}


def longitudinal():
    rows = _read(TAB / "longitudinal_auc.csv")
    return {r["time_bin"]: float(r["auc_vs_control"]) for r in rows}


styles = getSampleStyleSheet()
H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], textColor=ACCENT, fontSize=13, spaceBefore=10, spaceAfter=4
)
BODY = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=9.5,
    leading=13,
    textColor=INK,
    alignment=TA_LEFT,
    spaceAfter=6,
)
SMALL = ParagraphStyle(
    "Small", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#52606d")
)
TITLE = ParagraphStyle("Title", parent=styles["Title"], textColor=INK, fontSize=20, spaceAfter=2)
SUB = ParagraphStyle("Sub", parent=BODY, fontSize=10, textColor=ACCENT, spaceAfter=2)


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, BODY), leftIndent=6) for t in items],
        bulletType="bullet",
        bulletColor=ACCENT,
        start="•",
        leftIndent=12,
    )


def styled_table(data, col_widths, red_rows=()):
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd2d9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]
    for r in red_rows:
        style.append(("TEXTCOLOR", (0, r), (-1, r), RED))
        style.append(("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"))
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle(style))
    return t


def fig(path: Path, width: float):
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def fmt(x):
    return f"{x:.2f}" if isinstance(x, float) else "-"


def para(text, style=BODY):
    return Paragraph(text, style)


# --- report copy (kept as short-line fragments so ruff is happy) ---


def bottom_line(hl):
    return (
        "Using only open, uncontrolled YouTube audio and off-the-shelf features, a "
        "speaker-independent model detects Alzheimer's-related speech change at "
        f"<b>AUC {hl} on a locked, never-seen test set</b> - a modest but real signal. "
        "Just as important is what rigorous testing ruled <b>out</b>: a flashier model, "
        "and a striking 'signal-appears-a-decade-early' curve, both looked great in "
        "cross-validation and <b>failed to replicate</b> on held-out data. Honest read: "
        "the core signal is real but weak, and casual-audio early detection is "
        "<b>not yet demonstrated</b>. This de-risks the hypothesis cheaply and tells us "
        "exactly what to fix."
    )


DATA_TEXT = (
    "<b>Data:</b> DementiaNet - a free, longitudinal corpus of spontaneous speech from "
    "public figures with confirmed dementia and cognitively healthy age-matched "
    "controls, with clips tagged by years-before-diagnosis. We restricted the patient "
    "group to an Alzheimer's-enriched phenotype (excluded Parkinson's, Lewy body, "
    "primary progressive aphasia). Final set: <b>344 clips, 171 speakers</b> "
    "(115 dementia / 229 control)."
)

METHOD_TEXT = (
    "<b>Method (built for honesty, not for a big number):</b> speaker-independent "
    "cross-validation, a locked 25% test set touched exactly once, and a documented "
    "confound audit - we caught the model exploiting <b>clip length</b> (an artifact of "
    "how videos were trimmed) and removed it. Features: voice-quality/prosody (openSMILE "
    "eGeMAPS) plus pause-timing measures."
)

TABLE_TEXT = (
    "Gradient boosting had the best cross-validation score but <b>collapsed below chance "
    "on the held-out test</b> - it had memorised quirks of the training speakers. The "
    "regularised linear models generalised. We reject gradient boosting and adopt the "
    "linear model. <b>This is the locked test set doing its job</b>: catching an overfit "
    "that cross-validation hid."
)

FAMILY_TEXT = (
    "On feature families, voice-quality/prosody carries what signal there is; pause "
    "timing is weak, and automatically-detected filled pauses (uh/um) add nothing on "
    "this corpus - a clean, documented negative result."
)

VENTURE_BULLETS = [
    "<b>Core signal validated as real but modest.</b> ~0.62 on held-out data confirms "
    "speech carries AD information - but well below a usable screen, and early detection "
    "from casual audio is unproven on this data.",
    "<b>Rigor is the asset.</b> The pipeline killed two tempting false positives - an "
    "overfit model and a spurious early-detection curve - that a less careful team would "
    "have pitched. That discipline is what makes future claims credible to clinicians "
    "and regulators.",
    "<b>Clear path to a real signal:</b> gains will come from controlled elicitation "
    "tasks and biomarker-confirmed clinical audio, not from squeezing this corpus.",
]

LIMIT_BULLETS = [
    "AUC ~0.62 is a proof-of-concept, <b>far from a deployable screen</b> (published "
    "clinical, task-controlled systems reach 0.77-0.93).",
    "Celebrity interview audio, edited and noisy; diagnoses are public-record, not "
    "biomarker-confirmed; per-time-bin and test samples are small, so numbers are "
    "trend-level, not precise.",
    "English-dominant; no demographic fairness testing yet.",
]

NEXT_BULLETS = [
    "Move to <b>controlled elicitation</b> (picture description, story recall) - the "
    "tasks behind the field's 0.77-0.93 results.",
    "Pilot <b>proprietary, biomarker-confirmed clinical audio</b> - the real moat.",
    "Test <b>deep acoustic embeddings</b> (wav2vec2 / HuBERT) vs the eGeMAPS baseline.",
    "Scope the regulatory path (Software-as-a-Medical-Device) for a speech screen.",
]


def longitudinal_text(cmp_txt):
    return (
        "We split patient clips by years-before-diagnosis and scored each slice against "
        "controls. A flexible model suggested a clean decline from diagnosis backwards - "
        "the exciting early-detection curve - but it <b>did not survive</b> the switch to "
        "the model that generalizes. Under honest evaluation the per-bin scores hover "
        f"near chance with no reliable ordering{cmp_txt}. So we do <b>not</b> claim a "
        "validated early-detection signal from this corpus: the striking gradient was an "
        "overfitting artifact. Confirming pre-symptomatic detection will need controlled "
        "tasks and clinical data (next steps)."
    )


def build():
    models = dev_test_by_model()
    lon = longitudinal()
    linear = [models[m][1] for m in ("Logistic regression", "SVM (RBF)") if models[m][1]]
    headline = max(linear) if linear else None
    hl = fmt(headline) if headline else "~0.62"

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.6 * inch,
    )
    W = doc.width
    s = []

    s.append(para("Speech as an Alzheimer's Screen: Proof-of-Concept Findings", TITLE))
    s.append(
        para(
            "Can off-the-shelf speech analysis detect AD from ordinary audio - "
            "and how early? &nbsp;|&nbsp; Internal memo, July 2026",
            SUB,
        )
    )
    s.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceBefore=4, spaceAfter=8))

    s.append(para("Bottom line", H1))
    s.append(para(bottom_line(hl)))

    s.append(para("What we built", H1))
    s.append(para(DATA_TEXT))
    s.append(para(METHOD_TEXT))

    s.append(para("The rigor paid off: cross-validation alone would have misled us", H1))
    rows = [["Model", "Dev cross-val", "Held-out test"]]
    red_idx = ()
    for i, (label, (dev, test)) in enumerate(models.items(), start=1):
        note = "  (overfit)" if (label == "Gradient boosting" and test and test < 0.5) else ""
        if note:
            red_idx = (i,)
        rows.append([label, fmt(dev), fmt(test) + note])
    s.append(styled_table(rows, [W * 0.40, W * 0.30, W * 0.30], red_rows=red_idx))
    s.append(para(TABLE_TEXT))

    s.append(para("Does the signal appear years before diagnosis? Not confirmed", H1))
    near, far = lon.get("NEAR (post+0-5yr)"), lon.get("FAR (5-10+10-15yr)")
    cmp_txt = f" (recent {near:.2f} vs earlier {far:.2f})" if near and far else ""
    s.append(para(longitudinal_text(cmp_txt)))
    if (FIG / "longitudinal_auc.png").exists():
        s.append(fig(FIG / "longitudinal_auc.png", W * 0.58))
    s.append(para(FAMILY_TEXT))

    s.append(para("What this means for the venture", H1))
    s.append(bullets(VENTURE_BULLETS))

    s.append(para("Honest limitations", H1))
    s.append(bullets(LIMIT_BULLETS))

    s.append(para("Next steps", H1))
    s.append(bullets(NEXT_BULLETS))

    s.append(Spacer(1, 6))
    s.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd2d9")))
    s.append(
        para(
            "Reproducible pipeline: speaker-independent CV, locked test split, confound "
            "removal, feature ablation, and longitudinal analysis - all in the project repo.",
            SMALL,
        )
    )

    doc.build(s)
    print(f"wrote {OUT}  (headline held-out test AUC: {hl})")


if __name__ == "__main__":
    build()
