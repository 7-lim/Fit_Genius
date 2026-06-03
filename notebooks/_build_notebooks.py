"""
Generator for the Fit Genius dataset-exploration notebooks (build-order step 1).

Run from anywhere:  python notebooks/_build_notebooks.py
It writes:
    notebooks/explore_squat_dataset.ipynb
    notebooks/explore_deadlift_dataset.ipynb
Then execute them with nbconvert to embed outputs.

This file is just the authoring source; the notebooks are the deliverable.
"""
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- shared setup
SETUP = r'''
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# Locate the repo root (the folder that contains `backend/`) regardless of cwd.
root = Path.cwd()
while not (root / "backend").exists() and root != root.parent:
    root = root.parent
DATA = root / "backend" / "training" / "data"
FIGS = (root / "notebooks" / "figures"); FIGS.mkdir(parents=True, exist_ok=True)
print("repo root :", root)
print("data dir  :", DATA)

LANDMARK_COLS = [f"{a}{i}" for i in range(1, 34) for a in ("x", "y", "z", "v")]
print("expected landmark feature count:", len(LANDMARK_COLS), "(33 landmarks x [x,y,z,v])")
'''

# ---------------------------------------------------------------- squat helpers
SQ_LOAD = r'''
# --- Source A: raw MediaPipe landmark coordinates (real footage) ---
sq = pd.read_csv(DATA / "CSV_files" / "coords_SQ_C.csv")
print("coords_SQ_C.csv  shape:", sq.shape)
print("label column     :", sq.columns[0])
print("feature columns  :", sq.shape[1] - 1, "(should be 132 landmark features)")
print("null cells       :", int(sq.isnull().sum().sum()))
sq.head(3)
'''

SQ_DIST = r'''
# Class distribution -- each class encodes PHASE + FORM together
counts = sq["class"].value_counts()
print(counts.to_string())
print("\ntotal frames:", len(sq))

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
for i, v in enumerate(counts.values):
    ax.text(v + 3, i, f"{v}  ({v/len(sq)*100:.1f}%)", va="center", fontsize=9)
ax.set(title="Squat (coords_SQ_C) - class distribution", xlabel="frames", ylabel="class")
plt.tight_layout(); plt.savefig(FIGS / "squat_raw_class_dist.png"); plt.show()
'''

SQ_PHASE = r'''
# Decompose the combined class into PHASE (up/down) + FORM tag
def phase_of(c):  return "up" if c.startswith("up") else "down"
def form_of(c):
    parts = c.split("_", 1)
    return "correct" if len(parts) == 1 else parts[1]

sq["phase"] = sq["class"].map(phase_of)
sq["form"]  = sq["class"].map(form_of)

print("PHASE balance:");        print(sq["phase"].value_counts(), "\n")
print("FORM tag balance:");     print(sq["form"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
sq["phase"].value_counts().plot.bar(ax=axes[0], color=["#4c72b0", "#dd8452"], rot=0)
axes[0].set(title="Squat phase", ylabel="frames")
sq["form"].value_counts().plot.bar(ax=axes[1], color="#55a868", rot=0)
axes[1].set(title="Squat form tag")
plt.tight_layout(); plt.savefig(FIGS / "squat_raw_phase_form.png"); plt.show()
'''

SQ_VIS = r'''
# Quick landmark-quality sanity check: mean visibility per frame
vis_cols = [c for c in sq.columns if c.startswith("v")]
sq["mean_visibility"] = sq[vis_cols].mean(axis=1)
print(sq["mean_visibility"].describe().round(3))
ax = sq["mean_visibility"].plot.hist(bins=40, figsize=(7, 3.2), color="#8172b3")
ax.set(title="Mean landmark visibility per frame (squat)", xlabel="visibility")
plt.tight_layout(); plt.savefig(FIGS / "squat_visibility.png"); plt.show()
'''

SQ_AUG = r'''
# --- Source B: pre-engineered features, AUGMENTED (synthetic) ---
aug = pd.read_csv(DATA / "squat_dataset" / "squat_features_augmented.csv")
print("squat_features_augmented.csv  shape:", aug.shape)
print("\ncolumns:", list(aug.columns))
print("\nNOTE: these are computed ANGLES/metrics, NOT raw 132-dim landmarks.")
aug.head(3)
'''

SQ_AUG_DIST = r'''
lc = aug["label"].value_counts().sort_index()
print("label counts (numeric 0-5):"); print(lc.to_string())
print("\n-> perfectly balanced:", lc.nunique() == 1, "(", lc.iloc[0], "each ) -- a hallmark of synthetic augmentation")
print("source videos:", aug["video_file"].nunique(), "clips, all 'co' (correct-form) recordings:",
      sorted(aug['video_file'].unique())[:5], "...")

fig, ax = plt.subplots(figsize=(7, 3.4))
sns.barplot(x=lc.index.astype(str), y=lc.values, ax=ax, palette="rocket")
ax.set(title="Augmented squat - label distribution (balanced)", xlabel="label", ylabel="rows")
plt.tight_layout(); plt.savefig(FIGS / "squat_aug_label_dist.png"); plt.show()
'''

SQ_AUG_DECODE = r'''
# Decode what each numeric label means by looking at per-label feature means.
feat = [c for c in aug.columns if c not in ("video_file", "frame", "label")]
means = aug.groupby("label")[feat].mean().round(2)
print(means.T)

# z-score each feature across labels to see which feature each label perturbs
z = (means - means.mean()) / means.std(ddof=0)
fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(z.T, center=0, cmap="coolwarm", annot=means.T, fmt=".1f",
            cbar_kws={"label": "z-score across labels"}, ax=ax)
ax.set(title="Augmented squat - per-label feature signature\n(annot = raw mean, colour = z-score)",
       xlabel="label", ylabel="feature")
plt.tight_layout(); plt.savefig(FIGS / "squat_aug_label_signature.png"); plt.show()
'''

# ------------------------------------------------------------- deadlift helpers
DL_LOAD = r'''
dl = pd.read_csv(DATA / "CSV_files" / "coords_DL_C.csv")
print("coords_DL_C.csv  shape:", dl.shape)
print("feature columns :", dl.shape[1] - 1, "(132 landmark features expected)")
print("null cells      :", int(dl.isnull().sum().sum()))
dl.head(3)
'''

DL_DIST = r'''
counts = dl["class"].value_counts()
print(counts.to_string()); print("\ntotal frames:", len(dl))

fig, ax = plt.subplots(figsize=(7, 4.2))
sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="mako")
for i, v in enumerate(counts.values):
    ax.text(v + 3, i, f"{v}  ({v/len(dl)*100:.1f}%)", va="center", fontsize=9)
ax.set(title="Deadlift (coords_DL_C) - class distribution", xlabel="frames", ylabel="class")
plt.tight_layout(); plt.savefig(FIGS / "deadlift_raw_class_dist.png"); plt.show()
'''

DL_PHASE = r'''
def phase_of(c):  return "up" if c.startswith("up") else "down"
def form_of(c):
    parts = c.split("_", 1)
    return "correct" if len(parts) == 1 else parts[1]

dl["phase"] = dl["class"].map(phase_of)
dl["form"]  = dl["class"].map(form_of)
print("PHASE balance:"); print(dl["phase"].value_counts(), "\n")
print("FORM tag balance:"); print(dl["form"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
dl["phase"].value_counts().plot.bar(ax=axes[0], color=["#4c72b0", "#dd8452"], rot=0)
axes[0].set(title="Deadlift phase", ylabel="frames")
dl["form"].value_counts().plot.bar(ax=axes[1], color="#c44e52", rot=0)
axes[1].set(title="Deadlift form tag")
plt.tight_layout(); plt.savefig(FIGS / "deadlift_raw_phase_form.png"); plt.show()
'''

DL_BENCH = r'''
# A bench-press file (coords_BP_C.csv) also ships with the dataset.
# Bench press is NOT a Fit Genius target exercise -- shown here only for awareness.
bp = pd.read_csv(DATA / "CSV_files" / "coords_BP_C.csv")
print("coords_BP_C.csv shape:", bp.shape, "(NOT a target exercise)")
print(bp["class"].value_counts().to_string())
'''


def md(text):   return nbf.v4.new_markdown_cell(text.strip("\n"))
def code(text): return nbf.v4.new_code_cell(text.strip("\n"))


# ============================================================= SQUAT NOTEBOOK
sq_nb = nbf.v4.new_notebook()
sq_nb.cells = [
    md("""
# Fit Genius — Squat Dataset Exploration

**Build-order step 1.** Understand the squat data and its label distributions before any training.

Two distinct squat sources ship with the project:

| Source | File | Representation | Nature |
|---|---|---|---|
| **A** | `CSV_files/coords_SQ_C.csv` | raw 132-dim MediaPipe landmarks + combined phase/form `class` | real footage |
| **B** | `squat_dataset/squat_features_augmented.csv` | 12 engineered angles/metrics + numeric `label` | **synthetic (augmented)** |

These are *not* interchangeable — A matches the LSTM pipeline in `CLAUDE.md` (raw landmarks → angles),
B is a balanced, feature-engineered set derived only from correct-form clips.
"""),
    code(SETUP),
    md("## Source A — raw MediaPipe landmarks (`coords_SQ_C.csv`)"),
    code(SQ_LOAD),
    md("### Class distribution\nEach `class` value bundles the **phase** (up/down) with a **form** tag."),
    code(SQ_DIST),
    md("### Phase vs. form breakdown"),
    code(SQ_PHASE),
    md("### Landmark quality check\nLow visibility means MediaPipe was unsure — useful to know before windowing."),
    code(SQ_VIS),
    md("""
## Source B — augmented engineered features (`squat_features_augmented.csv`)

⚠️ This file holds **computed angles**, not raw landmarks, and the labels are perfectly balanced —
a strong sign it was **synthetically augmented** from correct-form videos.
"""),
    code(SQ_AUG),
    md("### Label distribution"),
    code(SQ_AUG_DIST),
    md("""
### Decoding the numeric labels (0–5)

There is no label legend file, so we infer meaning from each label's feature signature.
Each label appears to perturb **one** feature group away from the correct baseline:
"""),
    code(SQ_AUG_DECODE),
    md("""
### Read of the augmented set (interpretation)

- **0** baseline / correct
- **1** knee & hip angle more open → *insufficient depth (shallow squat)*
- **2** lower spine angle + higher torso lean → *excessive forward lean / back rounding*
- **3** knee angle very closed + higher knee-lateral → *too deep + knees caving*
- **4** raised ankle angle → *heels lifting / ankle issue*
- **5** ≈ baseline (near-identical to 0) → likely *noise-augmented correct*

`hip_depth` and `symmetry_score` are **constant across all labels** — they were never perturbed.
This is decent for prototyping but synthetic single-feature errors may not generalise to real lifters.
"""),
    md("""
## Summary — squat

- **Source A (real, raw landmarks):** 1,303 frames, 4 classes, no nulls — matches the CLAUDE.md pipeline.
  Imbalanced (`up` ≈ 40%, `down_forward` ≈ 15%) and **no `frame`/`video` column**, so temporal
  windowing assumes rows are already in capture order within the file.
- **Source B (synthetic, engineered):** 47,442 rows, 6 perfectly-balanced labels from 15 correct-form
  clips — feature vectors only, not raw landmarks.
- **Open question:** which source drives the squat LSTM? They imply different model inputs.
"""),
]
nbf.write(sq_nb, OUT / "explore_squat_dataset.ipynb")
print("wrote explore_squat_dataset.ipynb")

# ========================================================== DEADLIFT NOTEBOOK
dl_nb = nbf.v4.new_notebook()
dl_nb.cells = [
    md("""
# Fit Genius — Deadlift Dataset Exploration

**Build-order step 1.** Understand the deadlift data and its label distributions before training.

Source: `CSV_files/coords_DL_C.csv` — raw 132-dim MediaPipe landmarks with a combined
phase/form `class` label (DanielGuarnizo repo style). A bench-press file ships alongside it
but bench press is **not** a Fit Genius target exercise.
"""),
    code(SETUP),
    md("## Raw MediaPipe landmarks (`coords_DL_C.csv`)"),
    code(DL_LOAD),
    md("### Class distribution\nClasses bundle **phase** (up/down) with a **form** tag (low, roll, back …)."),
    code(DL_DIST),
    md("### Phase vs. form breakdown"),
    code(DL_PHASE),
    md("### Aside — bench-press file (not a target exercise)"),
    code(DL_BENCH),
    md("""
## Summary — deadlift

- 2,478 frames, **6 classes**, no nulls, full 132-dim landmark features — richest of the raw sets.
- Form errors captured: **low** (insufficient hip height), **roll** (back rounding), **back**
  (hyperextension on lockout). Phase split is roughly balanced (`up`-family vs `down`-family).
- Like the squat raw set, there is **no `frame`/`video` column**, so sequence windowing relies on
  in-file row order.
- `coords_BP_C.csv` (bench press) is present but out of scope for the squat/deadlift models.
"""),
]
nbf.write(dl_nb, OUT / "explore_deadlift_dataset.ipynb")
print("wrote explore_deadlift_dataset.ipynb")
