#!/usr/bin/env python3
"""Draw the methodology architecture as a vector PDF from the stated model."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/methodology_framework.pdf"
fig, ax = plt.subplots(figsize=(7.10, 3.55))
ax.set_xlim(0, 14.2); ax.set_ylim(0, 7.1); ax.axis("off")
edge = "#435365"
palette = ["#EAF1F8", "#FFF3E3", "#E8F3ED", "#E9EFF8", "#F2ECF7", "#F7F0E6"]

def box(x, y, w, h, title, body, color, body_fs=7.35):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                       linewidth=0.9, edgecolor=edge, facecolor=color)
    ax.add_patch(p)
    ax.text(x+w/2, y+h-0.30, title, ha="center", va="center", fontsize=7.8,
            fontweight="bold", color="#1D2B39")
    ax.text(x+w/2, y+h/2-0.12, body, ha="center", va="center", fontsize=body_fs,
            color="#263645", linespacing=1.28)

def arrow(a, b, label=None, label_xy=None, color=edge, lw=1.0, ms=9):
    p = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                        color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(p)
    if label and label_xy:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=7.0,
                color="#354657", bbox=dict(boxstyle="round,pad=0.12",fc="white",ec="none",alpha=.96))

xs = [0.25, 3.78, 7.31, 10.84]
w, y, h = 3.10, 4.15, 1.58
box(xs[0],y,w,h,"Workload and client state",
    "writer partition and $n_i$\nparticipation debt; link rate\nKEM session age and cap",palette[0],6.6)
box(xs[1],y,w,h,"Per-client costs",
    "local training: energy and time\nmodel exchange: bytes and radio\nrefresh: ML-KEM and handshake",palette[1],6.6)
box(xs[2],y,w,h,"Admission rule",
    "utility per client-energy\nreject if latency exceeds $D$\nbudget $B$; at most $K$ clients",palette[2],6.6)
box(xs[3],y,w,h,"Federated training",
    "one local epoch per client\nsample-weighted FedAvg\nnew global model $w_{t+1}$",palette[3],6.6)
for i in range(3):
    arrow((xs[i]+w+0.04,y+h/2),(xs[i+1]-0.04,y+h/2))

# State carried from a completed round into the next round.
ax.plot([12.40,12.40,1.80,1.80],[5.77,6.54,6.54,5.77],color="#7A5B32",lw=1.0)
arrow((1.80,6.54),(1.80,5.78),"participation counts and refresh timestamp",
      label_xy=(7.10,6.74),color="#7A5B32",lw=1.0,ms=8)

# Measurement and evidence path.
box(3.25,1.35,4.15,1.75,"Evaluation",
    "validation writers every 5 rounds\nfinal-test writers scored once after round 100\nsample, writer, and class accuracy; fairness",palette[4],6.6)
box(8.20,1.35,4.55,1.75,"Campaign evidence",
    "6 policies; 5 paired seeds; 100 main rounds\n50-round session, link, budget, cohort sweeps\nround logs, run summaries, timing traces",palette[5],6.6)
arrow((12.38,4.11),(7.27,3.16),"round checkpoints / final model",
      label_xy=(10.45,3.54),color=edge,lw=1.0)
arrow((7.46,1.57),(8.15,1.57))

ax.text(7.1,0.62,
        "The learning run is executed on the complete writer partition. Device energy is calibrated from published profiles;\n"
        "ML-KEM and X25519 primitive latency and serialized sizes are separately measured on the host.",
        ha="center",va="center",fontsize=7.5,color="#435365",linespacing=1.3)
fig.subplots_adjust(left=.015,right=.985,bottom=.035,top=.985)
OUT.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT,bbox_inches="tight",pad_inches=.04)
plt.close(fig)
print(OUT)
