"""Generate all four figures for the RAG faithfulness benchmark paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

FIGURES_DIR = "writeup/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# 6 models, new 1000-example data
MODELS = ["llama3.2:3b", "llama3.1:8b", "mistral:7b", "qwen2.5:7b", "gpt-4o-mini", "claude-haiku"]
CONDITIONS = ["Original", "Paraphrase", "Entity swap", "Negation"]

# ── Figure 1: Hallucination bars ──────────────────────────────────────────────
# rows = models, cols = [original, paraphrase, entity_swap, negation]
halluc_data = np.array([
    [.04, .05, .07, .04],   # llama3.2:3b
    [.03, .04, .07, .04],   # llama3.1:8b
    [.03, .03, .07, .04],   # mistral:7b
    [.04, .04, .09, .05],   # qwen2.5:7b
    [.03, .05, .07, .05],   # gpt-4o-mini
    [.01, .00, .02, .00],   # claude-haiku
])

fig, ax = plt.subplots(figsize=(9, 3.8))
x = np.arange(len(MODELS))
width = 0.18
colors = ["#4e8098", "#90c2e7", "#c44d34", "#e8a838"]

for i, (cond, color) in enumerate(zip(CONDITIONS, colors)):
    offset = (i - 1.5) * width
    ax.bar(x + offset, halluc_data[:, i], width, label=cond, color=color,
           edgecolor="white", linewidth=0.5)

ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Hallucination rate", fontsize=11)
ax.set_ylim(0, 0.15)
ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=9)
ax.legend(title="Condition", fontsize=9, title_fontsize=9, loc="upper right")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig1_hallucination_bars.pdf", bbox_inches="tight")
plt.close()
print("Fig 1 saved.")

# ── Figure 2: Abstention heatmap ──────────────────────────────────────────────
# rows = models, cols = [original, paraphrase, entity_swap, negation]
abstain_data = np.array([
    [.14, .20, .30, .32],   # llama3.2:3b
    [.10, .18, .28, .31],   # llama3.1:8b
    [.13, .15, .26, .43],   # mistral:7b
    [.16, .19, .37, .43],   # qwen2.5:7b
    [.14, .20, .32, .44],   # gpt-4o-mini
    [.16, .23, .35, .46],   # claude-haiku
])

fig, ax = plt.subplots(figsize=(7.5, 4.2))
im = ax.imshow(abstain_data, cmap="Blues", vmin=0.05, vmax=0.55, aspect="auto")
ax.set_xticks(range(4))
ax.set_yticks(range(6))
ax.set_xticklabels(CONDITIONS, fontsize=10)
ax.set_yticklabels(MODELS, fontsize=10)
for i in range(6):
    for j in range(4):
        val = abstain_data[i, j]
        color = "white" if val > 0.38 else "black"
        ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                color=color, fontsize=11, fontweight="bold")
cbar = plt.colorbar(im, ax=ax, label="Abstention rate")
cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig2_abstention_heatmap.pdf", bbox_inches="tight")
plt.close()
print("Fig 2 saved.")

# ── Figure 3: Cliff scores ────────────────────────────────────────────────────
# sorted ascending so highest bar is at top
cliff_models  = ["llama3.2:3b", "llama3.1:8b", "mistral:7b",
                 "qwen2.5:7b",  "gpt-4o-mini", "claude-haiku"]
cliff_scores  = [0.173, 0.192, 0.222, 0.234, 0.241, 0.244]
cliff_colors  = ["#4e8098", "#90c2e7", "#c44d34",
                 "#e8a838", "#7b9e3f", "#6b4c9a"]
cliff_types   = ["Open-weight", "Open-weight", "Open-weight",
                 "Open-weight", "Closed-source", "Closed-source"]
sig_labels    = ["***", "***", "***", "***", "***", "***"]

fig, ax = plt.subplots(figsize=(6.5, 3.8))
y = np.arange(len(cliff_models))
bars = ax.barh(y, cliff_scores, color=cliff_colors,
               edgecolor="white", linewidth=0.5, height=0.5)

for i, (score, sig, mtype) in enumerate(zip(cliff_scores, sig_labels, cliff_types)):
    ax.text(score + 0.005, i, sig, va="center", fontsize=11,
            fontweight="bold", color="#333333")
    if mtype == "Closed-source":
        ax.text(-0.005, i, "●", va="center", ha="right",
                fontsize=8, color="#555")

ax.axvline(x=0, color="black", linewidth=1.0, linestyle="-")
ax.axvline(x=0.70, color="#888888", linewidth=1.0, linestyle="--",
           label="Ideal cliff (≈0.70)")

ax.set_yticks(y)
ax.set_yticklabels(cliff_models, fontsize=10)
ax.set_xlabel(r"$\Delta_\mathrm{abstain}$", fontsize=11)
ax.set_xlim(0, 0.82)
ax.legend(fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.4)

# Add type annotations on right side
for i, mtype in enumerate(cliff_types):
    label = "CS" if mtype == "Closed-source" else "OW"
    color = "#6b4c9a" if mtype == "Closed-source" else "#555555"
    ax.text(0.81, i, label, va="center", ha="left",
            fontsize=7, color=color,
            transform=ax.get_yaxis_transform())

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3_cliff_scores.pdf", bbox_inches="tight")
plt.close()
print("Fig 3 saved.")

# ── Figure 4: Taxonomy scatter ────────────────────────────────────────────────
# (faithfulness, abstention) per (model, condition) — new 1000-example values
faith_abstain = {
    "llama3.2:3b": [(.74,.14), (.71,.20), (.57,.30), (.51,.32)],
    "llama3.1:8b": [(.79,.10), (.74,.18), (.62,.28), (.55,.31)],
    "mistral:7b":  [(.67,.13), (.63,.15), (.51,.26), (.40,.43)],
    "qwen2.5:7b":  [(.74,.16), (.70,.19), (.53,.37), (.44,.43)],
    "gpt-4o-mini": [(.74,.14), (.69,.20), (.56,.32), (.45,.44)],
    "claude-haiku":[(.72,.16), (.67,.23), (.54,.35), (.47,.46)],
}
markers = {
    "llama3.2:3b": "o",
    "llama3.1:8b": "s",
    "mistral:7b":  "^",
    "qwen2.5:7b":  "D",
    "gpt-4o-mini": "P",
    "claude-haiku":"*",
}
cond_colors = ["#2166ac", "#74add1", "#d73027", "#f46d43"]

fig, ax = plt.subplots(figsize=(7, 5.5))

# Shaded archetype regions
ax.axhspan(0.35, 0.75, xmin=0, xmax=1, color="#fddbc7", alpha=0.35, zorder=0)
ax.fill_betweenx([0.10, 0.35], 0.45, 0.85, color="#d1e5f0", alpha=0.45, zorder=0)
ax.fill_betweenx([0.55, 0.75], 0.10, 0.42, color="#b8e186", alpha=0.45, zorder=0)

# Dividing lines
ax.axhline(y=0.35, color="#999999", linewidth=0.8, linestyle="--")
ax.axvline(x=0.45, color="#999999", linewidth=0.8, linestyle="--", ymax=(0.25/0.65))
ax.axhline(y=0.55, color="#999999", linewidth=0.8, linestyle="--", xmax=(0.35/0.75))

# Region labels
ax.text(0.63, 0.20, "Context-\nfollower", ha="center", va="center",
        fontsize=8.5, color="#1a6690", style="italic")
ax.text(0.22, 0.63, "Ideal", ha="center", va="center",
        fontsize=8.5, color="#4d7c2e", style="italic")
ax.text(0.62, 0.62, "Over-\nabstainer", ha="center", va="center",
        fontsize=8.5, color="#a63520", style="italic")

# Plot points
for model, pts in faith_abstain.items():
    for ci, (fx, ay) in enumerate(pts):
        ax.scatter(fx, ay, marker=markers[model], color=cond_colors[ci],
                   s=100, zorder=5, edgecolors="white", linewidths=0.6)

# Legend
model_handles = [
    mpatches.Patch(color="none", label="Model:"),
    plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#555", markersize=8, label="llama3.2:3b"),
    plt.Line2D([0],[0], marker="s", color="w", markerfacecolor="#555", markersize=8, label="llama3.1:8b"),
    plt.Line2D([0],[0], marker="^", color="w", markerfacecolor="#555", markersize=8, label="mistral:7b"),
    plt.Line2D([0],[0], marker="D", color="w", markerfacecolor="#555", markersize=8, label="qwen2.5:7b"),
    plt.Line2D([0],[0], marker="P", color="w", markerfacecolor="#555", markersize=8, label="gpt-4o-mini"),
    plt.Line2D([0],[0], marker="*", color="w", markerfacecolor="#555", markersize=10, label="claude-haiku"),
    mpatches.Patch(color="none", label=" "),
    mpatches.Patch(color="none", label="Condition:"),
    mpatches.Patch(color=cond_colors[0], label="Original"),
    mpatches.Patch(color=cond_colors[1], label="Paraphrase"),
    mpatches.Patch(color=cond_colors[2], label="Entity swap"),
    mpatches.Patch(color=cond_colors[3], label="Negation"),
]
ax.legend(handles=model_handles, fontsize=8, loc="upper right",
          borderaxespad=0.5, labelspacing=0.3)

ax.set_xlabel("Faithfulness rate", fontsize=11)
ax.set_ylabel("Abstention rate", fontsize=11)
ax.set_xlim(0.10, 0.85)
ax.set_ylim(0.10, 0.75)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig4_taxonomy_scatter.pdf", bbox_inches="tight")
plt.close()
print("Fig 4 saved.")

print("\nAll figures saved to", FIGURES_DIR)