"""Generate all four figures for the RAG faithfulness benchmark paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

FIGURES_DIR = "writeup/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS = ["llama3.2:3b", "llama3.1:8b", "mistral:7b", "qwen2.5:7b"]
CONDITIONS = ["Original", "Paraphrase", "Entity swap", "Negation"]
MODEL_LABELS = ["llama3.2:3b", "llama3.1:8b", "mistral:7b", "qwen2.5:7b"]

halluc_data = np.array([
    [.06, .11, .09, .05],   # llama3.2:3b
    [.05, .10, .12, .07],   # llama3.1:8b
    [.04, .05, .09, .05],   # mistral:7b
    [.07, .10, .12, .08],   # qwen2.5:7b
])

fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(len(MODELS))
width = 0.18
colors = ["#4e8098", "#90c2e7", "#c44d34", "#e8a838"]
cond_labels = CONDITIONS

for i, (cond, color) in enumerate(zip(cond_labels, colors)):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, halluc_data[:, i], width, label=cond, color=color,
                  edgecolor="white", linewidth=0.5)

ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Hallucination rate", fontsize=11)
ax.set_ylim(0, 0.20)
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS, fontsize=9)
ax.legend(title="Condition", fontsize=9, title_fontsize=9, loc="upper left")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig1_hallucination_bars.pdf", bbox_inches="tight")
plt.close()
print("Fig 1 saved.")


abstain_data = np.array([
    [.27, .40, .43, .43],   # llama3.2:3b
    [.26, .39, .48, .43],   # llama3.1:8b
    [.23, .35, .39, .58],   # mistral:7b
    [.29, .33, .52, .57],   # qwen2.5:7b
])

fig, ax = plt.subplots(figsize=(6.5, 3.2))
im = ax.imshow(abstain_data, cmap="Blues", vmin=0.15, vmax=0.65, aspect="auto")
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(CONDITIONS, fontsize=10)
ax.set_yticklabels(MODEL_LABELS, fontsize=10)
for i in range(4):
    for j in range(4):
        val = abstain_data[i, j]
        color = "white" if val > 0.45 else "black"
        ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                color=color, fontsize=11, fontweight="bold")
cbar = plt.colorbar(im, ax=ax, label="Abstention rate")
cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig2_abstention_heatmap.pdf", bbox_inches="tight")
plt.close()
print("Fig 2 saved.")


cliff_models = ["mistral:7b", "qwen2.5:7b", "llama3.1:8b", "llama3.2:3b"]
cliff_scores = [0.250, 0.250, 0.197, 0.157]
cliff_colors = ["#c44d34", "#e8a838", "#90c2e7", "#4e8098"]
sig_labels = ["***", "***", "***", "**"]

fig, ax = plt.subplots(figsize=(6, 3.2))
y = np.arange(len(cliff_models))
bars = ax.barh(y, cliff_scores, color=cliff_colors, edgecolor="white", linewidth=0.5, height=0.5)

# Significance stars
for i, (score, sig) in enumerate(zip(cliff_scores, sig_labels)):
    ax.text(score + 0.005, i, sig, va="center", fontsize=11, fontweight="bold", color="#333333")

# Reference lines
ax.axvline(x=0, color="black", linewidth=1.0, linestyle="-")
ax.axvline(x=0.70, color="#888888", linewidth=1.0, linestyle="--", label="Ideal cliff (≈0.70)")

ax.set_yticks(y)
ax.set_yticklabels(cliff_models, fontsize=10)
ax.set_xlabel(r"$\Delta_\mathrm{abstain}$", fontsize=11)
ax.set_xlim(0, 0.82)
ax.legend(fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3_cliff_scores.pdf", bbox_inches="tight")
plt.close()
print("Fig 3 saved.")


faith_abstain = {
    "llama3.2:3b": [(.54,.27), (.50,.40), (.34,.43), (.36,.43)],
    "llama3.1:8b": [(.57,.26), (.51,.39), (.36,.48), (.40,.43)],
    "mistral:7b":  [(.38,.23), (.38,.35), (.27,.39), (.23,.58)],
    "qwen2.5:7b":  [(.53,.29), (.48,.33), (.34,.52), (.30,.57)],
}
markers = {"llama3.2:3b": "o", "llama3.1:8b": "s", "mistral:7b": "^", "qwen2.5:7b": "D"}
cond_colors = ["#2166ac", "#74add1", "#d73027", "#f46d43"]   # orig, para, entity_swap, negation

fig, ax = plt.subplots(figsize=(6.5, 5))


ax.axhspan(0.35, 0.72, color="#fddbc7", alpha=0.45, zorder=0)
# Context-follower: x > 0.45, y < 0.35 (light blue)
ax.fill_betweenx([0, 0.35], 0.45, 0.72, color="#d1e5f0", alpha=0.55, zorder=0)
# Ideal region: x < 0.40, y > 0.55
ax.fill_betweenx([0.55, 0.72], 0, 0.40, color="#b8e186", alpha=0.55, zorder=0)


ax.axhline(y=0.35, color="#999999", linewidth=0.8, linestyle="--")
ax.axvline(x=0.45, color="#999999", linewidth=0.8, linestyle="--", ymax=(0.35/0.72))
ax.axhline(y=0.55, color="#999999", linewidth=0.8, linestyle="--", xmax=(0.40/0.72))


ax.text(0.56, 0.16, "Context-\nfollower", ha="center", va="center",
        fontsize=8.5, color="#1a6690", style="italic")
ax.text(0.20, 0.615, "Ideal", ha="center", va="center",
        fontsize=8.5, color="#4d7c2e", style="italic")
ax.text(0.55, 0.58, "Over-\nabstainer", ha="center", va="center",
        fontsize=8.5, color="#a63520", style="italic")


for model, pts in faith_abstain.items():
    for ci, ((fx, ay), cond) in enumerate(zip(pts, CONDITIONS)):
        ax.scatter(fx, ay, marker=markers[model], color=cond_colors[ci],
                   s=90, zorder=5, edgecolors="white", linewidths=0.6)


model_handles = [
    mpatches.Patch(color="none", label="Model:"),
    plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#555", markersize=8, label="llama3.2:3b"),
    plt.Line2D([0],[0], marker="s", color="w", markerfacecolor="#555", markersize=8, label="llama3.1:8b"),
    plt.Line2D([0],[0], marker="^", color="w", markerfacecolor="#555", markersize=8, label="mistral:7b"),
    plt.Line2D([0],[0], marker="D", color="w", markerfacecolor="#555", markersize=8, label="qwen2.5:7b"),
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
ax.set_xlim(0.10, 0.72)
ax.set_ylim(0.10, 0.72)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig4_taxonomy_scatter.pdf", bbox_inches="tight")
plt.close()
print("Fig 4 saved.")

print("\nAll figures saved to", FIGURES_DIR)
