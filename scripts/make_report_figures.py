"""Generate 3 figures for STRATEGY.md:

1. fig_training_curves.png  — CER vs epoch, 3 runs on one plot
2. fig_ablation_waterfall.png — waterfall of component contributions
3. fig_speaker_evolution.png — per-speaker CER across milestones
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "report_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def plot_training_curves() -> None:
    data = json.loads((FIG_DIR / "epochs.json").read_text())

    scratch_42 = data["scratch_42"]            # epochs 1..20
    reverb_42 = data["reverb_cont_42"]          # epochs 21..26
    seed_43 = data["scratch_reverb_43"]         # epochs 1..22

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Соединяем две фазы сида 42 в одну линию, эпохи 1..26
    xs_42 = [e for e, _ in scratch_42] + [e for e, _ in reverb_42]
    ys_42 = [c for _, c in scratch_42] + [c for _, c in reverb_42]
    ax.plot(xs_42, ys_42, marker="o", linewidth=2, markersize=5,
            color="#1f77b4",
            label="Сид 42 · с нуля (1–20) + дообучение с ревербом (21–26)")

    # Отметка перехода к ревербу у сида 42
    ax.axvline(x=20.5, color="#1f77b4", linestyle=":", alpha=0.5)
    ax.text(20.6, 0.35, "включён\nреверб",
            color="#1f77b4", fontsize=9, va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.9))

    xs_43 = [e for e, _ in seed_43]
    ys_43 = [c for _, c in seed_43]
    ax.plot(xs_43, ys_43, marker="s", linewidth=2, markersize=5,
            color="#2ca02c",
            label="Сид 43 · с нуля с ревербом с 1-й эпохи")

    ax.axhline(y=0.0115, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.9,
               label="Финал: ансамбль + TTA = 0.0115")

    ax.set_yscale("log")
    ax.set_ylim(0.009, 1.2)
    ax.set_xlim(0, 28)
    ax.set_xlabel("Эпоха обучения")
    ax.set_ylabel("Валидационный CER (log-шкала)")
    ax.set_title("Динамика валидационного CER по эпохам для трёх обучающих циклов")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=3, framealpha=0.95, fontsize=9)
    fig.subplots_adjust(bottom=0.22)

    # Подписи ключевых точек — смещены так, чтобы не перекрывать линии
    ax.annotate("сид 42\nepoch 20: 0.0248", xy=(20, 0.0248), xytext=(10.5, 0.09),
                fontsize=9, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.6))
    ax.annotate("сид 42\nepoch 26: 0.0166\n(после реверба)", xy=(26, 0.0166), xytext=(22.3, 0.06),
                fontsize=9, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.6))
    ax.annotate("сид 43\nepoch 20: 0.0128", xy=(20, 0.0128), xytext=(14, 0.028),
                fontsize=9, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", alpha=0.6))

    path = FIG_DIR / "fig_training_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved -> {path}")


def plot_ablation_waterfall() -> None:
    # (label, delta in absolute CER, color)
    stages = [
        ("Лучший финал сессии 2\n(дообучение от базы)", 0.0394, "#9ca3af"),
        ("+ Обучение с нуля\nсо всеми аугментациями", 0.0223, "#2563eb"),
        ("+ Реверберация\n(дообучение 6 эпох)", 0.0152, "#0ea5e9"),
        ("+ Ансамбль двух сидов\n+ TTA + лучевой декодер", 0.0115, "#16a34a"),
    ]

    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colors = [s[2] for s in stages]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Каждый бар показывает абсолютное значение CER
    x = np.arange(len(stages))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.8, width=0.6)

    # Линии-соединители, показывающие «падение» от бара к бару
    for i in range(len(stages) - 1):
        ax.plot([x[i] + 0.3, x[i + 1] - 0.3],
                [values[i], values[i]],
                color="gray", linestyle=":", linewidth=1.2)
        delta = values[i + 1] - values[i]
        rel = delta / values[i] * 100
        ax.annotate(f"{delta:+.4f}\n({rel:+.0f}%)",
                    xy=((x[i] + x[i + 1]) / 2, (values[i] + values[i + 1]) / 2),
                    ha="center", va="center", fontsize=9,
                    color="#d62728", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="#d62728", pad=2, alpha=0.95))

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0009, f"{v:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Валидационный CER")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title("Водопад главных вкладов: от 0.0394 к 0.0115")
    ax.grid(axis="y", alpha=0.3)

    path = FIG_DIR / "fig_ablation_waterfall.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved -> {path}")


def plot_speaker_evolution() -> None:
    data = json.loads((FIG_DIR / "milestones_cer.json").read_text())

    milestones = [
        ("baseline", "База\n(dev 0.074)"),
        ("scratch_swa_tta", "С нуля\n+ SWA + TTA\n(dev 0.022)"),
        ("reverb_swa_tta", "+ реверб\n+ SWA + TTA\n(dev 0.015)"),
        ("ensemble_tta", "Ансамбль\n+ TTA\n(dev 0.012)"),
    ]

    speakers = ["spk_A", "spk_B", "spk_C", "spk_D", "spk_E", "spk_F",
                "spk_H", "spk_I", "spk_J", "spk_K"]
    in_domain = {"spk_A", "spk_B", "spk_C", "spk_D", "spk_E", "spk_F"}
    colors_ms = ["#9ca3af", "#2563eb", "#0ea5e9", "#16a34a"]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    x = np.arange(len(speakers))
    width = 0.2

    for i, (key, label) in enumerate(milestones):
        ys = [data[key]["per_speaker"].get(sp, 0.0) for sp in speakers]
        ax.bar(x + (i - 1.5) * width, ys, width,
               label=label, color=colors_ms[i], edgecolor="black", linewidth=0.5)

    # Разделитель in-domain и OOD
    ax.axvline(x=5.5, color="black", linestyle="--", alpha=0.4)
    max_y = max(max(data[key]["per_speaker"].values()) for key, _ in milestones)
    ax.set_ylim(0, max_y * 1.25)
    ax.text(2.5, max_y * 1.18, "In-domain (обучающие говорящие)",
            ha="center", fontsize=10, fontweight="bold", color="#2d6a4f")
    ax.text(7.5, max_y * 1.18, "OOD (новые говорящие)",
            ha="center", fontsize=10, fontweight="bold", color="#9d0208")

    ax.set_xticks(x)
    labels = [f"{sp}{'*' if sp not in in_domain else ''}" for sp in speakers]
    ax.set_xticklabels(labels)
    ax.set_xlabel("Говорящий (* — OOD)")
    ax.set_ylabel("CER по говорящему")
    ax.set_title("Эволюция CER по говорящим: в каждой итерации сильнее всего падает OOD")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.95, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    path = FIG_DIR / "fig_speaker_evolution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved -> {path}")


def main() -> None:
    plot_training_curves()
    plot_ablation_waterfall()
    plot_speaker_evolution()


if __name__ == "__main__":
    main()
