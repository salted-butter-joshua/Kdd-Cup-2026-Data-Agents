"""Generate PNG figures for PHASE_1 README / architecture docs.

Run from PHASE_1/:
  python assets/gen_readme_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parent


def _configure_fonts() -> None:
    from matplotlib import font_manager

    candidates = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": chosen,
            "axes.unicode_minus": False,
            "figure.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
        }
    )
    print("font:", chosen)


_configure_fonts()


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


def score_evolution() -> None:
    labels = [
        "Early\nparallel",
        "§13\ngates",
        "Cache\nmiss",
        "Task\nBudget",
        "P0/P1\npeak",
        "Over-\nstrict",
        "Recovered",
        "Qwen\nfirst",
        "Qwen\nbest",
    ]
    values = [0.59, 0.82, 0.63, 0.83, 0.85, 0.78, 0.84, 0.79, 0.82]
    colors = [
        "#94a3b8",
        "#38bdf8",
        "#f87171",
        "#34d399",
        "#2563eb",
        "#f97316",
        "#22c55e",
        "#a78bfa",
        "#7c3aed",
    ]
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, width=0.72, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0.5, 0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean_score (50 tasks)", fontsize=11)
    ax.set_title("Public demo mean_score by architecture stage", fontsize=13, pad=12, fontweight="medium")
    ax.axhline(0.8538, color="#2563eb", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(8.2, 0.858, "MiniMax peak 0.8538", color="#2563eb", fontsize=8, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    _save(fig, "score_evolution.png")

    # Chinese twin
    labels_zh = [
        "早期\n并行",
        "§13\n门禁",
        "缺抽\n缓存",
        "Task\nBudget",
        "P0/P1\n峰值",
        "过严\n门禁",
        "恢复",
        "Qwen\n首通",
        "Qwen\n最佳",
    ]
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    bars = ax.bar(x, values, color=colors, width=0.72, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0.5, 0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_zh, fontsize=9)
    ax.set_ylabel("mean_score（50题）", fontsize=11)
    ax.set_title("公开 demo 均分演进（按架构阶段）", fontsize=13, pad=12, fontweight="medium")
    ax.axhline(0.8538, color="#2563eb", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(8.2, 0.858, "MiniMax 峰值 0.8538", color="#2563eb", fontsize=8, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    _save(fig, "score_evolution_zh.png")


def model_comparison() -> None:
    labels = ["MiniMax\nbest", "MiniMax\nrecovered", "Qwen\nfirst", "Qwen\nbest"]
    values = [0.8538, 0.8364, 0.7940, 0.8193]
    colors = ["#2563eb", "#60a5fa", "#a78bfa", "#7c3aed"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="white")
    ax.set_ylim(0.7, 0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("mean_score (50 tasks)", fontsize=11)
    ax.set_title("Same agent stack · MiniMax vs Qwen3.6", fontsize=13, pad=12, fontweight="medium")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    legend = [
        mpatches.Patch(color="#2563eb", label="MiniMax-M2.5"),
        mpatches.Patch(color="#7c3aed", label="Qwen3.6-35B-A3B-FP8 (local)"),
    ]
    ax.legend(handles=legend, frameon=False, loc="upper right", fontsize=9)
    _save(fig, "model_comparison.png")

    labels_zh = ["MiniMax\n最佳", "MiniMax\n恢复", "Qwen\n首通", "Qwen\n最佳"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="white")
    ax.set_ylim(0.7, 0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_zh, fontsize=10)
    ax.set_ylabel("mean_score（50题）", fontsize=11)
    ax.set_title("同套 Agent · MiniMax vs Qwen3.6", fontsize=13, pad=12, fontweight="medium")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    legend_zh = [
        mpatches.Patch(color="#2563eb", label="MiniMax-M2.5"),
        mpatches.Patch(color="#7c3aed", label="Qwen3.6-35B（本地）"),
    ]
    ax.legend(handles=legend_zh, frameon=False, loc="upper right", fontsize=9)
    _save(fig, "model_comparison_zh.png")


def impact_map() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Architecture changes → measured score impact", fontsize=13, pad=8, fontweight="medium")

    left = FancyBboxPatch(
        (0.4, 0.6), 5.3, 4.6, boxstyle="round,pad=0.04,rounding_size=0.15",
        facecolor="#ecfdf5", edgecolor="#059669", linewidth=1.6,
    )
    right = FancyBboxPatch(
        (6.3, 0.6), 5.3, 4.6, boxstyle="round,pad=0.04,rounding_size=0.15",
        facecolor="#fff7ed", edgecolor="#ea580c", linewidth=1.6,
    )
    ax.add_patch(left)
    ax.add_patch(right)
    ax.text(3.05, 4.85, "Raises mean", ha="center", fontsize=12, color="#047857", fontweight="medium")
    ax.text(8.95, 4.85, "Often lowers mean", ha="center", fontsize=12, color="#c2410c", fontweight="medium")

    gains = [
        ("L3 submit gates", "~0.59 → 0.82"),
        ("TaskBudget + Episode", "missing↓ · ~0.63 → 0.83"),
        ("P0/P1 answer contract", "peak → 0.85"),
        ("Local HTTP client fix", "Qwen runnable → 0.82"),
    ]
    risks = [
        ("Over-strict gates", "→ 0.78"),
        ("LLM 502 / disconnects", "burn steps / empty CSV"),
        ("Doc tasks / fake zero", "352 · 396 · 418"),
    ]
    for i, (t, s) in enumerate(gains):
        y = 4.2 - i * 0.85
        box = FancyBboxPatch(
            (0.75, y - 0.35), 4.6, 0.7, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#ffffff", edgecolor="#6ee7b7", linewidth=1.1,
        )
        ax.add_patch(box)
        ax.text(1.0, y + 0.05, t, fontsize=10, color="#064e3b", va="center")
        ax.text(5.1, y + 0.05, s, fontsize=9, color="#047857", ha="right", va="center")
    for i, (t, s) in enumerate(risks):
        y = 3.9 - i * 1.0
        box = FancyBboxPatch(
            (6.65, y - 0.35), 4.6, 0.7, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#ffffff", edgecolor="#fdba74", linewidth=1.1,
        )
        ax.add_patch(box)
        ax.text(6.9, y + 0.05, t, fontsize=10, color="#7c2d12", va="center")
        ax.text(11.0, y + 0.05, s, fontsize=9, color="#c2410c", ha="right", va="center")
    _save(fig, "impact_map.png")

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("架构改动 → 实测分数影响", fontsize=13, pad=8, fontweight="medium")
    ax.add_patch(
        FancyBboxPatch(
            (0.4, 0.6), 5.3, 4.6, boxstyle="round,pad=0.04,rounding_size=0.15",
            facecolor="#ecfdf5", edgecolor="#059669", linewidth=1.6,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (6.3, 0.6), 5.3, 4.6, boxstyle="round,pad=0.04,rounding_size=0.15",
            facecolor="#fff7ed", edgecolor="#ea580c", linewidth=1.6,
        )
    )
    ax.text(3.05, 4.85, "拉升均分", ha="center", fontsize=12, color="#047857", fontweight="medium")
    ax.text(8.95, 4.85, "易拉低均分", ha="center", fontsize=12, color="#c2410c", fontweight="medium")
    gains_zh = [
        ("L3 提交门禁", "~0.59 → 0.82"),
        ("TaskBudget + Episode", "missing↓ · ~0.63 → 0.83"),
        ("P0/P1 答法契约", "峰值 → 0.85"),
        ("本地 HTTP 客户端修复", "Qwen 可跑 → 0.82"),
    ]
    risks_zh = [
        ("过严门禁", "→ 0.78"),
        ("LLM 502 / 断连", "吃步数 / 空 CSV"),
        ("文档题 / 假 0", "352 · 396 · 418"),
    ]
    for i, (t, s) in enumerate(gains_zh):
        y = 4.2 - i * 0.85
        ax.add_patch(
            FancyBboxPatch(
                (0.75, y - 0.35), 4.6, 0.7, boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="#ffffff", edgecolor="#6ee7b7", linewidth=1.1,
            )
        )
        ax.text(1.0, y + 0.05, t, fontsize=10, color="#064e3b", va="center")
        ax.text(5.1, y + 0.05, s, fontsize=9, color="#047857", ha="right", va="center")
    for i, (t, s) in enumerate(risks_zh):
        y = 3.9 - i * 1.0
        ax.add_patch(
            FancyBboxPatch(
                (6.65, y - 0.35), 4.6, 0.7, boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="#ffffff", edgecolor="#fdba74", linewidth=1.1,
            )
        )
        ax.text(6.9, y + 0.05, t, fontsize=10, color="#7c2d12", va="center")
        ax.text(11.0, y + 0.05, s, fontsize=9, color="#c2410c", ha="right", va="center")
    _save(fig, "impact_map_zh.png")


def architecture_layers() -> None:
    """Layered stack with side legend — for README and 架构设计.md."""
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(
        7.0, 9.55, "Data Agent layered architecture",
        ha="center", fontsize=15, fontweight="medium", color="#0f172a",
    )
    ax.text(
        7.0, 9.15, "Warehouse → Budget → ReAct → Gates → Submit → Eval",
        ha="center", fontsize=10, color="#64748b",
    )

    layers = [
        # y, h, color, edge, code, title, detail
        (7.55, 1.05, "#e0f2fe", "#0284c7", "L0", "Warehouse", "CSV/JSON/SQLite + doc extract → DuckDB task DB"),
        (6.35, 1.05, "#ecfeff", "#0891b2", "L-1", "Budget & Episode", "Easy 3′ / Med 7′ / Hard 10′ · Plan / Ban / PROGRESS"),
        (5.15, 1.05, "#f5f3ff", "#7c3aed", "L2", "ReAct loop", "JSON actions · list_tables / run_sql / search_docs / answer"),
        (3.95, 1.05, "#fef3c7", "#d97706", "L3", "Submit gates", "empty · ties · grain · relation · shape · fake-zero (fail-open)"),
        (2.75, 1.05, "#ffedd5", "#ea580c", "L4/L5", "Repair & submit", "hint escalation · vote / fallback → prediction.csv"),
        (1.55, 1.05, "#dcfce7", "#16a34a", "L6", "Eval", "column-signature score · λ=0.1 extra-col penalty"),
    ]

    for y, h, face, edge, code, title, detail in layers:
        box = FancyBboxPatch(
            (0.45, y), 8.7, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=face, edgecolor=edge, linewidth=1.8,
        )
        ax.add_patch(box)
        badge = FancyBboxPatch(
            (0.65, y + 0.28), 1.15, 0.5, boxstyle="round,pad=0.01,rounding_size=0.08",
            facecolor=edge, edgecolor=edge,
        )
        ax.add_patch(badge)
        ax.text(1.22, y + 0.53, code, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(2.1, y + 0.68, title, va="center", fontsize=12, color="#0f172a", fontweight="medium")
        ax.text(2.1, y + 0.32, detail, va="center", fontsize=9, color="#334155")

    # vertical flow arrows between layers
    for y in [7.55, 6.35, 5.15, 3.95, 2.75]:
        ax.annotate(
            "",
            xy=(4.8, y - 0.02),
            xytext=(4.8, y - 0.12),
            arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.2),
        )

    # side legend panel
    legend_box = FancyBboxPatch(
        (9.5, 1.55), 4.2, 7.05, boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.4,
    )
    ax.add_patch(legend_box)
    ax.text(11.6, 8.25, "Legend", ha="center", fontsize=12, color="#0f172a", fontweight="medium")

    legend_items = [
        ("#0284c7", "L0 Data plane", "Build warehouse; versioned extract cache"),
        ("#0891b2", "L-1 Control plane", "Wall-clock tiers; episode blackboard"),
        ("#7c3aed", "L2 Reasoning", "LLM proposes SQL/tools only"),
        ("#d97706", "L3 Verification", "Deterministic reject + actionable hint"),
        ("#ea580c", "L4/L5 Delivery", "Repair loop; never empty CSV"),
        ("#16a34a", "L6 Scoring", "Official column-signature metric"),
    ]
    for i, (c, title, desc) in enumerate(legend_items):
        y = 7.55 - i * 0.95
        ax.add_patch(
            FancyBboxPatch(
                (9.75, y), 0.45, 0.45, boxstyle="round,pad=0.01,rounding_size=0.06",
                facecolor=c, edgecolor=c,
            )
        )
        ax.text(10.4, y + 0.32, title, fontsize=9.5, color="#0f172a", va="center", fontweight="medium")
        ax.text(10.4, y + 0.05, desc, fontsize=8, color="#64748b", va="center")

    ax.text(
        7.0, 0.55,
        "Principles: mechanism-level · no task_id special cases · fail-open when undecidable · no empty CSV",
        ha="center", fontsize=9, color="#64748b",
    )
    _save(fig, "architecture_layers.png")


def architecture_flow_cn() -> None:
    """Chinese-labeled companion for 架构设计.md."""
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(7.0, 9.55, "Data Agent 分层架构", ha="center", fontsize=15, fontweight="medium", color="#0f172a")
    ax.text(
        7.0, 9.15, "建仓 → 分档预算 → ReAct 探查 → 确定性门禁 → 交卷 → 打分",
        ha="center", fontsize=10, color="#64748b",
    )

    layers = [
        (7.55, 1.05, "#e0f2fe", "#0284c7", "L0", "仓库层 Warehouse", "CSV/JSON/SQLite + 文档抽取 → 每题 DuckDB"),
        (6.35, 1.05, "#ecfeff", "#0891b2", "L-1", "控制层 Budget / Episode", "Easy 3′ / Med 7′ / Hard 10′ · Plan / Ban / PROGRESS"),
        (5.15, 1.05, "#f5f3ff", "#7c3aed", "L2", "推理层 ReAct", "JSON 动作 · list_tables / run_sql / search_docs / answer"),
        (3.95, 1.05, "#fef3c7", "#d97706", "L3", "验证层 Submit Gates", "空表 · 并列 · 粒度 · 关系 · 形状 · 假0（无法判定则放行）"),
        (2.75, 1.05, "#ffedd5", "#ea580c", "L4/L5", "修复与交卷", "hint 升级 · 投票 / fallback → prediction.csv"),
        (1.55, 1.05, "#dcfce7", "#16a34a", "L6", "评测层 Eval", "列签名打分 · λ=0.1 多余列惩罚"),
    ]
    for y, h, face, edge, code, title, detail in layers:
        ax.add_patch(
            FancyBboxPatch(
                (0.45, y), 8.7, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=face, edgecolor=edge, linewidth=1.8,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.65, y + 0.28), 1.15, 0.5, boxstyle="round,pad=0.01,rounding_size=0.08",
                facecolor=edge, edgecolor=edge,
            )
        )
        ax.text(1.22, y + 0.53, code, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(2.1, y + 0.68, title, va="center", fontsize=12, color="#0f172a", fontweight="medium")
        ax.text(2.1, y + 0.32, detail, va="center", fontsize=9, color="#334155")

    ax.add_patch(
        FancyBboxPatch(
            (9.5, 1.55), 4.2, 7.05, boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.4,
        )
    )
    ax.text(11.6, 8.25, "图例", ha="center", fontsize=12, color="#0f172a", fontweight="medium")
    legend_items = [
        ("#0284c7", "L0 数据面", "建仓；版本化抽取缓存"),
        ("#0891b2", "L-1 控制面", "分档墙钟；Episode 黑板"),
        ("#7c3aed", "L2 推理面", "模型只提议工具/SQL"),
        ("#d97706", "L3 验证面", "代码拒答 + 可执行 hint"),
        ("#ea580c", "L4/L5 交付面", "修复回路；禁止空 CSV"),
        ("#16a34a", "L6 评测面", "官方 column-signature"),
    ]
    for i, (c, title, desc) in enumerate(legend_items):
        y = 7.55 - i * 0.95
        ax.add_patch(
            FancyBboxPatch(
                (9.75, y), 0.45, 0.45, boxstyle="round,pad=0.01,rounding_size=0.06",
                facecolor=c, edgecolor=c,
            )
        )
        ax.text(10.4, y + 0.32, title, fontsize=9.5, color="#0f172a", va="center", fontweight="medium")
        ax.text(10.4, y + 0.05, desc, fontsize=8, color="#64748b", va="center")

    ax.text(
        7.0, 0.55,
        "原则：机制级 · 零 task_id 特判 · 无法判定则放行 · 不交空 CSV",
        ha="center", fontsize=9, color="#64748b",
    )
    _save(fig, "architecture_layers_zh.png")


if __name__ == "__main__":
    score_evolution()
    model_comparison()
    impact_map()
    architecture_layers()
    architecture_flow_cn()
    print("done")
