"""Tarea 3: DOE factorial completo para ajustar T0 del recocido simulado.

Opción A: 3 temperaturas x 3 tamaños x 10 réplicas = 90 corridas.
Genera tablas de resultados y figuras experimentales; el reporte se prepara por separado.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# Matrícula del estudiante usada como semilla para personalizar r.
EXPERIMENT_SEED = 2151976
# r se genera una sola vez y se reutiliza para calcular todos los niveles.
RNG_FOR_R = random.Random(EXPERIMENT_SEED)
R = RNG_FOR_R.random()
TEMPERATURE_RANGE = (1.0, 100.0)
TEMPERATURE_LABELS = ("Baja", "Media", "Alta")
SIZES = (5, 10, 15)
REPLICATES = 10
ITERATIONS = 800
BOOTSTRAP_SAMPLES = 20_000

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "graficas"


def temperature_levels() -> dict[str, float]:
    """Convierte una sola r en tres niveles del rango definido."""
    fractions = (0.3 * R, 0.3 * R + 0.3, 0.3 * R + 0.6)
    t_min, t_max = TEMPERATURE_RANGE
    return {
        label: t_min + fraction * (t_max - t_min)
        for label, fraction in zip(TEMPERATURE_LABELS, fractions)
    }


def tour_cost(points: list[tuple[float, float]], tour: list[int]) -> float:
    return sum(
        math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]])
        for i in range(len(tour))
    )


def anneal(points: list[tuple[float, float]], initial_temperature: float,
           rng: random.Random) -> float:
    """Recocido simulado con movimiento 2-opt y enfriamiento lineal."""
    n = len(points)
    current = list(range(n))
    rng.shuffle(current)
    current_cost = tour_cost(points, current)
    best_cost = current_cost
    pairs = [
        (i, k)
        for i in range(n - 1)
        for k in range(i + 1, n)
        if not (i == 0 and k == n - 1)
    ]

    for iteration in range(1, ITERATIONS + 1):
        i, k = rng.choice(pairs)
        candidate = current[:i] + current[i:k + 1][::-1] + current[k + 1:]
        candidate_cost = tour_cost(points, candidate)
        temperature = initial_temperature * (1.0 - iteration / ITERATIONS)
        delta = candidate_cost - current_cost
        if delta < 0 or (temperature > 0 and rng.random() < math.exp(-delta / temperature)):
            current, current_cost = candidate, candidate_cost
        if current_cost < best_cost:
            best_cost = current_cost
    return best_cost


def bootstrap_median(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mediana muestral e IC percentil bootstrap bilateral del 95%."""
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(np.median(values)), float(low), float(high)


def make_figures(data: pd.DataFrame, levels: dict[str, float]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    order = list(levels)
    colors = {5: "#3b82f6", 10: "#e07a5f", 15: "#2a9d8f"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    temp_stats = data.groupby("temperatura", observed=True)["costo"].agg(["mean", "std", "count"])
    temp_stats = temp_stats.reindex(order)
    temp_err = 1.96 * temp_stats["std"] / np.sqrt(temp_stats["count"])
    axes[0].errorbar(order, temp_stats["mean"], yerr=temp_err, marker="o", capsize=4,
                     linewidth=2, color="#4c78a8")
    axes[0].set(title="Efecto principal: temperatura", xlabel="Nivel de T₀",
                ylabel="Costo medio del tour (menor es mejor)")

    size_stats = data.groupby("tamano", observed=True)["costo"].agg(["mean", "std", "count"])
    size_order = list(SIZES)
    size_err = 1.96 * size_stats.loc[size_order, "std"] / np.sqrt(size_stats.loc[size_order, "count"])
    axes[1].errorbar([str(n) for n in size_order], size_stats.loc[size_order, "mean"],
                     yerr=size_err, marker="o", capsize=4, linewidth=2, color="#7b2cbf")
    axes[1].set(title="Efecto principal: tamaño", xlabel="Número de ciudades (n)",
                ylabel="Costo medio del tour")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Efectos principales (barras: IC aproximado del 95% de la media)")
    fig.tight_layout()
    main_effects = OUTPUT_DIR / "Tarea3_efectos_principales.png"
    fig.savefig(main_effects, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for n in SIZES:
        subset = data[data["tamano"] == n]
        means = subset.groupby("temperatura", observed=True)["costo"].mean().reindex(order)
        errors = subset.groupby("temperatura", observed=True)["costo"].agg(["std", "count"]).reindex(order)
        ci = 1.96 * errors["std"] / np.sqrt(errors["count"])
        ax.errorbar(order, means, yerr=ci, marker="o", capsize=4, linewidth=2,
                    color=colors[n], label=f"n = {n}")
    ax.set(title="Interacción temperatura × tamaño", xlabel="Nivel de temperatura inicial T₀",
           ylabel="Costo medio del tour")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Tamaño del TSP")
    fig.tight_layout()
    interaction = OUTPUT_DIR / "Tarea3_interaccion.png"
    fig.savefig(interaction, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return main_effects, interaction


def main() -> None:
    levels = temperature_levels()
    temperature_lookup = levels
    records = []
    # Una instancia fija e independiente para cada tamaño del problema.
    instances = {}
    for n in SIZES:
        instance_rng = random.Random(EXPERIMENT_SEED * 10 + n)
        instances[n] = [(instance_rng.uniform(0, 100), instance_rng.uniform(0, 100))
                        for _ in range(n)]

    for n in SIZES:
        for level_index, label in enumerate(TEMPERATURE_LABELS):
            for replicate in range(REPLICATES):
                run_seed = EXPERIMENT_SEED * 1_000 + n * 100 + level_index * REPLICATES + replicate
                run_rng = random.Random(run_seed)
                cost = anneal(instances[n], temperature_lookup[label], run_rng)
                records.append({
                    "tamano": n,
                    "temperatura": label,
                    "T0": temperature_lookup[label],
                    "replica": replicate + 1,
                    "semilla": run_seed,
                    "costo": cost,
                })

    data = pd.DataFrame.from_records(records)
    data["temperatura"] = pd.Categorical(data["temperatura"], categories=TEMPERATURE_LABELS, ordered=True)
    data["tamano"] = pd.Categorical(data["tamano"], categories=SIZES, ordered=True)

    model = ols("costo ~ C(temperatura) * C(tamano)", data=data).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_DIR / "Tarea3_datos_experimentales.csv", index=False)
    anova.to_csv(OUTPUT_DIR / "Tarea3_anova.csv")

    bootstrap_rng = np.random.default_rng(EXPERIMENT_SEED + 3)
    pointwise_rows = []
    bootstrap_results = {}
    report_all_points = (
        anova.loc["C(temperatura)", "PR(>F)"] < 0.05
        or anova.loc["C(temperatura):C(tamano)", "PR(>F)"] < 0.05
    )
    for n in SIZES:
        for label in TEMPERATURE_LABELS:
            point_costs = data[(data["tamano"] == n) & (data["temperatura"] == label)]["costo"].to_numpy()
            median, ci_low, ci_high = bootstrap_median(point_costs, bootstrap_rng)
            bootstrap_results[(n, label)] = (median, ci_low, ci_high)
            if report_all_points:
                pointwise_rows.append({
                    "tamano": n, "temperatura": label, "T0": levels[label],
                    "mediana_bootstrap": median, "ic95_inferior": ci_low,
                    "ic95_superior": ci_high, "replicas": REPLICATES,
                })
    pointwise_bootstrap = pd.DataFrame(pointwise_rows)
    pointwise_bootstrap.to_csv(OUTPUT_DIR / "Tarea3_bootstrap_puntos.csv", index=False)

    summary_rows = []
    means = data.groupby(["tamano", "temperatura"], observed=True)["costo"].mean()
    for n in SIZES:
        costs_by_level = means.loc[n]
        chosen_label = str(costs_by_level.idxmin())
        selected = data[(data["tamano"] == n) & (data["temperatura"] == chosen_label)]["costo"].to_numpy()
        median, ci_low, ci_high = bootstrap_results[(n, chosen_label)]
        summary_rows.append({
            "tamano": n,
            "temperatura_optima": chosen_label,
            "T0_optima": levels[chosen_label],
            "media_costo": float(np.mean(selected)),
            "mediana_bootstrap": median,
            "ic95_inferior": ci_low,
            "ic95_superior": ci_high,
            "replicas": REPLICATES,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "Tarea3_resumen_optimos.csv", index=False)

    make_figures(data, levels)

    print(f"Semilla={EXPERIMENT_SEED}; r (generado una sola vez)={R:.8f}")
    print("Niveles T0: " + ", ".join(f"{key}={value:.4f}" for key, value in levels.items()))
    print(f"Corridas ejecutadas: {len(data)}")
    print("ANOVA (p-values):")
    for source in ("C(temperatura)", "C(tamano)", "C(temperatura):C(tamano)"):
        print(f"  {source}: p={anova.loc[source, 'PR(>F)']:.6g}")
    print("Mejor nivel por tamaño (mediana bootstrap e IC 95% del costo):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Datos y gráficas: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
