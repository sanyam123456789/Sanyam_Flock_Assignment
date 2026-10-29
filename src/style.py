"""Shared matplotlib style (reference palette, light mode, static PNGs)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GREY, LIGHT, INK, INK2 = "#8a8986", "#d9d8d4", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.6, "figure.dpi": 120,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titlelocation": "left",
    "legend.frameon": False,
})
