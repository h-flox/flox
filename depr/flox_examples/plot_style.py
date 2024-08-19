import seaborn as sns

HEX_COLORS = [
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#726762",
    "#54A24B",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]

# COLORS = mpl.colors.cycler("color", HEX_COLORS)

palette = sns.color_palette(HEX_COLORS)
palette = sns.set_palette(palette)

sns.set_style(
    "ticks",
    rc={
        "axes.grid": True,
        "grid.linestyle": "solid",
        "grid.color": "#D8D8D8",
    },
)
