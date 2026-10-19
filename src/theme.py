"""One Plotly template for every chart in the dashboard, notebook and report."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from src import config

TEMPLATE_NAME = "f1board"
HEADING_FONT = "'Titillium Web', 'Segoe UI', sans-serif"
BODY_FONT = "'IBM Plex Sans', 'Segoe UI', Arial, sans-serif"
GRID = "#232329"
AXIS = "#45454E"
TICK = "#B4B4BE"


def register_template() -> str:
    """Register the dark template and make it the Plotly default.

    Readability choices: 13px tick labels in a light grey, axis titles one
    step brighter, faint gridlines, and the legend above the plot where the
    eye lands first rather than under the axis title.
    """
    axis = dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickcolor=AXIS, ticks="outside", ticklen=4,
                title=dict(standoff=12, font=dict(size=13.5, color="#D2D2D9")),
                tickfont=dict(size=13, color=TICK), automargin=True)
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY_FONT, color=config.TEXT, size=13.5),
        title=dict(font=dict(family=HEADING_FONT, size=18, color="#ECECEF"), x=0.0, xanchor="left",
                   y=0.98, yanchor="top", pad=dict(l=2)),
        colorway=config.COLORWAY,
        margin=dict(l=60, r=24, t=88, b=56),
        xaxis=axis,
        yaxis=axis,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, xanchor="left", bgcolor="rgba(0,0,0,0)",
                    font=dict(size=13, color="#D2D2D9"), title=dict(text="")),
        hoverlabel=dict(bgcolor="#202026", bordercolor=AXIS, font=dict(family=BODY_FONT, color="#ECECEF", size=13.5)),
        colorscale=dict(sequential=config.SEQUENTIAL, diverging=config.DIVERGING),
        coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=10, tickfont=dict(color=TICK, size=12))),
        bargap=0.3,
        hovermode="closest",
    )
    template.data.scatter = [go.Scatter(marker=dict(size=8, line=dict(width=0)), line=dict(width=2.25))]
    template.data.bar = [go.Bar(marker=dict(line=dict(width=0), cornerradius=3))]
    template.data.histogram = [go.Histogram(marker=dict(line=dict(width=0)))]
    template.data.box = [go.Box(marker=dict(size=5), line=dict(width=1.5))]
    pio.templates[TEMPLATE_NAME] = template
    pio.templates.default = TEMPLATE_NAME
    return TEMPLATE_NAME


def finish(fig: go.Figure, title: str, height: int = 440, legend: bool = True) -> go.Figure:
    """Apply the shared finishing touches: sentence-case title, height, template."""
    title = title[:1].upper() + title[1:] if title else title
    if (fig.layout.meta or {}).get("direct_labels"):    # lines are labelled at their ends instead
        legend = False
    top = 88 if legend else 60
    fig.update_layout(template=TEMPLATE_NAME, title_text=title, height=height, showlegend=legend,
                      margin=dict(t=top))
    return fig


def label_line_ends(fig: go.Figure, frame: pd.DataFrame, x: str, y: str, group: str,
                    names: dict[str, str] | None = None, colours: dict[str, str] | None = None) -> go.Figure:
    """Write each series' name at the end of its line, instead of relying on a legend.

    Labels that would collide are nudged apart vertically so they stay legible
    when several lines finish close together.
    """
    data = frame.dropna(subset=[x, y]).sort_values(x)
    if data.empty:
        return fig
    ends = data.groupby(group).tail(1)[[group, x, y]].sort_values(y).reset_index(drop=True)
    span = float(data[y].max() - data[y].min()) or 1.0
    gap = span * 0.045
    positions = ends[y].to_numpy(dtype=float).copy()
    for i in range(1, len(positions)):                      # push labels up until none overlap
        positions[i] = max(positions[i], positions[i - 1] + gap)
    overshoot = positions[-1] - (data[y].max() + gap)
    if overshoot > 0:                                       # keep the stack inside the plot
        positions -= overshoot
        for i in range(len(positions) - 2, -1, -1):
            positions[i] = min(positions[i], positions[i + 1] - gap)
    for (_, row), pos in zip(ends.iterrows(), positions):
        key = row[group]
        fig.add_annotation(x=row[x], y=pos, text=(names or {}).get(key, str(key)), showarrow=False,
                           xanchor="left", xshift=8, font=dict(size=12.5, color=(colours or {}).get(key, TICK)))
    lo, hi = float(data[x].min()), float(data[x].max())
    fig.update_xaxes(range=[lo - (hi - lo) * 0.02, hi + (hi - lo) * 0.02])
    fig.update_layout(showlegend=False, margin=dict(r=130), meta={"direct_labels": True}, hovermode="x unified")
    return fig


def team_color(lineage: str) -> str:
    """Fixed colour for a team lineage (grey for anything unknown)."""
    return config.TEAM_COLOR.get(lineage, config.MUTED)


def team_label(lineage: str) -> str:
    """Display name for a team lineage."""
    return config.LINEAGE_LABEL.get(lineage, str(lineage))


def team_short(label: str) -> str:
    """Short team name for chart labels: 'Alpine / Renault' -> 'Alpine'."""
    return str(label).split(" /")[0]


register_template()

