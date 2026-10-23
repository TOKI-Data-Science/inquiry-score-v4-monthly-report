"""Inquiry Score v4 monthly performance report (single HTML file, Plotly via CDN)."""
from __future__ import annotations

import html
from datetime import datetime
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import roc_auc_score

from src.module.helper import logging_timer
from src.module.settings import settings

OOT_AUC = {
    'overall': 0.666,
    'product_type': {'credit': 0.660, 'leasing': 0.681, 'bnpl': 0.663},
    'model_type': {
        'ndsh_credit': 0.671,
        'no_ndsh_credit': 0.654,
        'ndsh_lease': 0.705,
        'no_ndsh_lease': 0.680,
    },
}
# OOT-valid grade profile from model development: bin -> (population share, event rate)
OOT_GRADE = {
    '1': (0.023, 0.036), '2': (0.045, 0.065), '3': (0.051, 0.071), '4': (0.097, 0.118), '5': (0.146, 0.157),
    '6': (0.149, 0.175), '7': (0.169, 0.223), '8': (0.163, 0.270), '9': (0.086, 0.327), '10': (0.072, 0.441),
}
GRADE_SHIFT_ALERT_PP = 5.0
WARNING_DROP = 0.05
CRITICAL_DROP = 0.10
REQUIRED_COLUMNS = {'id', 'product_type', 'model_type', 'v4_score', 'v4_bin', 'base_month', 'event'}

# Harbor Blue color book, dark mode (HTML only): primary becomes Tint 400 so it stays readable.
ACCENT = '#7FA8EC'  # Tint 400
TINT_200, TINT_50, DEEP = '#CFE0FA', '#F2F7FE', '#0D2A4A'
PALETTE = ['#2E6BD6', '#7FA8EC', '#2BA3A0', '#E59AB4', '#9C8FE0', '#8A94A6']  # Chart 1-6, assigned in order
BG, SURFACE, SURFACE2, LINE, TEXT, MUTED = '#10151F', '#18202D', '#1F2938', '#2A3445', '#E8EEF7', '#9BA8BD'
HOVER_BG = SURFACE2
GOOD, WARN, BAD = '#3A9E6E', '#E0A33A', '#D9604C'
GOOD_BG, WARN_BG, BAD_BG = '#E2F2E9', '#FBF0DA', '#FAE4E0'
TICK = MUTED
# Segment colors from the color book: bnpl = deepest visible blue, credit = amber, leasing = soft blue.
SEGMENT_COLORS = {
    'bnpl': '#2E6BD6', 'credit': '#E0A33A', 'leasing': '#7FA8EC',
    # models: clearly different hues, still from the book (amber/rose = credit side, blue/teal = lease side)
    'ndsh_credit': '#E0A33A', 'no_ndsh_credit': '#E59AB4',
    'ndsh_lease': '#7FA8EC', 'no_ndsh_lease': '#2BA3A0',
}


def segment_colors(names, fallback=PALETTE):
    """Fixed color per segment name; unknown names take unused chart colors in order."""
    spare = cycle([c for c in fallback if c not in SEGMENT_COLORS.values()] or fallback)
    return [SEGMENT_COLORS.get(str(n)) or next(spare) for n in names]
GRADE_INVERSION_THRESHOLD = 0.01
FONT = 'Segoe UI, Arial, sans-serif'

def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.str.lower()
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f'Dataset is missing columns: {", ".join(sorted(missing))}')
    frame = frame.dropna(subset=['v4_score', 'event', 'base_month'])
    frame['v4_score'] = pd.to_numeric(frame['v4_score'], errors='coerce')
    frame['event'] = pd.to_numeric(frame['event'], errors='raise').astype(int)
    if not frame['event'].isin([0, 1]).all():
        raise ValueError('event must contain only 0 and 1')
    frame['base_month'] = frame['base_month'].astype(int)
    frame = frame[frame['base_month'] >= settings.report_start_month]
    for col in ('product_type', 'model_type', 'v4_bin'):
        frame[col] = frame[col].fillna('n/a').astype(str).str.strip()
  
    num_bin = pd.to_numeric(frame['v4_bin'], errors='coerce')
    if num_bin.notna().all() and (num_bin % 1 == 0).all():
        frame['v4_bin'] = num_bin.astype(int).astype(str)
    return frame.reset_index(drop=True)


def auc_or_nan(group: pd.DataFrame) -> float:
    valid = group['v4_score'].notna() & group['event'].notna()
    if group.loc[valid, 'event'].nunique() < 2:
        return np.nan
    auc = roc_auc_score(group.loc[valid, 'event'], group.loc[valid, 'v4_score'])
    return 1 - auc if settings.score_higher_is_better else auc


def monthly_metrics(frame: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    keys = ['base_month'] if by is None else [by, 'base_month']
    grouped = frame.groupby(keys, dropna=False)
    outcomes = grouped['event'].agg(population='size', events='sum', event_rate='mean').reset_index()
    auc = grouped[['v4_score', 'event']].apply(auc_or_nan).rename('auc').reset_index()
    return outcomes.merge(auc, on=keys).sort_values(keys).reset_index(drop=True)


def grade_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    grade = frame.groupby('v4_bin', dropna=False)['event'].agg(population='size', events='sum', event_rate='mean').reset_index()
    grade['share'] = grade['population'] / grade['population'].sum()
    grade['order'] = pd.to_numeric(grade['v4_bin'].str.extract(r'(-?\d+\.?\d*)')[0], errors='coerce')
    return grade.sort_values(['order', 'v4_bin'], na_position='last').drop(columns='order').reset_index(drop=True)


def latest_auc(monthly: pd.DataFrame) -> tuple[float, str]:
    valid = monthly.dropna(subset=['auc'])
    if valid.empty:
        return np.nan, 'n/a'
    row = valid.iloc[-1]
    return float(row['auc']), str(int(row['base_month']))


def assess(latest: float, oot: float) -> tuple[str, str, str]:
    if np.isnan(oot):
        return 'neutral', 'No benchmark', 'No OOT-valid AUC is defined for this segment; monitor trend only.'
    if np.isnan(latest):
        return 'neutral', 'Insufficient signal', 'Latest month has a single event class or no rows. Wait for more outcomes.'
    drop = oot - latest
    if drop >= CRITICAL_DROP:
        return 'critical', 'Redevelop model', f'AUC is {drop:.3f} below OOT-valid (threshold {CRITICAL_DROP:.2f}). Begin model redevelopment.'
    if drop >= WARNING_DROP:
        return 'warning', 'Retrain / wait one month', f'AUC is {drop:.3f} below OOT-valid (threshold {WARNING_DROP:.2f}). Retrain now or confirm persistence next month.'
    return 'healthy', 'Continue monitoring', f'AUC is within {WARNING_DROP:.2f} of OOT-valid ({-drop:+.3f}). No model change is required.'


def segment_status(monthly: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for name, grp in monthly.groupby(by):
        oot = OOT_AUC[by].get(str(name), np.nan)
        latest, month = latest_auc(grp)
        level, title, _ = assess(latest, oot)
        rows.append(
            {
                by: name,
                'population': int(grp['population'].sum()),
                'event_rate': grp['events'].sum() / grp['population'].sum(),
                'oot_auc': oot,
                'avg_auc': grp['auc'].mean(),
                'latest_month': month,
                'latest_auc': latest,
                'delta': latest - oot,
                'status': (level, title),
            }
        )
    return pd.DataFrame(rows)

AXIS_TITLE = dict(size=13, color=TICK)


def base_layout(fig: go.Figure, height: int = 320) -> None:
    """Shared chart style. No title inside the figure: the section heading is the title."""
    fig.update_layout(
        title=None,
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=TICK, size=13),
        margin=dict(l=52, r=52, t=36, b=44),
        hoverlabel=dict(bgcolor=HOVER_BG, font_color=TEXT, bordercolor=LINE, font_size=13),
        legend=dict(orientation='h', y=1.02, yanchor='bottom', x=1, xanchor='right', bgcolor='rgba(0,0,0,0)', font=dict(size=11, color=TICK), itemwidth=30, tracegroupgap=0),
        xaxis=dict(showgrid=False, zeroline=False, fixedrange=True, showline=True, linecolor=LINE, tickfont=dict(size=12, color=TICK)),
        yaxis=dict(showgrid=False, zeroline=False, fixedrange=True, tickfont=dict(size=12, color=TICK)),
    )


def trajectory_chart(monthly: pd.DataFrame, by: str | None = None) -> go.Figure:
    months = sorted(monthly['base_month'].astype(str).unique())
    categories = ['OOT-valid', *months]
    if by is None:
        series = [('Overall', monthly, OOT_AUC['overall'])]
    else:
        series = [(str(name), grp, OOT_AUC[by].get(str(name), np.nan)) for name, grp in monthly.groupby(by)]

    fig = go.Figure()
    all_values = []
    colors = [PALETTE[0]] if by is None else segment_colors([n for n, _, _ in series])
    for color, (name, grp, oot) in zip(colors, series):
        by_month = grp.set_index(grp['base_month'].astype(str))['auc'].reindex(months)
        y = [oot, *by_month.tolist()]
        all_values.extend(v for v in y if pd.notna(v))
        fig.add_scatter(
            x=categories,
            y=y,
            name=name,
            mode='lines+markers',
            line=dict(color=color, width=2.5, shape='spline', smoothing=0.6),
            marker=dict(size=8, color=color, symbol=['diamond', *['circle'] * len(months)]),
            hovertemplate='%{x}<br>AUC %{y:.3f}<extra>' + html.escape(name) + '</extra>',
        )
    fig.add_vrect(x0=-0.5, x1=0.5, fillcolor=ACCENT, opacity=0.08, line_width=0)
    lo, hi = (min(all_values), max(all_values)) if all_values else (0.5, 0.8)
    base_layout(fig)
    fig.update_layout(showlegend=by is not None)
    fig.update_xaxes(type='category', categoryorder='array', categoryarray=categories)
    fig.update_yaxes(range=[lo - 0.03, hi + 0.04], tickformat='.2f')
    return fig


def grade_chart(grade: pd.DataFrame, color: str = PALETTE[0]) -> go.Figure:
    bins = grade['v4_bin'].tolist()
    fig = go.Figure()
    fig.add_bar(
        x=bins, y=grade['share'], name='Population share', marker_color=TINT_200, marker_line_width=0, opacity=0.35,
        customdata=grade['population'],
        hovertemplate='Bin %{x}<br>Share %{y:.1%} (%{customdata:,} population)<extra></extra>',
    )
    fig.add_scatter(
        x=bins, y=grade['event_rate'], name='Event rate', yaxis='y2', mode='lines+markers',
        line=dict(color=color, width=2.5, shape='spline', smoothing=0.6), marker=dict(size=7, color=color),
        hovertemplate='Bin %{x}<br>Event rate %{y:.2%}<extra></extra>',
    )
    rates = grade['event_rate'].tolist()
    inverted_bins = [bins[i] for i in range(len(rates) - 1) if rates[i] > rates[i + 1] + GRADE_INVERSION_THRESHOLD]
    if inverted_bins:
        fig.add_scatter(
            x=inverted_bins, y=[rates[bins.index(b)] for b in inverted_bins], yaxis='y2', name='Inversion',
            mode='markers', marker=dict(size=18, color='rgba(0,0,0,0)', line=dict(color=BAD, width=2)),
            hovertemplate='Bin %{x}: rate inversion (drops >%{customdata:.0%} into next bin)<extra></extra>',
            customdata=[GRADE_INVERSION_THRESHOLD] * len(inverted_bins), showlegend=False,
        )
    base_layout(fig)
    max_share = float(grade['share'].max()) if len(grade) else 1.0
    max_rate = float(grade['event_rate'].max()) if len(grade) else 1.0
    fig.update_layout(
        xaxis=dict(type='category', categoryorder='array', categoryarray=bins),
        yaxis=dict(title=dict(text='Population share', font=AXIS_TITLE), tickformat='.0%', range=[0, max_share * 1.15]),
        yaxis2=dict(title=dict(text='Event rate', font=AXIS_TITLE), overlaying='y', side='right', tickformat='.0%', range=[0, max_rate * 1.15], showgrid=False, fixedrange=True, tickfont=dict(size=12, color=TICK)),
        bargap=0.5,
    )
    return fig


def grade_compare_frame(grade: pd.DataFrame) -> pd.DataFrame:
    """OOT-valid vs current share and event rate per bin (bins ordered numerically where possible)."""
    cur = grade.set_index('v4_bin')
    bins = sorted(set(OOT_GRADE) | set(cur.index), key=lambda b: (pd.to_numeric(b, errors='coerce'), b))
    out = pd.DataFrame({'bin': bins})
    out['oot_share'] = [OOT_GRADE.get(b, (np.nan, np.nan))[0] for b in bins]
    out['oot_rate'] = [OOT_GRADE.get(b, (np.nan, np.nan))[1] for b in bins]
    out['cur_share'] = [cur['share'].get(b, np.nan) for b in bins]
    out['cur_rate'] = [cur['event_rate'].get(b, np.nan) for b in bins]
    return out


def _shift_labels(fig: go.Figure, bins, cur, oot) -> None:
    """Current-minus-OOT label (percentage points) above each bin; red past the alert threshold."""
    for b, c, o in zip(bins, cur, oot):
        if pd.isna(c) or pd.isna(o):
            continue
        pp = (c - o) * 100
        fig.add_annotation(
            x=b, y=max(c, o), yref='y', text=f'{pp:+.1f}'.replace('-', '−'), showarrow=False, yshift=12,
            font=dict(size=11, color=BAD if abs(pp) > GRADE_SHIFT_ALERT_PP else MUTED),
        )


def grade_share_compare_chart(cmp: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=cmp['bin'], y=cmp['oot_share'], name='OOT', marker_color=PALETTE[5], marker_line_width=0, opacity=0.6,
                hovertemplate='Bin %{x}<br>OOT share %{y:.1%}<extra></extra>')
    fig.add_bar(x=cmp['bin'], y=cmp['cur_share'], name='Current', marker_color=PALETTE[1], marker_line_width=0,
                hovertemplate='Bin %{x}<br>Current share %{y:.1%}<extra></extra>')
    base_layout(fig, height=360)
    fig.update_layout(
        barmode='group', bargap=0.25, bargroupgap=0.05,
        xaxis=dict(type='category', categoryorder='array', categoryarray=cmp['bin'].tolist(), title=dict(text='Grade', font=AXIS_TITLE)),
        yaxis=dict(title=dict(text='Population share', font=AXIS_TITLE), tickformat='.0%', range=[0, float(np.nanmax(cmp[['oot_share', 'cur_share']].to_numpy())) * 1.2]),
    )
    _shift_labels(fig, cmp['bin'], cmp['cur_share'], cmp['oot_share'])
    return fig


def grade_rate_compare_chart(cmp: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name, col, color in (('OOT', 'oot_rate', WARN), ('Current', 'cur_rate', PALETTE[1])):
        fig.add_scatter(
            x=cmp['bin'], y=cmp[col], name=name, mode='lines+markers',
            line=dict(color=color, width=2.5), marker=dict(size=7, color=color),
            hovertemplate='Bin %{x}<br>' + name + ' event rate %{y:.2%}<extra></extra>',
        )
    base_layout(fig, height=360)
    fig.update_layout(
        xaxis=dict(type='category', categoryorder='array', categoryarray=cmp['bin'].tolist(), title=dict(text='Grade', font=AXIS_TITLE)),
        yaxis=dict(title=dict(text='Event rate', font=AXIS_TITLE), tickformat='.0%', range=[0, float(np.nanmax(cmp[['oot_rate', 'cur_rate']].to_numpy())) * 1.2]),
    )
    _shift_labels(fig, cmp['bin'], cmp['cur_rate'], cmp['oot_rate'])
    return fig


def grade_compare_section(frame: pd.DataFrame) -> str:
    cmp = grade_compare_frame(grade_metrics(frame))
    return f"""<section id="grade-dist"><h2>Grade distribution · OOT-valid vs overall</h2>
{pair(
    chart_card('Is the share of users in each grade still what it was at development?', grade_share_compare_chart(cmp), legend=True),
    chart_card('Is the event rate in each grade still what it was at development?', grade_rate_compare_chart(cmp), legend=True),
)}
<div class="meta">Numbers above each grade are current minus OOT in percentage points; red when the change is more than {GRADE_SHIFT_ALERT_PP:.0f}pp.</div>
</section>

"""


def figure_html(fig: go.Figure) -> str:
    fig.update_layout(autosize=True, width=None)
    return pio.to_html(
        fig, include_plotlyjs=False, full_html=False, default_width='100%',
        config={'displayModeBar': False, 'responsive': True},
    )


def segment_rate_chart(monthly: pd.DataFrame, key: str) -> go.Figure:
    """Population mix (100% stacked bars) and event rate (lines) per segment, by base_month."""
    monthly = monthly.copy()
    monthly['base_month'] = monthly['base_month'].astype(str)
    monthly['share'] = monthly['population'] / monthly.groupby('base_month')['population'].transform('sum')
    months = sorted(monthly['base_month'].unique())

    fig = go.Figure()
    groups = list(monthly.groupby(key))
    for color, (name, grp) in zip(segment_colors([g[0] for g in groups]), groups):
        row = grp.set_index('base_month').reindex(months)
        name = str(name)
        fig.add_bar(
            x=months, y=row['share'], name=name, legendgroup=name, marker_color=color, marker_line_width=0, opacity=0.4,
            customdata=row['population'], hovertemplate=f'{html.escape(name)} · ' + '%{x}<br>Mix %{y:.1%} (%{customdata:,.0f} population)<extra></extra>',
        )
        fig.add_scatter(
            x=months, y=row['event_rate'], name=name, legendgroup=name, showlegend=False, yaxis='y2',
            mode='lines+markers', line=dict(color=color, width=3, shape='spline', smoothing=0.6),
            marker=dict(size=8, color=color, line=dict(color=SURFACE, width=1.5)),
            hovertemplate=f'{html.escape(name)} · ' + '%{x}<br>Event rate %{y:.2%}<extra></extra>',
        )
    max_rate = float(monthly['event_rate'].max()) if len(monthly) else 1.0
    base_layout(fig)
    fig.update_layout(
        barmode='stack',
        bargap=0.5,
        legend=dict(groupclick='togglegroup'),
        xaxis=dict(type='category', categoryorder='array', categoryarray=months, title=dict(text='Base month', font=AXIS_TITLE)),
        yaxis=dict(title=dict(text='Population mix', font=AXIS_TITLE), tickformat='.0%', range=[0, 1]),
        yaxis2=dict(title=dict(text='Event rate', font=AXIS_TITLE), overlaying='y', side='right', tickformat='.0%', range=[0, max_rate * 1.15], showgrid=False, fixedrange=True, tickfont=dict(size=12, color=TICK)),
    )
    return fig


def vintage_table(frame: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Event rate pivot: rows = v4_bin (ascending, optionally prefixed by segment), columns = base_month."""
    keys = ['v4_bin', 'base_month'] if by is None else [by, 'v4_bin', 'base_month']
    grouped = frame.groupby(keys)['event'].mean().reset_index()
    grouped['bin_order'] = pd.to_numeric(grouped['v4_bin'], errors='coerce')
    if by is None:
        grouped['row'] = grouped['v4_bin']
        sort_cols = ['bin_order', 'v4_bin']
    else:
        grouped['row'] = grouped[by].astype(str) + ' · bin ' + grouped['v4_bin'].astype(str)
        sort_cols = [by, 'bin_order', 'v4_bin']
    rows_sorted = grouped.sort_values(sort_cols).drop_duplicates('row')['row'].tolist()
    months_sorted = sorted(grouped['base_month'].unique())
    return grouped.pivot(index='row', columns='base_month', values='event').reindex(index=rows_sorted, columns=months_sorted)


def vintage_heatmap(table: pd.DataFrame, row_title: str = 'Score bin') -> go.Figure:
    x = [str(c) for c in table.columns]
    y = table.index.tolist()
    z = table.to_numpy()
    text = [['' if pd.isna(v) else f'{v:.1%}' for v in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=x, y=y, text=text, texttemplate='%{text}', textfont=dict(size=12, color=TEXT),
            # red ramp (low -> high event rate): dark surface -> deep red -> Bad
            colorscale=[[0, SURFACE2], [0.5, '#7A3A38'], [1, BAD]], zmin=0,
            colorbar=dict(title=dict(text='Event rate', font=dict(size=13, color=TICK)), tickformat='.0%', tickfont=dict(size=12, color=TICK), outlinewidth=0),
            hovertemplate='%{y} · %{x}<br>Event rate %{z:.2%}<extra></extra>', xgap=3, ygap=3,
        )
    )
    base_layout(fig, height=max(320, 26 * len(y) + 120))
    fig.update_layout(
        xaxis=dict(type='category', title=dict(text='Base month', font=AXIS_TITLE)),
        yaxis=dict(type='category', title=dict(text=row_title, font=AXIS_TITLE), autorange='reversed'),
    )
    return fig


def html_legend(fig: go.Figure) -> str:
    """Single-row legend rendered in HTML (Plotly wraps long names onto a second row); hides the Plotly one."""
    items = {}
    for tr in fig.data:
        if tr.name and tr.showlegend is not False and tr.name not in items:
            items[tr.name] = tr.marker.color
    fig.update_layout(showlegend=False, margin=dict(t=12))
    chips = ''.join(f'<span><i style="background:{c}"></i>{html.escape(n)}</span>' for n, c in items.items())
    return f'<div class="legend">{chips}</div>'


def chart_card(heading: str, fig: go.Figure, legend: bool = False) -> str:
    """A chart with its plain-language heading (the figure itself carries no title)."""
    key = html_legend(fig) if legend else ''
    return f'<div class="chart"><h3>{html.escape(heading)}</h3>{key}{figure_html(fig)}</div>'


def pair(*cards: str) -> str:
    return f'<div class="grid">{"".join(cards)}</div>'


def vintage_grid(frame: pd.DataFrame, by: str) -> str:
    """One small bin x month heatmap per segment value (product/model name)."""
    cards = [
        chart_card(f'{name} · n={len(grp):,}', vintage_heatmap(vintage_table(grp)))
        for name, grp in frame.groupby(by)
    ]
    return pair(*cards)


def vintage_section(frame: pd.DataFrame) -> str:
    """Dropdown-switchable vintage matrix: overall / one matrix per product / one matrix per model."""
    overall_body = chart_card('Overall: how the event rate changes by score bin and base month', vintage_heatmap(vintage_table(frame)))
    views = (
        ('overall', 'Overall', overall_body),
        ('products', 'Products', vintage_grid(frame, 'product_type')),
        ('models', 'Models', vintage_grid(frame, 'model_type')),
    )
    options = ''.join(f'<option value="{key}">{label}</option>' for key, label, _ in views)
    panels = ''.join(
        f'<div id="vintage-{key}" style="{"" if i == 0 else "display:none"}">{body}</div>'
        for i, (key, _, body) in enumerate(views)
    )
    keys_js = ','.join(f"'{key}'" for key, _, _ in views)
    return f"""<div class="vintage-picker">
  <label for="vintage-select">View</label>
  <select id="vintage-select">{options}</select>
</div>
{panels}
<script>
(function () {{
  var keys = [{keys_js}];
  document.getElementById('vintage-select').addEventListener('change', function (e) {{
    keys.forEach(function (k) {{
      var el = document.getElementById('vintage-' + k);
      var active = k === e.target.value;
      el.style.display = active ? '' : 'none';
      if (active) el.querySelectorAll('.plotly-graph-div').forEach(function (g) {{ Plotly.Plots.resize(g); }});
    }});
  }});
}})();
</script>"""

def fmt(col: str, value) -> str:
    if isinstance(value, tuple):  # (level, title) status
        level, title = value
        return f'<span class="badge {level}">{html.escape(title)}</span>'
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '<span class="na">–</span>'
    if col in ('event_rate', 'share'):
        return f'{value:.2%}'
    if col == 'delta':
        return f'<span class="{"neg" if value < 0 else "pos"}">{value:+.3f}</span>'
    if col.endswith('auc'):
        return f'{value:.3f}'
    if col in ('population', 'events', 'base_month'):
        return f'{int(value):,}' if col != 'base_month' else str(int(value))
    return html.escape(str(value))


def table_html(frame: pd.DataFrame, labels: dict[str, str]) -> str:
    cols = [c for c in labels if c in frame.columns]
    head = ''.join(f'<th>{html.escape(labels[c])}</th>' for c in cols)
    body = ''.join('<tr>' + ''.join(f'<td>{fmt(c, row[c])}</td>' for c in cols) + '</tr>' for _, row in frame.iterrows())
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def grade_grid(frame: pd.DataFrame, by: str) -> str:
    cards = [
        chart_card(f'{name} · n={len(grp):,}', grade_chart(grade_metrics(grp), color))
        for color, (name, grp) in zip(segment_colors(sorted(frame[by].unique())), frame.groupby(by))
    ]
    return pair(*cards)


STATUS_LABELS = {
    'product_type': 'Product',
    'model_type': 'Model',
    'population': 'Population',
    'event_rate': 'Event rate',
    'oot_auc': 'OOT-valid AUC',
    'avg_auc': 'Avg monthly AUC',
    'latest_month': 'Latest month',
    'latest_auc': 'Latest AUC',
    'delta': 'Δ vs OOT',
    'status': 'Needed action',
}
MONTHLY_LABELS = {
    'base_month': 'Base month',
    'population': 'Population',
    'events': 'Events',
    'event_rate': 'Event rate',
    'auc': 'AUC',
    'delta': 'Δ vs OOT',
}

CSS = f"""
:root{{--bg:{BG};--surface:{SURFACE};--surface2:{SURFACE2};--line:{LINE};--text:{TEXT};--muted:{MUTED};--accent:{ACCENT};--good:{GOOD};--warn:{WARN};--bad:{BAD};color-scheme:dark}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth;scroll-padding-top:80px}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 {FONT}}}
nav{{position:sticky;top:0;z-index:10;height:64px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:20px;padding:0 max(24px,calc((100% - 1120px)/2));background:var(--surface)}}
.brand{{font-size:16px;font-weight:600;white-space:nowrap}} .brand span{{color:var(--accent)}}
.links{{display:flex;gap:4px;flex-wrap:wrap}} .links a{{color:var(--muted);text-decoration:none;font-size:12px;text-transform:uppercase;letter-spacing:.04em;padding:6px 10px;border-radius:4px}}
.links a:hover{{color:var(--text);background:var(--surface2)}} .runtime{{color:var(--muted);font-size:12px;text-align:right;white-space:nowrap}}
main{{max-width:1120px;margin:auto;padding:32px 24px 48px;display:grid;gap:20px}}
section{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:20px;display:grid;gap:16px;min-width:0}}
h2{{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0;font-weight:600}}
h3{{font-size:15px;color:var(--text);margin:0 0 8px;font-weight:600}} h3.sub{{font-size:13px;color:var(--muted);margin:8px 0 0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
.kpi{{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:16px}} .label{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}} .value{{font-size:26px;font-weight:600;margin-top:4px;letter-spacing:-.01em}} .value small{{font-size:12px;color:var(--muted);font-weight:400;margin-left:6px}}
.legend{{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:4px 16px;font-size:12px;color:var(--muted)}} .legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
.chart{{min-width:0;overflow:hidden}} .chart .plotly-graph-div{{width:100%!important}} .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}
.vintage-picker{{display:flex;align-items:center;gap:10px}}
.vintage-picker label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em}}
.vintage-picker select{{background:var(--surface);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font-size:13px}}
.action{{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:20px;border-left:3px solid var(--accent);display:grid;grid-template-columns:1fr auto;gap:20px;align-items:center}}
.action.critical{{border-left-color:var(--bad)}} .action.warning{{border-left-color:var(--warn)}} .action.healthy{{border-left-color:var(--good)}}
.action h3{{margin:0 0 4px;font-size:18px}} .action p{{margin:0;color:var(--muted);font-size:13px}}
.delta{{font-size:24px;font-weight:600;text-align:right}} .delta small{{display:block;font-size:11px;color:var(--muted);font-weight:400}}
.tablewrap{{overflow:auto;max-height:360px;border:1px solid var(--line);border-radius:8px}} table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{position:sticky;top:0;z-index:1;background:var(--surface);text-align:left;color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em;padding:10px 14px;border-bottom:1px solid var(--line)}}
td{{padding:8px 14px;border-bottom:1px solid var(--line);white-space:nowrap}} tr:last-child td{{border-bottom:0}}
tbody tr:nth-child(even) td{{background:rgba(127,168,236,.05)}} tbody tr:hover td{{background:rgba(127,168,236,.14)}}
.badge{{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600}}
.badge.healthy{{background:{GOOD_BG};color:#1F6B47}} .badge.warning{{background:{WARN_BG};color:#8A5A0B}}
.badge.critical{{background:{BAD_BG};color:#9C3524}} .badge.neutral{{background:var(--surface2);color:var(--muted)}}
.pos{{color:var(--good)}} .neg{{color:var(--bad)}} .na{{color:var(--muted)}}
.meta{{color:var(--muted);font-size:12px}} footer{{color:var(--muted);border-top:1px solid var(--line);padding-top:16px;font-size:12px}}
@media(max-width:1000px){{.grid{{grid-template-columns:1fr}}}}
@media(max-width:760px){{nav{{height:auto;flex-wrap:wrap;padding-top:12px;padding-bottom:12px}}.action{{grid-template-columns:1fr}}.delta{{text-align:left}}}}
"""


def render_report(frame: pd.DataFrame, report_month: str) -> str:
    overall_monthly = monthly_metrics(frame)
    overall_monthly['delta'] = overall_monthly['auc'] - OOT_AUC['overall']
    product_monthly = monthly_metrics(frame, 'product_type')
    model_monthly = monthly_metrics(frame, 'model_type')
    product_status = segment_status(product_monthly, 'product_type')
    model_status = segment_status(model_monthly, 'model_type')

    overall_auc = auc_or_nan(frame)
    latest, latest_month = latest_auc(overall_monthly)
    level, action_title, action_text = assess(latest, OOT_AUC['overall'])
    delta = latest - OOT_AUC['overall']
    latest_text = 'N/A' if np.isnan(latest) else f'{latest:.3f}'
    delta_text = 'N/A' if np.isnan(delta) else f'{delta:+.3f}'
    generated_at = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
    months = sorted(frame['base_month'].unique())
    worst = [
        f'{html.escape(str(r[key]))} ({r["status"][1].lower()})'
        for key, status in (('product_type', product_status), ('model_type', model_status))
        for _, r in status.iterrows()
        if r['status'][0] in ('warning', 'critical')
    ]
    segment_note = ('Segments needing attention: ' + ', '.join(worst)) if worst else 'All product and model segments are within tolerance of their OOT-valid AUC.'

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inquiry Score V4 · {html.escape(report_month)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style></head><body>
<nav>
  <div class="brand">Inquiry Score <span>V4</span></div>
  <div class="links">
    <a href="#overview">Overview</a><a href="#monthly">Monthly</a><a href="#product">Product</a>
    <a href="#model">Model</a><a href="#grades">Grades</a><a href="#action">Action</a>
  </div>
  <div class="runtime">Report {html.escape(report_month)}<br>{html.escape(generated_at)}</div>
</nav>
<main>
<section id="overview"><h2>Overview</h2>
<div class="kpis">
  <div class="kpi"><div class="label">Population</div><div class="value">{len(frame):,}</div></div>
  <div class="kpi"><div class="label">Event rate</div><div class="value">{frame['event'].mean():.2%}</div></div>
  <div class="kpi"><div class="label">Overall AUC</div><div class="value">{overall_auc:.3f}<small>OOT {OOT_AUC['overall']:.3f}</small></div></div>
  <div class="kpi"><div class="label">Latest month AUC</div><div class="value">{latest_text}<small>{html.escape(latest_month)}</small></div></div>
  <div class="kpi"><div class="label">Δ vs OOT-valid</div><div class="value {'neg' if delta < 0 else 'pos'}">{delta_text}</div></div>
</div>
<div class="meta">Aged base months {months[0]} – {months[-1]} ({len(months)} months) · source <code>{html.escape(settings.aged_pool_table)}</code> · event = 90+ DPD within 12 months after base month</div>
</section>

<section id="monthly"><h2>Monthly performance · overall</h2>
{chart_card('Is the score still ranking risk as well as at development?', trajectory_chart(overall_monthly))}
<div class="tablewrap">{table_html(overall_monthly, MONTHLY_LABELS)}</div>
</section>

<section id="product"><h2>Monthly performance · by product</h2>
{pair(
    chart_card('How well does the score rank risk for each product?', trajectory_chart(product_monthly, 'product_type'), legend=True),
    chart_card('Product mix (bars) and event rate (lines) by month', segment_rate_chart(product_monthly, 'product_type'), legend=True),
)}
<div class="tablewrap">{table_html(product_status, STATUS_LABELS)}</div>
</section>

<section id="model"><h2>Monthly performance · by model</h2>
{pair(
    chart_card('How well does the score rank risk for each model?', trajectory_chart(model_monthly, 'model_type'), legend=True),
    chart_card('Model mix (bars) and event rate (lines) by month', segment_rate_chart(model_monthly, 'model_type'), legend=True),
)}
<div class="tablewrap">{table_html(model_status, STATUS_LABELS)}</div>
</section>

{grade_compare_section(frame)}<section id="grades"><h2>Grade population &amp; event rate</h2>
{chart_card('Share of population and event rate by score grade', grade_chart(grade_metrics(frame)))}
<h3 class="sub">By product</h3>{grade_grid(frame, 'product_type')}
<h3 class="sub">By model</h3>{grade_grid(frame, 'model_type')}
<h3 class="sub">Event rate by score bin and base month</h3>
{vintage_section(frame)}
</section>

<section id="action"><h2>Needed action</h2>
<div class="action {level}"><div><h3>{html.escape(action_title)}</h3><p>{html.escape(action_text)}</p><p class="meta">{segment_note}</p></div>
<div class="delta">{delta_text}<small>OOT {OOT_AUC['overall']:.3f} → {html.escape(latest_month)} {latest_text}</small></div></div>
<div class="meta">Rules: Δ ≤ −{WARNING_DROP:.2f} → retrain or wait one month · Δ ≤ −{CRITICAL_DROP:.2f} → redevelop · otherwise continue monitoring. Applied per segment in the tables above.</div>
</section>

<footer>Event: 90+ days overdue on any invoice due within 12 months after base month. {'Higher' if settings.score_higher_is_better else 'Lower'} v4_score indicates lower risk. OOT-valid AUC benchmarks come from model development.</footer>
</main>
<script>
// Plotly only listens to window.resize; re-fit each chart whenever its card changes size
document.querySelectorAll('.chart').forEach(function (card) {{
  var graph = card.querySelector('.plotly-graph-div');
  if (graph && window.ResizeObserver) new ResizeObserver(function () {{ Plotly.Plots.resize(graph); }}).observe(card);
}});
</script>
</body></html>"""


@logging_timer(entry=True, exit=True)
def generate_report(data: pd.DataFrame, report_month: str, output: Path | None = None) -> Path:
    """Prepare data, render HTML and write it. Returns the output path."""
    frame = prepare_data(data)
    if frame.empty:
        raise ValueError(f'No rows with base_month >= {settings.report_start_month}')
    output = output or Path(settings.report_dir) / f'inquiry_score_v4_report_{report_month}.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(frame, report_month), encoding='utf-8')
    return output
