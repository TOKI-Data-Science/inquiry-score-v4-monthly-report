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
WARNING_DROP = 0.05
CRITICAL_DROP = 0.10
REQUIRED_COLUMNS = {'id', 'product_type', 'model_type', 'v4_score', 'v4_bin', 'base_month', 'event'}

ACCENT = '#7289da'
PALETTE = [ACCENT, "#ebcf45", "#f25776", "#9a5cfe", '#ff8c5a', '#45c7e9', '#b39dfb', '#ed4245']
BG, SURFACE, SURFACE2, LINE, TEXT, MUTED = '#050507', '#0c0d11', '#15161c', '#1f2029', '#f2f3f5', "#adb2bb"
BAR = '#8ea1e1'
HOVER_BG = '#0a0b0e'
GOOD, WARN, BAD = '#57f287', '#fee75c', '#ed4245'
TICK = '#b7bcc6' 
GRADE_INVERSION_THRESHOLD = 0.01 

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
    outcomes = grouped['event'].agg(customers='size', events='sum', event_rate='mean').reset_index()
    auc = grouped[['v4_score', 'event']].apply(auc_or_nan).rename('auc').reset_index()
    return outcomes.merge(auc, on=keys).sort_values(keys).reset_index(drop=True)


def grade_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    grade = frame.groupby('v4_bin', dropna=False)['event'].agg(customers='size', events='sum', event_rate='mean').reset_index()
    grade['share'] = grade['customers'] / grade['customers'].sum()
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
                'customers': int(grp['customers'].sum()),
                'event_rate': grp['events'].sum() / grp['customers'].sum(),
                'oot_auc': oot,
                'avg_auc': grp['auc'].mean(),
                'latest_month': month,
                'latest_auc': latest,
                'delta': latest - oot,
                'status': (level, title),
            }
        )
    return pd.DataFrame(rows)

def base_layout(fig: go.Figure, title: str, height: int = 380) -> None:
    fig.update_layout(
        title=dict(text=f'<b>{title}</b>', font=dict(size=15, color=TEXT), x=0.02, y=0.96),
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family='Segoe UI, sans-serif', color=TICK, size=12),
        margin=dict(l=48, r=48, t=60, b=44),
        hoverlabel=dict(bgcolor=HOVER_BG, font_color=TEXT, bordercolor=LINE, font_size=12),
        legend=dict(orientation='h', y=1.12, x=1, xanchor='right', bgcolor='rgba(0,0,0,0)', font=dict(size=12, color=TICK)),
        xaxis=dict(showgrid=False, zeroline=False, fixedrange=True, showline=True, linecolor=LINE, tickfont=dict(size=11, color=TICK)),
        yaxis=dict(showgrid=False, zeroline=False, fixedrange=True, tickfont=dict(size=11, color=TICK)),
    )


def trajectory_chart(monthly: pd.DataFrame, title: str, by: str | None = None) -> go.Figure:
    months = sorted(monthly['base_month'].astype(str).unique())
    categories = ['OOT-valid', *months]
    if by is None:
        series = [('Overall', monthly, OOT_AUC['overall'])]
    else:
        series = [(str(name), grp, OOT_AUC[by].get(str(name), np.nan)) for name, grp in monthly.groupby(by)]

    fig = go.Figure()
    all_values = []
    for color, (name, grp, oot) in zip(cycle(PALETTE), series):
        by_month = grp.set_index(grp['base_month'].astype(str))['auc'].reindex(months)
        y = [oot, *by_month.tolist()]
        all_values.extend(v for v in y if pd.notna(v))
        fig.add_scatter(
            x=categories,
            y=y,
            name=name,
            mode='lines+markers+text',
            text=['' if pd.isna(v) else f'{v:.3f}' for v in y],
            textposition='top center',
            textfont=dict(color=color if by else TEXT, size=11),
            line=dict(color=color, width=2.5, shape='spline', smoothing=0.6),
            marker=dict(size=8, color=color, symbol=['diamond', *['circle'] * len(months)]),
            hovertemplate='%{x}<br>AUC %{y:.3f}<extra>' + html.escape(name) + '</extra>',
        )
    fig.add_vrect(x0=-0.5, x1=0.5, fillcolor=ACCENT, opacity=0.08, line_width=0)
    lo, hi = (min(all_values), max(all_values)) if all_values else (0.5, 0.8)
    base_layout(fig, title)
    fig.update_layout(showlegend=by is not None)
    fig.update_xaxes(type='category', categoryorder='array', categoryarray=categories)
    fig.update_yaxes(range=[lo - 0.03, hi + 0.03], tickformat='.2f')
    return fig


def grade_chart(grade: pd.DataFrame, title: str, color: str = ACCENT) -> go.Figure:
    bins = grade['v4_bin'].tolist()
    fig = go.Figure()
    fig.add_bar(
        x=bins, y=grade['share'], name='Population share', marker_color=BAR, marker_line_width=0, opacity=0.55,
        customdata=grade['customers'],
        hovertemplate='Bin %{x}<br>Share %{y:.1%} (%{customdata:,} customers)<extra></extra>',
    )
    fig.add_scatter(
        x=bins, y=grade['event_rate'], name='Event rate', yaxis='y2', mode='lines+markers+text',
        text=[f'{v:.1%}' for v in grade['event_rate']], textposition='top center', textfont=dict(size=11, color=TEXT),
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
    base_layout(fig, title, height=340)
    fig.update_layout(
        xaxis=dict(type='category', categoryorder='array', categoryarray=bins),
        yaxis=dict(title=dict(text='Population share', font=dict(size=11)), tickformat='.0%', rangemode='tozero'),
        yaxis2=dict(title=dict(text='Event rate', font=dict(size=11)), overlaying='y', side='right', tickformat='.0%', rangemode='tozero', showgrid=False, fixedrange=True, tickfont=dict(size=11, color=TICK)),
        bargap=0.35,
    )
    return fig


def figure_html(fig: go.Figure) -> str:
    fig.update_layout(autosize=True, width=None)
    return pio.to_html(
        fig, include_plotlyjs=False, full_html=False, default_width='100%',
        config={'displayModeBar': False, 'responsive': True},
    )

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
    if col in ('customers', 'events', 'base_month'):
        return f'{int(value):,}' if col != 'base_month' else str(int(value))
    return html.escape(str(value))


def table_html(frame: pd.DataFrame, labels: dict[str, str]) -> str:
    cols = [c for c in labels if c in frame.columns]
    head = ''.join(f'<th>{html.escape(labels[c])}</th>' for c in cols)
    body = ''.join('<tr>' + ''.join(f'<td>{fmt(c, row[c])}</td>' for c in cols) + '</tr>' for _, row in frame.iterrows())
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def grade_grid(frame: pd.DataFrame, by: str) -> str:
    cards = []
    for color, (name, grp) in zip(cycle(PALETTE), frame.groupby(by)):
        cards.append(f'<div class="chart">{figure_html(grade_chart(grade_metrics(grp), f"{html.escape(str(name))} · n={len(grp):,}", color))}</div>')
    return f'<div class="grid">{"".join(cards)}</div>'


STATUS_LABELS = {
    'product_type': 'Product',
    'model_type': 'Model',
    'customers': 'Customers',
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
    'customers': 'Customers',
    'events': 'Events',
    'event_rate': 'Event rate',
    'auc': 'AUC',
    'delta': 'Δ vs OOT',
}

CSS = f"""
:root{{--bg:{BG};--surface:{SURFACE};--surface2:{SURFACE2};--line:{LINE};--text:{TEXT};--muted:{MUTED};--accent:{ACCENT};--good:{GOOD};--warn:{WARN};--bad:{BAD}}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth;scroll-padding-top:80px}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 "Segoe UI",system-ui,sans-serif}}
nav{{position:sticky;top:0;z-index:10;height:64px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:20px;padding:0 max(24px,calc((100% - 1200px)/2));background:rgba(5,5,7,.94);backdrop-filter:blur(8px)}}
.brand{{font-size:16px;font-weight:650;white-space:nowrap}} .brand span{{color:var(--accent)}}
.links{{display:flex;gap:4px;flex-wrap:wrap}} .links a{{color:var(--muted);text-decoration:none;font-size:12px;text-transform:uppercase;letter-spacing:.04em;padding:6px 10px;border-radius:4px}}
.links a:hover{{color:var(--text);background:var(--surface2)}} .runtime{{color:var(--muted);font-size:12px;text-align:right;white-space:nowrap}}
main{{max-width:1200px;margin:auto;padding:34px 24px 50px}} section{{margin-bottom:34px}}
h2{{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 12px;font-weight:700}}
h3{{font-size:13px;color:#c7cad1;margin:18px 0 8px;font-weight:600}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
.kpi,.chart,.action,.tablewrap{{background:var(--surface);border:1px solid var(--line);border-radius:8px}}
.kpi{{padding:18px 20px}} .label{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}} .value{{font-size:28px;font-weight:700;margin-top:4px;letter-spacing:-.01em}} .value small{{font-size:12px;color:var(--muted);font-weight:500;margin-left:6px}}
.chart{{padding:6px;min-width:0;overflow:hidden}} .chart .plotly-graph-div{{width:100%!important}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(460px,100%),1fr));gap:12px}}
.action{{padding:22px;border-left:3px solid var(--accent);display:grid;grid-template-columns:1fr auto;gap:20px;align-items:center;margin-bottom:12px}}
.action.critical{{border-left-color:var(--bad)}} .action.warning{{border-left-color:var(--warn)}} .action.healthy{{border-left-color:var(--good)}}
.action h3{{margin:0 0 5px;font-size:19px;color:var(--text);font-weight:700}} .action p{{margin:0;color:var(--muted);font-size:13px}}
.delta{{font-size:26px;font-weight:700;text-align:right}} .delta small{{display:block;font-size:11px;color:var(--muted);font-weight:500}}
.tablewrap{{overflow-x:auto;margin-top:10px}} table{{width:100%;border-collapse:collapse;font-size:13.5px}}
th{{text-align:left;color:var(--muted);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.05em;padding:10px 14px;border-bottom:1px solid var(--line)}}
td{{padding:9px 14px;border-bottom:1px solid var(--surface2);white-space:nowrap}} tr:last-child td{{border-bottom:0}} tr:hover td{{background:var(--surface2)}}
td:first-child{{color:var(--text);font-weight:600}}
.badge{{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600}}
.badge.healthy{{background:rgba(87,242,135,.14);color:var(--good)}} .badge.warning{{background:rgba(254,231,92,.14);color:var(--warn)}}
.badge.critical{{background:rgba(237,66,69,.16);color:var(--bad)}} .badge.neutral{{background:rgba(148,155,164,.15);color:var(--muted)}}
.pos{{color:var(--good)}} .neg{{color:var(--bad)}} .na{{color:var(--muted)}}
.meta{{color:var(--muted);font-size:12px;margin-top:10px}} footer{{color:#6d6f78;border-top:1px solid var(--line);padding-top:18px;font-size:12px}}
@media(max-width:760px){{nav{{height:auto;flex-wrap:wrap;padding-top:12px;padding-bottom:12px}}.grid{{grid-template-columns:1fr}}.action{{grid-template-columns:1fr}}.delta{{text-align:left}}}}
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
  <div class="kpi"><div class="label">Customers</div><div class="value">{len(frame):,}</div></div>
  <div class="kpi"><div class="label">Event rate</div><div class="value">{frame['event'].mean():.2%}</div></div>
  <div class="kpi"><div class="label">Overall AUC</div><div class="value">{overall_auc:.3f}<small>OOT {OOT_AUC['overall']:.3f}</small></div></div>
  <div class="kpi"><div class="label">Latest month AUC</div><div class="value">{latest_text}<small>{html.escape(latest_month)}</small></div></div>
  <div class="kpi"><div class="label">Δ vs OOT-valid</div><div class="value {'neg' if delta < 0 else 'pos'}">{delta_text}</div></div>
</div>
<div class="meta">Aged base months {months[0]} – {months[-1]} ({len(months)} months) · source <code>{html.escape(settings.aged_pool_table)}</code> · event = 90+ DPD within 12 months after base month</div>
</section>

<section id="monthly"><h2>Monthly performance · overall</h2>
<div class="chart">{figure_html(trajectory_chart(overall_monthly, 'Overall AUC trajectory · OOT-valid → monthly'))}</div>
<div class="tablewrap">{table_html(overall_monthly, MONTHLY_LABELS)}</div>
</section>

<section id="product"><h2>Monthly performance · by product</h2>
<div class="chart">{figure_html(trajectory_chart(product_monthly, 'AUC by product_type', 'product_type'))}</div>
<div class="tablewrap">{table_html(product_status, STATUS_LABELS)}</div>
</section>

<section id="model"><h2>Monthly performance · by model</h2>
<div class="chart">{figure_html(trajectory_chart(model_monthly, 'AUC by model_type', 'model_type'))}</div>
<div class="tablewrap">{table_html(model_status, STATUS_LABELS)}</div>
</section>

<section id="grades"><h2>Grade population &amp; event rate</h2>
<div class="chart">{figure_html(grade_chart(grade_metrics(frame), 'Overall'))}</div>
<h3>By product</h3>{grade_grid(frame, 'product_type')}
<h3>By model</h3>{grade_grid(frame, 'model_type')}
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
