# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="Streamlit + Plotly Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Streamlit + Plotly Interactive Dashboard")
st.caption("Demo app using Plotly Express with subplots, filters, and KPIs. Replace the data loader with your own CSV when ready.")

# ----------------------------
# Data loading (cached)
# ----------------------------
@st.cache_data
def load_data():
    # Built-in sample dataset to keep the app self-contained.
    # Columns: country, continent, year, lifeExp, pop, gdpPercap, iso_alpha, iso_num
    df = px.data.gapminder()
    return df

df = load_data()

# Basic guards
if df.empty:
    st.error("No data loaded. Replace load_data() with your own dataset if needed.")
    st.stop()

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Filters")

continents = sorted(df["continent"].dropna().unique().tolist())
default_conts = continents if len(continents) <= 3 else continents[:3]
sel_continents = st.sidebar.multiselect("Continent", options=continents, default=default_conts)

# Countries depend on continent filter
if sel_continents:
    country_pool = df[df["continent"].isin(sel_continents)]["country"].unique()
else:
    country_pool = df["country"].unique()
countries = sorted(country_pool.tolist())
sel_countries = st.sidebar.multiselect("Country (optional)", options=countries, default=[])

year_min, year_max = int(df["year"].min()), int(df["year"].max())
sel_year_range = st.sidebar.slider("Year range", min_value=year_min, max_value=year_max,
                                   value=(max(year_min, year_max - 10), year_max), step=1)

metric_field = st.sidebar.selectbox("Metric", ["lifeExp", "gdpPercap", "pop"], index=0)
group_field = st.sidebar.selectbox("Group/Color", ["continent", "country"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: When you switch to your own data, adjust filters to your column names.")

# ----------------------------
# Filtering
# ----------------------------
mask = df["year"].between(sel_year_range[0], sel_year_range[1])
if sel_continents:
    mask &= df["continent"].isin(sel_continents)
if sel_countries:
    mask &= df["country"].isin(sel_countries)

fdf = df.loc[mask].copy()
latest_year = int(fdf["year"].max()) if not fdf.empty else int(df["year"].max())
f_latest = fdf[fdf["year"] == latest_year].copy() if not fdf.empty else pd.DataFrame()

# ----------------------------
# KPI / Key insights
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
if not fdf.empty:
    k1.metric(f"{metric_field} • mean", f"{fdf[metric_field].mean():,.2f}")
    k2.metric(f"{metric_field} • median", f"{fdf[metric_field].median():,.2f}")
    k3.metric(f"{metric_field} • min", f"{fdf[metric_field].min():,.2f}")
    k4.metric(f"{metric_field} • max", f"{fdf[metric_field].max():,.2f}")
else:
    for c in (k1, k2, k3, k4):
        c.metric("No data", "—")

st.markdown("")

# ----------------------------
# Main chart (overview)
# ----------------------------
st.subheader("Overview Chart")
if fdf.empty:
    st.info("No rows match your filters. Try widening the year range or selecting more continents/countries.")
else:
    fig_main = px.line(
        fdf.sort_values(["country", "year"]),
        x="year",
        y=metric_field,
        color=group_field,
        hover_data=["country", "continent"],
        markers=True,
        title=f"{metric_field} over time by {group_field}"
    )
    fig_main.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=450)
    st.plotly_chart(fig_main, use_container_width=True)

# ----------------------------
# Subplots (like the Part 3 tutorial)
# ----------------------------
st.subheader("Subplots: Distribution (top) + Latest-Year Scatter (bottom)")

if fdf.empty:
    st.info("Subplots unavailable because the filtered dataset is empty.")
else:
    # Top subplot uses entire selected range; bottom uses the latest year in the filtered data
    fig_sub = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=(
            f"Distribution of {metric_field} by {group_field} ({sel_year_range[0]}–{sel_year_range[1]})",
            f"{metric_field} vs GDP per Capita • Latest Year: {latest_year}"
        )
    )

    # --- Row 1: Box plots by selected group (aggregating across the chosen year range)
    if group_field == "continent":
        for cont, sub in fdf.groupby("continent"):
            fig_sub.add_trace(
                go.Box(
                    y=sub[metric_field],
                    name=str(cont),
                    boxmean=True,
                    hovertext=sub["country"],
                    hovertemplate="<b>%{hovertext}</b><br>"
                                  f"{metric_field}: %{y:,.2f}<extra></extra>"
                ),
                row=1, col=1
            )
    else:  # group_field == "country"
        # To avoid an overly long legend/labels, limit to top N countries by count in filtered data
        top_countries = (
            fdf["country"]
            .value_counts()
            .head(12)  # adjust as you like
            .index
            .tolist()
        )
        for ctry in top_countries:
            sub = fdf[fdf["country"] == ctry]
            fig_sub.add_trace(
                go.Box(
                    y=sub[metric_field],
                    name=str(ctry),
                    boxmean=True,
                    hovertext=sub["country"],
                    hovertemplate="<b>%{hovertext}</b><br>"
                                  f"{metric_field}: %{y:,.2f}<extra></extra>"
                ),
                row=1, col=1
            )

    # --- Row 2: Scatter (latest year in the filtered data)
    if not f_latest.empty:
        fig_sub.add_trace(
            go.Scatter(
                x=f_latest["gdpPercap"],
                y=f_latest[metric_field],
                mode="markers",
                text=f_latest["country"],
                hovertemplate="<b>%{text}</b><br>GDP per Capita: %{x:,.0f}<br>"
                              f"{metric_field}: %{y:,.2f}<extra></extra>"
            ),
            row=2, col=1
        )

    # Axes and layout
    fig_sub.update_yaxes(title_text=metric_field, row=1, col=1)
    fig_sub.update_xaxes(title_text="gdpPercap", row=2, col=1)
    fig_sub.update_yaxes(title_text=metric_field, row=2, col=1)

    fig_sub.update_layout(
        height=900,
        showlegend=False,
        margin=dict(l=10, r=10, t=70, b=10)
    )

    st.plotly_chart(fig_sub, use_container_width=True)

# ----------------------------
# Data preview
# ----------------------------
with st.expander("🔎 Peek at filtered data"):
    st.dataframe(fdf, use_container_width=True)

# ----------------------------
# Notes for adapting to your own data
# ----------------------------
st.markdown(
    """
**How to use your own CSV**

1. Replace the `load_data()` function with:  
   `df = pd.read_csv("data/your_data.csv")` (and parse dates if needed).
2. Update the sidebar filters to match your column names (e.g., categories, years, metrics).
3. Adjust charts to use the fields you care about.
"""
)


