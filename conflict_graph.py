# full_dashboard_with_trend.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import HeatMap

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Traffic Conflict & Volume Dashboard", layout="wide")
st.title("📊 Traffic Conflict & Traffic Volume Dashboard")

# =============================
# 1️⃣ UPLOAD CONFLICT DATA
# =============================
st.subheader("📂 Upload Conflict Data (CSV or Excel)")
uploaded_conflict_file = st.file_uploader(
    "Upload conflict dataset",
    type=["csv", "xlsx"],
    key="conflict_uploader"
)

if uploaded_conflict_file:
    if uploaded_conflict_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_conflict_file)
    else:
        df = pd.read_excel(uploaded_conflict_file)
    st.success(f"Loaded {len(df)} records × {len(df.columns)} columns (Conflict Data)")

    # Excel date conversion
    if "Day" in df.columns:
        df["Day"] = pd.to_numeric(df["Day"], errors="coerce")
        df["Day_dt"] = pd.to_datetime(df["Day"], unit='d', origin='1899-12-30')
        df["Day_only"] = df["Day_dt"].dt.date
        df["Hour"] = (df["Day"] % 1 * 24).astype(int)
    else:
        st.error("No 'Day' column found in conflict dataset.")
        st.stop()

    # Map encounter types
    if "Encounter_type" in df.columns:
        df["Encounter_grouped"] = df["Encounter_type"].apply(
            lambda enc: "Merging" if enc in ["Adjacent-Approaches", "Opposing-through"] else enc
        )
    else:
        df["Encounter_grouped"] = "Unknown"

    # -----------------------------
    # DAILY DISTRIBUTION
    # -----------------------------
    st.subheader("📅 Daily Distribution of Conflicts")
    daily_conflicts = df.groupby("Day_only").size().reset_index(name="Number of Conflicts").sort_values("Day_only")
    # Add trend line (rolling mean)
    daily_conflicts["Trend"] = daily_conflicts["Number of Conflicts"].rolling(window=3, min_periods=1).mean()
    fig_daily = px.bar(daily_conflicts, x="Day_only", y="Number of Conflicts", width=900, height=500)
    fig_daily.add_scatter(x=daily_conflicts["Day_only"], y=daily_conflicts["Trend"], mode="lines", name="Trend", line=dict(color="orange", width=3))
    fig_daily.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=daily_conflicts["Day_only"],
            ticktext=[d.strftime("%d-%b (%a)") for d in pd.to_datetime(daily_conflicts["Day_only"])],
            tickangle=-45
        )
    )
    st.plotly_chart(fig_daily, use_container_width=False)

    # -----------------------------
    # TEMPORAL DISTRIBUTION (5-19H)
    # -----------------------------
    st.subheader("⏱ Temporal Distribution of Conflicts (5 AM - 7 PM)")
    df_temp = df[(df["Hour"] >= 5) & (df["Hour"] <= 19)]
    hour_bins = list(range(5, 20))
    hour_count = df_temp.groupby("Hour").size().reindex(hour_bins, fill_value=0).reset_index()
    hour_count.columns = ["Hour", "Number of Conflicts"]
    hour_count["Hour Interval"] = [f"{h}:00 - {h+1}:00" for h in hour_count["Hour"]]
    # Trend line
    hour_count["Trend"] = hour_count["Number of Conflicts"].rolling(window=2, min_periods=1).mean()
    fig_hourly = px.bar(hour_count, x="Hour Interval", y="Number of Conflicts", width=900, height=500)
    fig_hourly.add_scatter(x=hour_count["Hour Interval"], y=hour_count["Trend"], mode="lines", name="Trend", line=dict(color="orange", width=3))
    st.plotly_chart(fig_hourly, use_container_width=False)

    # -----------------------------
    # ENCOUNTER TYPE PIE
    # -----------------------------
    st.subheader("🥧 Distribution of Conflicts by Encounter Type")
    encounter_counts = df["Encounter_grouped"].value_counts().reset_index()
    encounter_counts.columns = ["Encounter_type", "Count"]
    encounter_counts["Label"] = encounter_counts.apply(lambda row: f"{row['Encounter_type']}: ({row['Count']})", axis=1)
    fig_pie = px.pie(encounter_counts, names="Label", values="Count", hole=0.3)
    fig_pie.update_traces(textinfo="label+percent", insidetextorientation='radial')
    st.plotly_chart(fig_pie, use_container_width=True)

    # -----------------------------
    # ROAD USER PIE CHARTS
    # -----------------------------
    st.subheader("🛣 Conflict Count by Road User Type (Follower)")
    def map_roaduser_category(code):
        if code in [3, 23]:
            return "Passenger car"
        elif code in [4, 9, 10, 11, 15]:
            return "Pedestrian"
        elif code == 5:
            return "Bicycle"
        elif code in [6, 14]:
            return "Motorbike/Scooters"
        elif code in [7, 8]:
            return "Ute/Pickup truck"
        elif code in [12, 13, 16, 17, 18, 24]:
            return "Others"
        elif code in [1, 2, 19, 20, 21, 22]:
            return "Heavy vehicle"
        else:
            return "Unknown"

    def plot_roaduser2_pie_reduced(df_in, conflict_name):
        if df_in.empty or "RoadUser2_type" not in df_in.columns:
            st.info(f"No data for {conflict_name}")
            return
        users2_mapped = df_in["RoadUser2_type"].map(map_roaduser_category)
        counts = users2_mapped.value_counts().reset_index()
        counts.columns = ["Road User", "Count"]
        counts["Label"] = counts.apply(lambda row: f"{row['Road User']}: ({row['Count']})", axis=1)
        fig = px.pie(counts, names="Label", values="Count", title=f"{conflict_name} Conflicts by Road User Type", hole=0.1)
        fig.update_traces(textinfo="label+percent", insidetextorientation='radial')
        st.plotly_chart(fig, use_container_width=True)

    rear_end_df = df[df["Encounter_grouped"] == "Rear-End"]
    vru_df = df[df["Encounter_grouped"] == "VRU"]
    merging_df = df[df["Encounter_grouped"] == "Merging"]
    cols_ru = st.columns(3)
    with cols_ru[0]:
        plot_roaduser2_pie_reduced(rear_end_df, "Rear-End")
    with cols_ru[1]:
        plot_roaduser2_pie_reduced(vru_df, "VRU")
    with cols_ru[2]:
        plot_roaduser2_pie_reduced(merging_df, "Merging")

    # -----------------------------
    # HISTOGRAMS
    # -----------------------------
    display_labels = {
        "ttc": "TTC",
        "ttc_deltav": "TTC DeltaV",
        "total_conflict_duration_sec": "Conflict duration (TTC<3), sec",
        "pet": "PET",
        "Gap_time": "Gap time",
        "Gap_distance": "Gap distance"
    }

    conflict_hist_vars = {
        "Rear-End": ["ttc", "ttc_deltav", "total_conflict_duration_sec"],
        "VRU": ["pet", "Gap_time", "Gap_distance"],
        "Merging": ["pet", "Gap_time", "Gap_distance"]
    }

    bin_widths = {
        "ttc": 0.5,
        "ttc_deltav": 2,
        "total_conflict_duration_sec": 0.5,
        "pet": 0.5,
        "Gap_time": 0.5,
        "Gap_distance": 1
    }

    def plot_histogram_bins(df_in, column, title):
        if column not in df_in.columns or df_in[column].dropna().empty:
            st.info(f"No {title} values available")
            return
        df_in = df_in.copy()
        max_val = df_in[column].max()
        bin_width = bin_widths.get(column, 0.5)
        bins = list(np.arange(0, max_val + bin_width, bin_width))
        df_in["Bin"] = pd.cut(df_in[column], bins=bins, include_lowest=True)
        bin_counts = df_in.groupby("Bin").size().reset_index(name="Frequency")
        bin_counts["Bin_label"] = bin_counts["Bin"].apply(lambda x: f"{x.left:.1f}-{x.right:.1f}")
        fig = px.bar(bin_counts, x="Bin_label", y="Frequency", title=title)
        fig.update_layout(xaxis_title=title, yaxis_title="Frequency", title_font_size=20)
        st.plotly_chart(fig, use_container_width=True)

    for conflict_type in df["Encounter_grouped"].unique():
        st.markdown(f"### {conflict_type} Conflicts")
        df_conflict = df[df["Encounter_grouped"] == conflict_type]
        cols = st.columns(3)
        i = 0
        for var in conflict_hist_vars.get(conflict_type, []):
            display_name = display_labels.get(var, var)
            with cols[i % 3]:
                plot_histogram_bins(df_conflict, var, f"{conflict_type} - {display_name}")
            i += 1

    # -----------------------------
    # DESCRIPTIVE STATISTICS
    # -----------------------------
    st.subheader("📋 Descriptive Statistics by Conflict Type")
    rows = []
    for conflict_type in df["Encounter_grouped"].unique():
        df_conflict = df[df["Encounter_grouped"] == conflict_type]
        for var in conflict_hist_vars.get(conflict_type, []):
            display_name = display_labels.get(var, var)
            if var in df_conflict.columns and df_conflict[var].dropna().any():
                desc = df_conflict[var].describe()
                row = {
                    "Conflict Type": conflict_type,
                    "Variable": display_name,
                    "Count": int(desc["count"]),
                    "Mean": round(desc["mean"], 3),
                    "Min": round(desc["min"], 3),
                    "Max": round(desc["max"], 3)
                }
            else:
                row = {"Conflict Type": conflict_type, "Variable": display_name, "Count":0, "Mean":None, "Min":None, "Max":None}
            rows.append(row)
    stats_df = pd.DataFrame(rows)
    stats_df["Conflict Type Display"] = stats_df["Conflict Type"]
    stats_df.loc[stats_df["Conflict Type"].duplicated(), "Conflict Type Display"] = ""
    stats_df = stats_df[["Conflict Type Display", "Variable", "Count", "Mean", "Min", "Max"]]
    st.dataframe(stats_df)

    # -----------------------------
    # HEATMAPS
    # -----------------------------
    st.subheader("🌍 Conflict Heatmaps by Type")
    heatmap_configs = {
        "Rear-End": ("ttc_lat", "ttc_lng"),
        "VRU": ("pet_lat", "pet_lng"),
        "Merging": ("pet_lat", "pet_lng")
    }
    zoom = st.slider("Select Heatmap Zoom Level", 12, 22, 17)
    map_width, map_height = 700, 700
    cols = st.columns(3)

    def create_heatmap(conflict_df, lat_col, lon_col):
        if conflict_df.empty or lat_col not in conflict_df.columns or lon_col not in conflict_df.columns:
            return None
        conflict_df = conflict_df[(conflict_df[lat_col] != 0) & (conflict_df[lon_col] != 0)]
        center_lat = float(conflict_df[lat_col].mean())
        center_lon = float(conflict_df[lon_col].mean())
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None, max_zoom=22)
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite",
            name="Google Satellite",
            max_zoom=22
        ).add_to(m)
        points = conflict_df[[lat_col, lon_col]].dropna().values.tolist()
        if points:
            HeatMap(points, radius=10, blur=15, max_zoom=22).add_to(m)
        folium.LayerControl().add_to(m)
        return m

    for i, (conflict_type, (lat_col, lon_col)) in enumerate(heatmap_configs.items()):
        df_conflict = df[df["Encounter_grouped"] == conflict_type]
        m = create_heatmap(df_conflict, lat_col, lon_col)
        with cols[i]:
            if m:
                st.markdown(f"**{conflict_type} Heatmap**")
                st.components.v1.html(m._repr_html_(), width=map_width, height=map_height, scrolling=True)
            else:
                st.info(f"No data for {conflict_type}")

# =============================
# 2️⃣ UPLOAD TRAFFIC VOLUME DATA
# =============================
st.subheader("📂 Upload Traffic Volume Excel File")
uploaded_volume_file = st.file_uploader(
    "Upload Traffic Volume Excel File",
    type=["xlsx"],
    key="volume_uploader"
)

if uploaded_volume_file:
    xls = pd.ExcelFile(uploaded_volume_file)
    for sheet in xls.sheet_names:
        st.markdown(f"### Traffic Volume - {sheet}")
        df_vol = pd.read_excel(uploaded_volume_file, sheet_name=sheet)
        df_vol["Date"] = pd.to_datetime(df_vol["Date"], errors="coerce")
        df_vol["Day_only"] = df_vol["Date"].dt.date

        # Sum numeric vehicle columns
        vehicle_cols = df_vol.columns[3:]
        vehicle_cols_numeric = df_vol[vehicle_cols].select_dtypes(include=np.number).columns
        df_vol["Total Volume"] = df_vol[vehicle_cols_numeric].sum(axis=1)

        # Hour calculation
        df_vol["IntervalStart"] = pd.to_datetime(df_vol["IntervalStart"], errors="coerce").dt.hour
        df_vol["IntervalEnd"] = pd.to_datetime(df_vol["IntervalEnd"], errors="coerce").dt.hour
        df_vol["Hour"] = ((df_vol["IntervalStart"] + df_vol["IntervalEnd"]) / 2).astype(int)
        df_hour = df_vol[(df_vol["Hour"] >= 5) & (df_vol["Hour"] <= 19)]
        hour_bins = list(range(5, 20))
        hourly_volume = df_hour.groupby("Hour")["Total Volume"].sum().reindex(hour_bins, fill_value=0).reset_index()
        hourly_volume.columns = ["Hour", "Total Volume"]
        hourly_volume["Hour Interval"] = [f"{h}:00 - {h+1}:00" for h in hourly_volume["Hour"]]

        # -----------------------------
        # Side by side plot
        # -----------------------------
        cols_vol = st.columns(2)
        with cols_vol[0]:
            daily_volume = df_vol.groupby("Day_only")["Total Volume"].sum().reset_index()
            daily_volume["Trend"] = daily_volume["Total Volume"].rolling(window=3, min_periods=1).mean()
            fig_daily = px.bar(daily_volume, x="Day_only", y="Total Volume", width=900, height=500)
            fig_daily.add_scatter(x=daily_volume["Day_only"], y=daily_volume["Trend"], mode="lines", name="Trend", line=dict(color="orange", width=3))
            st.plotly_chart(fig_daily, use_container_width=False)

        with cols_vol[1]:
            hourly_volume["Trend"] = hourly_volume["Total Volume"].rolling(window=2, min_periods=1).mean()
            fig_hourly = px.bar(hourly_volume, x="Hour Interval", y="Total Volume", width=900, height=500)
            fig_hourly.add_scatter(x=hourly_volume["Hour Interval"], y=hourly_volume["Trend"], mode="lines", name="Trend", line=dict(color="orange", width=3))
            st.plotly_chart(fig_hourly, use_container_width=False)
