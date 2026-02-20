# full_dashboard_with_trend_updated.py
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
            lambda enc: "Merging" if enc in ["Adjacent-Approaches", "Opposing-through", "Opposing-Approaches"] else enc
        )
    else:
        df["Encounter_grouped"] = "Unknown"
    # ✅ Save conflict df for later rate calculation
    st.session_state["conflict_df"] = df.copy()

    # -----------------------------
    # DAILY DISTRIBUTION
    # -----------------------------
    st.subheader("📅 Daily Distribution of Conflicts")
    daily_conflicts = df.groupby("Day_only").size().reset_index(name="Number of Conflicts").sort_values("Day_only")
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
    hour_count["Trend"] = hour_count["Number of Conflicts"].rolling(window=2, min_periods=1).mean()
    fig_hourly = px.bar(hour_count, x="Hour Interval", y="Number of Conflicts", width=900, height=500)
    fig_hourly.add_scatter(x=hour_count["Hour Interval"], y=hour_count["Trend"], mode="lines", name="Trend", line=dict(color="orange", width=3))
    st.plotly_chart(fig_hourly, use_container_width=False)

    # -----------------------------
    # ENCOUNTER TYPE PIE
    # -----------------------------
    desired_order = ["Vehicle-VRU", "Merging", "Rear-End"]

    encounter_counts = (
        df["Encounter_grouped"]
        .replace({"VRU": "Vehicle-VRU"})
        .value_counts()
        .reindex(desired_order, fill_value=0)
        .reset_index()
    )
    
    encounter_counts.columns = ["Encounter_type", "Count"]
    
    color_map = {
        "Vehicle-VRU": "red",
        "Rear-End": "Blue",
        "Merging": "lightblue"
    }
    
    fig_pie = px.pie(
        encounter_counts,
        names="Encounter_type",
        values="Count",
        hole=0.3,
        color="Encounter_type",
        color_discrete_map=color_map
    )
    
    fig_pie.update_traces(
        sort=False,
        rotation=0,  # start at 12 o’clock
        texttemplate="%{label}<br>%{value} (%{percent})",
        textposition="inside"
    )
    
    st.plotly_chart(fig_pie, use_container_width=True, key="pie_encounter")




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
        "ttc": "TTC (s)",
        "ttc_deltav": "TTC ΔV",
        "total_conflict_duration_sec": "Conflict duration (TTC<3), sec",
        "pet": "PET (s)",
        "Gap_time": "Gap time (s)",
        "Gap_distance": "Gap distance (m)",
        "max_DRAC": "Max DRAC (m/s²)",     # ✅ new
        "DeltaV": "Delta-V (km/h)" ,
        "mttc_ttc": "MTTC/TTC"        # ✅ new
    }

    conflict_hist_vars = {
        "Rear-End": ["ttc", "ttc_deltav", "total_conflict_duration_sec", "max_DRAC", "mttc_ttc"],  # ✅ added max_DRAC and mttc_ttc
        "VRU": ["pet", "Gap_time", "Gap_distance", "DeltaV"],                          # ✅ added DeltaV
        "Merging": ["pet", "Gap_time", "Gap_distance", "DeltaV"]                       # ✅ added DeltaV
    }


    BIN_COUNT_CONTROL_VARS = {"Gap_distance", "DeltaV", "max_DRAC"}
    MIN_BINS = 5
    MAX_BINS = 6

    bin_widths = {
        "ttc": 0.5,
        "ttc_deltav": 2,
        "total_conflict_duration_sec": 0.5,
        "pet": 0.5,
        "Gap_time": 1,
        "mttc_ttc": 0.5
    }

    force_zero_underflow = {"ttc", "ttc_deltav", "total_conflict_duration_sec", "pet", "Gap_time", "mttc_ttc"}
    force_zero_overflow = {"ttc", "ttc_deltav", "total_conflict_duration_sec", "pet", "Gap_time", "mttc_ttc"}

    def safe_underflow(df, col):
        if col not in df.columns:
            return 0
        col_min = df[col].dropna().min()
        if pd.isna(col_min):
            return 0
        if col in force_zero_underflow:
            return 0
        return col_min
    
    def safe_overflow(df, col):
        if col not in df.columns:
            return 0
        col_max = df[col].dropna().max()
        if pd.isna(col_max):
            return 0
        if col in force_zero_overflow:
            return 0
        return col_max

    underflow_bins = {
        "ttc": 0,
        "mttc_ttc": 0,
        "ttc_deltav": 0,
        "total_conflict_duration_sec": 0,
        "pet": 0,
        "Gap_time": 0,
        "Gap_distance": safe_underflow(df, "Gap_distance"),
        "max_DRAC": safe_underflow(df, "max_DRAC"),
        "DeltaV": safe_underflow(df, "DeltaV")
    }
    
    overflow_bins = {
        "ttc": 3,
        "mttc_ttc": 3,
        "ttc_deltav": 12,
        "total_conflict_duration_sec": 3,
        "pet": 4,
        "Gap_time": 4,
        "Gap_distance": safe_overflow(df, "Gap_distance"),
        "max_DRAC": safe_overflow(df, "max_DRAC"),
        "DeltaV": safe_overflow(df, "DeltaV")
    }

    def plot_histogram_bins(df_in, column, title):
        if column not in df_in.columns or df_in[column].dropna().empty:
            st.info(f"No {title} values available")
            return

        df_plot = df_in[[column]].dropna().copy()
        underflow_min = underflow_bins.get(column, df_plot[column].min())
        overflow_max = overflow_bins.get(column, df_plot[column].max())
        df_plot[column + "_clipped"] = df_plot[column].clip(lower=underflow_min, upper=overflow_max)
        data_range = overflow_max - underflow_min
        if data_range <= 0:
            st.info("Not enough range to create bins")
            return
        base_bin_width = bin_widths.get(column, 0.5)

        # ---------- BIN COUNT CONTROL ----------
        if column in BIN_COUNT_CONTROL_VARS:
            # Floor the start to nearest whole number or multiple of bin width
            start = np.floor(underflow_min)

            # Total data range from rounded start to overflow max
            total_range = overflow_max - start

            # Decide bin width so number of bins is within MIN_BINS / MAX_BINS
            bin_width = total_range / MAX_BINS  # default
            n_bins = int(np.ceil(total_range / bin_width))
            if n_bins < MIN_BINS:
                bin_width = total_range / MIN_BINS
            elif n_bins > MAX_BINS:
                bin_width = total_range / MAX_BINS

            # Build bins
            bins = [start + i * bin_width for i in range(int(np.ceil(total_range / bin_width)) + 1)]
            bins.append(np.inf)

            # Labels
            labels = [
                f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 2)
            ] + [f">{overflow_max:.1f}"]

        else:
            bin_width = base_bin_width
            bins = [underflow_min]
            bins += list(np.arange(underflow_min + bin_width, overflow_max, bin_width))
            bins.append(overflow_max)
            bins.append(np.inf)
            labels = ([f"≤{underflow_min:.2f}"] +
                      [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(1, len(bins)-2)] +
                      [f">{overflow_max:.2f}"])

        df_plot["Bin"] = pd.cut(df_plot[column + "_clipped"], bins=bins, labels=labels, include_lowest=True, right=True)
        bin_counts = df_plot["Bin"].value_counts().sort_index().reset_index()
        bin_counts.columns = ["Bin_label", "Frequency"]

        fig = px.bar(bin_counts, x="Bin_label", y="Frequency", title=f"{title}")
        fig.update_layout(xaxis_title=title, yaxis_title="Frequency")
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
    zoom = st.slider("Select Heatmap Zoom Level", 12, 22, 19)
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
# =============================

# =============================
# 2️⃣ UPLOAD TRAFFIC VOLUME DATA  (FULL, UPDATED)
# =============================
st.subheader("📂 Upload Traffic Volume Excel File")
uploaded_volume_file = st.file_uploader(
    "Upload Traffic Volume Excel File",
    type=["xlsx"],
    key="volume_uploader"
)

# Exact sheet-name → category mapping (as you told)
SHEET_TO_CATEGORY = {
    "Slip-lane Vehicles": "Slip-lane vehicles",
    "Merging Vehicles": "Merging vehicles",
    "Pedestrians": "VRU"
}

def sum_data_block(df: pd.DataFrame, start_row_idx=1, start_col_idx=3) -> float:
    """
    Sum all numeric values in the rectangular block:
      rows >= start_row_idx  (2nd row)
      cols >= start_col_idx  (4th column)
    Non-numeric -> NaN -> ignored.
    """
    if df is None or df.empty:
        return 0.0
    if df.shape[0] <= start_row_idx or df.shape[1] <= start_col_idx:
        return 0.0

    block = df.iloc[start_row_idx:, start_col_idx:].copy()
    block = block.apply(pd.to_numeric, errors="coerce")
    total = float(block.sum(skipna=True).sum(skipna=True))
    return 0.0 if np.isnan(total) else total

if uploaded_volume_file:
    xls = pd.ExcelFile(uploaded_volume_file)

    # Totals across sheets for final pie
    totals = {"Slip-lane vehicles": 0.0, "Merging vehicles": 0.0, "VRU": 0.0}

    for sheet in xls.sheet_names:
        st.markdown(f"### Traffic Volume - {sheet}")
        df_vol = pd.read_excel(uploaded_volume_file, sheet_name=sheet)

        # -------------------------------------------------------
        # ✅ PIE TOTAL for this sheet (your rule: 2nd row + 4th col)
        # -------------------------------------------------------
        sheet_total_for_pie = sum_data_block(df_vol, start_row_idx=1, start_col_idx=3)

        if sheet in SHEET_TO_CATEGORY:
            totals[SHEET_TO_CATEGORY[sheet]] += sheet_total_for_pie
        else:
            st.warning(
                f"Sheet '{sheet}' not recognised for pie totals. "
                f"Expected: {list(SHEET_TO_CATEGORY.keys())}"
            )

        with st.expander("✅ Total used for final pie (this sheet)"):
            st.write("Sheet:", sheet)
            st.write("Mapped category:", SHEET_TO_CATEGORY.get(sheet, "Not used"))
            st.write("Summation rule: rows ≥ 2nd row, cols ≥ 4th column (numeric only)")
            st.write("Sheet total:", sheet_total_for_pie)

        # -------------------------------------------------------
        # DAILY/HOURLY CHARTS (same as your original, but fixed)
        # -------------------------------------------------------
        needed_cols = {"Date", "IntervalStart", "IntervalEnd"}
        if not needed_cols.issubset(df_vol.columns):
            st.info("This sheet doesn’t have Date/IntervalStart/IntervalEnd — skipping daily/hourly charts.")
            continue

        df_vol["Date"] = pd.to_datetime(df_vol["Date"], errors="coerce")
        df_vol["Day_only"] = df_vol["Date"].dt.date

        # Total Volume per row from numeric columns (starting from 4th column)
        candidate_cols = list(df_vol.columns[3:])
        numeric_cols = df_vol[candidate_cols].select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) == 0:
            st.info("No numeric volume columns found from 4th column onward — skipping charts.")
            continue

        df_vol["Total Volume"] = df_vol[numeric_cols].sum(axis=1)

        # Convert interval start/end to hour
        start_h = pd.to_datetime(df_vol["IntervalStart"], errors="coerce").dt.hour
        end_h   = pd.to_datetime(df_vol["IntervalEnd"], errors="coerce").dt.hour

        # Mid-hour may become .5, so DO NOT astype Int64 directly.
        # Floor keeps 7:00–8:00 → 7 (matches your hour-bin labels)
        mid_h = (start_h + end_h) / 2
        df_vol["Hour"] = np.floor(mid_h).astype("Int64")

        # Filter 5AM–7PM
        df_hour = df_vol[(df_vol["Hour"] >= 5) & (df_vol["Hour"] <= 19)]
        hour_bins = list(range(5, 20))

        hourly_volume = (
            df_hour.groupby("Hour")["Total Volume"].sum()
            .reindex(hour_bins, fill_value=0).reset_index()
        )
        hourly_volume.columns = ["Hour", "Total Volume"]
        hourly_volume["Hour Interval"] = [f"{h}:00 - {h+1}:00" for h in hourly_volume["Hour"]]

        cols_vol = st.columns(2)

        with cols_vol[0]:
            daily_volume = df_vol.groupby("Day_only")["Total Volume"].sum().reset_index()
            daily_volume["Trend"] = daily_volume["Total Volume"].rolling(window=3, min_periods=1).mean()
            daily_volume["Day_Label"] = pd.to_datetime(daily_volume["Day_only"]).dt.strftime("%d-%b (%a)")

            fig_daily = px.bar(
                daily_volume,
                x="Day_Label",
                y="Total Volume",
                width=900,
                height=500
            )
            fig_daily.add_scatter(
                x=daily_volume["Day_Label"],
                y=daily_volume["Trend"],
                mode="lines",
                name="Trend",
                line=dict(color="orange", width=3)
            )
            fig_daily.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_daily, use_container_width=False)

        with cols_vol[1]:
            hourly_volume["Trend"] = hourly_volume["Total Volume"].rolling(window=2, min_periods=1).mean()
            fig_hourly = px.bar(
                hourly_volume,
                x="Hour Interval",
                y="Total Volume",
                width=900,
                height=500
            )
            fig_hourly.add_scatter(
                x=hourly_volume["Hour Interval"],
                y=hourly_volume["Trend"],
                mode="lines",
                name="Trend",
                line=dict(color="orange", width=3)
            )
            st.plotly_chart(fig_hourly, use_container_width=False)

    # -------------------------------------------------------
    # ✅ FINAL PIE CHART (end)
    # -------------------------------------------------------
    st.subheader("🥧 Total Volume Composition (Slip-lane vs Merging vs VRU)")

    pie_df = pd.DataFrame({
        "Category": ["Slip-lane vehicles", "Merging vehicles", "VRU"],
        "Total Volume": [totals["Slip-lane vehicles"], totals["Merging vehicles"], totals["VRU"]]
    })

    if pie_df["Total Volume"].sum() == 0:
        st.warning(
            "Pie totals are zero — check that your numeric data really starts at 2nd row and 4th column "
            "and that the sheet names match exactly."
        )
        st.dataframe(pie_df)
    else:
        fig_pie_vol = px.pie(
            pie_df,
            names="Category",
            values="Total Volume",
            hole=0.25
        )
        fig_pie_vol.update_traces(
            texttemplate="%{label}<br>%{value:.0f} (%{percent})",
            textposition="inside"
        )
        st.plotly_chart(fig_pie_vol, use_container_width=True)
        st.dataframe(pie_df)
        # ✅ Save volume totals for later rate calculation
        st.session_state["volume_totals"] = totals.copy()

# =============================
# 3️⃣ CONFLICT RATES (CONFLICTS PER VOLUME)
# =============================
st.subheader("📈 Conflict Rates by Category (Conflicts per Volume %)")

if "conflict_df" not in st.session_state:
    st.info("Upload the Conflict dataset to compute conflict rates.")
elif "volume_totals" not in st.session_state:
    st.info("Upload the Traffic Volume Excel file to compute conflict rates.")
else:
    df_conf = st.session_state["conflict_df"].copy()
    totals = st.session_state["volume_totals"].copy()

    # --- conflict counts ---
    # Ensure VRU conflicts are counted correctly:
    # Your encounter grouping earlier keeps "VRU" as "VRU" (not "Vehicle-VRU").
    # We'll compute counts robustly.
    conflict_counts = df_conf["Encounter_grouped"].value_counts()

    n_vru      = int(conflict_counts.get("VRU", 0))
    n_rearend  = int(conflict_counts.get("Rear-End", 0))
    n_merging  = int(conflict_counts.get("Merging", 0))

    # --- exposure (volume) ---
    vol_vru     = float(totals.get("VRU", 0.0))                 # Pedestrians sheet
    vol_rearend = float(totals.get("Slip-lane vehicles", 0.0))  # Slip-lane Vehicles sheet
    vol_merging = float(totals.get("Merging vehicles", 0.0))    # Merging Vehicles sheet

    def safe_rate(n, v):
        return np.nan if (v is None or v == 0 or np.isnan(v)) else (n / v) * 100

    rate_vru     = safe_rate(n_vru, vol_vru)
    rate_rearend = safe_rate(n_rearend, vol_rearend)
    rate_merging = safe_rate(n_merging, vol_merging)

    rates_df = pd.DataFrame([
        {"Conflict Type": "VRU",      "Conflicts": n_vru,     "Volume (Exposure)": vol_vru,     "Conflict Rate (%)": rate_vru},
        {"Conflict Type": "Rear-End", "Conflicts": n_rearend, "Volume (Exposure)": vol_rearend, "Conflict Rate (%)": rate_rearend},
        {"Conflict Type": "Merging",  "Conflicts": n_merging, "Volume (Exposure)": vol_merging, "Conflict Rate (%)": rate_merging},
    ])

    # Nice formatting for display
    show_df = rates_df.copy()
    show_df["Volume (Exposure)"] = show_df["Volume (Exposure)"].map(lambda x: f"{x:,.0f}")
    show_df["Conflict Rate (%)"] = show_df["Conflict Rate (%)"].map(lambda x: "NA (0 volume)" if pd.isna(x) else f"{x:.4f}%")

    st.dataframe(show_df, use_container_width=True)

    # Bar chart for rates (numeric)
    plot_df = rates_df.dropna(subset=["Conflict Rate (%)"]).copy()
    if plot_df.empty:
        st.warning("Cannot plot rates because one or more exposure volumes are zero.")
    else:
        fig_rate = px.bar(
            plot_df,
            x="Conflict Type",
            y="Conflict Rate (%)",
            text=plot_df["Conflict Rate (%)"].map(lambda x: f"{x:.4f}%"),
            height=450
        )
        fig_rate.update_traces(textposition="outside")
        fig_rate.update_layout(yaxis_title="Conflict Rate (%)", xaxis_title="")
        st.plotly_chart(fig_rate, use_container_width=True)

    st.caption(
        "Rate definition: (Conflict count / corresponding volume) × 100. "
        "VRU uses Pedestrian volume; Rear-End uses Slip-lane vehicle volume; "
        "Merging uses Merging vehicle volume."
    )
















