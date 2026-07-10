import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")


# Data loading

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df


df = load_data()

st.sidebar.title("Sales Analytics")
page = st.sidebar.radio(
    "Navigate",
    [
        "1. Sales Overview",
        "2. Forecast Explorer",
        "3. Anomaly Report",
        "4. Product Demand Segments",
    ],
)


# Page 1 — Sales Overview Dashboard


if page == "1. Sales Overview":
    st.title("Sales Overview Dashboard")

    yearly_sales = df.groupby("Year")["Sales"].sum().reset_index()
    fig1 = px.bar(
        yearly_sales, x="Year", y="Sales",
        title="Total Sales by Year", text_auto=".2s",
    )
    fig1.update_xaxes(type="category")
    st.plotly_chart(fig1, use_container_width=True)

    monthly_sales = (
        df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
    )
    fig2 = px.line(
        monthly_sales, x="Order Date", y="Sales", markers=True,
        title="Monthly Sales Trend",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sales by Region and Category")
    col1, col2 = st.columns(2)
    with col1:
        regions = st.multiselect(
            "Region", sorted(df["Region"].unique()),
            default=sorted(df["Region"].unique()),
        )
    with col2:
        categories = st.multiselect(
            "Category", sorted(df["Category"].unique()),
            default=sorted(df["Category"].unique()),
        )

    filtered = df[df["Region"].isin(regions) & df["Category"].isin(categories)]
    grouped = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig3 = px.bar(
        grouped, x="Region", y="Sales", color="Category", barmode="group",
        title="Sales by Region and Category",
    )
    st.plotly_chart(fig3, use_container_width=True)


# Page 2 — Forecast Explorer (Prophet — best model per Task 3)

elif page == "2. Forecast Explorer":
    st.title("Forecast Explorer")

    dim = st.selectbox("Select dimension", ["Category", "Region"])
    value = st.selectbox(f"Select {dim}", sorted(df[dim].unique()))
    horizon = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    @st.cache_resource(show_spinner="Training Prophet model...")
    def train_prophet_model(dim, value):
        seg = df[df[dim] == value]
        monthly = (
            seg.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
            .sum()
            .reset_index()
            .rename(columns={"Order Date": "ds", "Sales": "y"})
        )
        model = Prophet()
        model.fit(monthly)
        future = model.make_future_dataframe(periods=3, freq="MS")
        forecast = model.predict(future)
        return monthly, forecast

    monthly, forecast = train_prophet_model(dim, value)
    forecast_future = forecast.tail(3).head(horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["ds"], y=monthly["y"], name="Actual", mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_future["ds"], y=forecast_future["yhat_upper"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_future["ds"], y=forecast_future["yhat_lower"],
        fill="tonexty", fillcolor="rgba(255,0,0,0.15)", line=dict(width=0),
        name="Confidence interval",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_future["ds"], y=forecast_future["yhat"], name="Forecast",
        mode="lines+markers", line=dict(color="red"),
    ))
    fig.update_layout(
        title=f"{horizon}-Month Sales Forecast — {value} ({dim})",
        xaxis_title="Date", yaxis_title="Sales",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .rename(columns={
            "ds": "Date", "yhat": "Predicted Sales",
            "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound",
        })
        .reset_index(drop=True)
    )

    # MAE / RMSE — same methodology as Task 3: fit on last 3 known months
    history_fit = forecast.iloc[:len(monthly)]
    actual_tail = monthly["y"].tail(3)
    pred_tail = history_fit["yhat"].tail(3)
    mae = mean_absolute_error(actual_tail, pred_tail)
    rmse = np.sqrt(mean_squared_error(actual_tail, pred_tail))

    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mae:,.2f}")
    c2.metric("RMSE", f"{rmse:,.2f}")
    st.caption(
        "Model: Facebook Prophet (best performing model from Task 3 evaluation). "
        "MAE/RMSE computed on the last 3 months of historical data for this segment."
    )

# Page 3 — Anomaly Report

elif page == "3. Anomaly Report":
    st.title("Anomaly Report")

    @st.cache_data
    def get_anomalies():
        weekly_sales = (
            df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
            .sum()
            .reset_index()
        )
        model = IsolationForest(contamination=0.05, random_state=42)
        weekly_sales["iforest"] = model.fit_predict(weekly_sales[["Sales"]])
        weekly_sales["iforest_anomaly"] = weekly_sales["iforest"] == -1

        window = 8
        weekly_sales["rolling_mean"] = weekly_sales["Sales"].rolling(window).mean()
        weekly_sales["rolling_std"] = weekly_sales["Sales"].rolling(window).std()
        weekly_sales["z_score"] = (
            weekly_sales["Sales"] - weekly_sales["rolling_mean"]
        ) / weekly_sales["rolling_std"]
        weekly_sales["zscore_anomaly"] = weekly_sales["z_score"].abs() > 2
        return weekly_sales

    weekly_sales = get_anomalies()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly_sales["Order Date"], y=weekly_sales["Sales"],
        name="Weekly Sales", mode="lines",
    ))
    iforest_pts = weekly_sales[weekly_sales["iforest_anomaly"]]
    fig.add_trace(go.Scatter(
        x=iforest_pts["Order Date"], y=iforest_pts["Sales"], mode="markers",
        name="Isolation Forest", marker=dict(color="red", size=10),
    ))
    z_pts = weekly_sales[weekly_sales["zscore_anomaly"]]
    fig.add_trace(go.Scatter(
        x=z_pts["Order Date"], y=z_pts["Sales"], mode="markers",
        name="Z-Score", marker=dict(color="green", size=11, symbol="x"),
    ))
    fig.update_layout(
        title="Weekly Sales with Detected Anomalies",
        xaxis_title="Date", yaxis_title="Sales",
    )
    st.plotly_chart(fig, use_container_width=True)

    anomalies = weekly_sales[
        weekly_sales["iforest_anomaly"] | weekly_sales["zscore_anomaly"]
    ].copy()
    anomalies["Detected By"] = anomalies.apply(
        lambda r: ", ".join(filter(None, [
            "Isolation Forest" if r["iforest_anomaly"] else "",
            "Z-Score" if r["zscore_anomaly"] else "",
        ])),
        axis=1,
    )

    st.subheader("Detected Anomaly Dates")
    st.dataframe(
        anomalies[["Order Date", "Sales", "Detected By"]]
        .rename(columns={"Order Date": "Week"})
        .reset_index(drop=True)
    )
    st.caption(
        f"Isolation Forest flagged {int(weekly_sales['iforest_anomaly'].sum())} weeks; "
        f"Z-Score flagged {int(weekly_sales['zscore_anomaly'].sum())} weeks."
    )

# Page 4 — Product Demand Segments

elif page == "4. Product Demand Segments":
    st.title("Product Demand Segments")

    @st.cache_data
    def get_clusters():
        d = df.copy()
        d["year"] = d["Order Date"].dt.year
        d["month"] = d["Order Date"].dt.to_period("M")

        monthly_sales = (
            d.groupby(["Sub-Category", "month"])["Sales"].sum().reset_index()
        )
        yearly_sales = (
            d.groupby(["Sub-Category", "year"])["Sales"].sum().reset_index()
        )
        total_sales = d.groupby("Sub-Category")["Sales"].sum()
        avg_order = d.groupby("Sub-Category")["Sales"].mean()
        volatility = monthly_sales.groupby("Sub-Category")["Sales"].std()

        growth = []
        for sub in yearly_sales["Sub-Category"].unique():
            temp = yearly_sales[yearly_sales["Sub-Category"] == sub].sort_values("year")
            rate = temp["Sales"].pct_change().mean() * 100 if len(temp) > 1 else 0
            growth.append([sub, rate])
        growth = pd.DataFrame(growth, columns=["Sub-Category", "Growth"])

        features = pd.DataFrame({
            "Total Sales": total_sales,
            "Volatility": volatility,
            "Average Order Value": avg_order,
        }).reset_index()
        features = features.merge(growth, on="Sub-Category").fillna(0)

        scaler = StandardScaler()
        x = scaler.fit_transform(
            features[["Total Sales", "Growth", "Volatility", "Average Order Value"]]
        )

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        features["Cluster"] = kmeans.fit_predict(x)

        pca = PCA(n_components=2)
        points = pca.fit_transform(x)
        features["PC1"] = points[:, 0]
        features["PC2"] = points[:, 1]
        return features

    def label_clusters(features):
        stats = features.groupby("Cluster")[
            ["Total Sales", "Growth", "Volatility", "Average Order Value"]
        ].mean()
        growth_c = stats["Growth"].idxmax()
        value_c = stats["Average Order Value"].idxmax()
        volume_c = stats["Total Sales"].idxmax()
        labels = {}
        for c in stats.index:
            if c == growth_c:
                labels[c] = "Growing Demand"
            elif c == value_c:
                labels[c] = "High Value Products"
            elif c == volume_c:
                labels[c] = "High Volume, Stable Demand"
            else:
                labels[c] = "Low Volume, Stable Demand"
        return labels

    features = get_clusters()
    labels = label_clusters(features)
    features["Segment"] = features["Cluster"].map(labels)

    fig = px.scatter(
        features, x="PC1", y="PC2", color="Segment", text="Sub-Category",
        size="Total Sales", size_max=40,
        title="K-Means Clustering of Product Demand (PCA Projection)",
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Category → Demand Cluster")
    st.dataframe(
        features[
            ["Sub-Category", "Segment", "Total Sales", "Growth",
             "Volatility", "Average Order Value"]
        ]
        .sort_values("Segment")
        .reset_index(drop=True)
    )

    with st.expander("Stocking strategy per segment"):
        st.markdown(
            """
| Segment | Recommendation |
|---|---|
| High Value Products | Keep adequate stock but monitor demand carefully due to high-value inventory and higher volatility. |
| Low Volume, Stable Demand | Maintain lower inventory levels and restock as needed to reduce storage costs. |
| High Volume, Stable Demand | Keep higher inventory levels with regular replenishment to avoid stockouts. |
| Growing Demand | Increase inventory gradually and monitor future sales trends to meet rising demand. |
"""
        )
