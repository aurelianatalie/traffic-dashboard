
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os

st.set_page_config(
    page_title="Traffic Dashboard",
    page_icon="🚦",
    layout="wide"
)

#STYLE
st.html("""
<style>
.stApp {
    background-color: #0B1220;
    color: #E8EDF5;
}

[data-testid="stSidebar"] {
    background-color: #111C2E;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #14243A, #192B43);
    border: 1px solid #2A405A;
    padding: 20px;
    border-radius: 14px;
}

[data-testid="stMetricValue"] {
    color: #55DDE0;
}

h1, h2, h3 {
    color: #F5EAF0;
}
</style>
""")

#DATA
@st.cache_data
def load_data():
    data = {}
    filenames = [
        "traffic_overview",
        "traffic_by_segment",
        "ml.comparison",
        "ml_evaluation",
        "graph_nodes",
        "graph_edges",
        "graph_communities"
    ]

    for name in filenames:
        data[name] = pd.read_csv(f"data/{name}.csv")

    return data

data = load_data()

overview = data["traffic_overview"].iloc[0]
traffic = data["traffic_by_segment"]
ml = data["ml.comparison"]
nodes = data["graph_nodes"]
edges = data["graph_edges"]
communities = data["graph_communities"]

#SIDEBAR
st.sidebar.title("🚦 TRAFFIC DASHBOARD")
st.sidebar.caption("Traffic Data Analytics")

page = st.sidebar.selectbox(
    "NAVIGATION",
    [
        "Overview",
        "Traffic Analysis",
        "Machine Learning",
        "Graph Analytics"
    ]
)

st.sidebar.divider()
st.sidebar.caption("TrafficTab23 | Big Data Project")

#OVERVIEW
if page == "Overview":

    st.title("Traffic Dashboard")
    st.caption("Traffic Data Analytics & Intelligence")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{int(overview['total_records']):,}")
    c2.metric("Road Segments", f"{int(overview['total_road_segments']):,}")
    c3.metric("Normal Records", f"{int(overview['normal_records']):,}")
    c4.metric("Anomaly Records", f"{int(overview['anomaly_records']):,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            names=["Normal", "Anomaly"],
            values=[
                overview["normal_records"],
                overview["anomaly_records"]
            ],
            hole=0.6,
            title="Traffic Data Distribution",
            color_discrete_sequence=["#22C55E", "#F97316"]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            x=["Normal", "Anomaly"],
            y=[
                overview["normal_percentage"],
                overview["anomaly_percentage"]
            ],
            title="Traffic Class Percentage",
            labels={"x": "Class", "y": "Percentage (%)"},
            color=["Normal", "Anomaly"],
            color_discrete_map={
                "Normal": "#22C55E",
                "Anomaly": "#F97316"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

#TRAFFIC ANALYSIS
elif page == "Traffic Analysis":

    st.title("Traffic Analysis")
    st.caption("Road Segment Characteristics")

    selected = st.selectbox(
        "Select Road Segment",
        sorted(traffic["road_segment_id"].unique())
    )

    row = traffic[
        traffic["road_segment_id"] == selected
    ].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Speed", f"{row['avg_speed']:.2f}")
    c2.metric("Average Vehicle Count", f"{row['avg_vehicle_count']:.2f}")
    c3.metric("Anomaly Rate", f"{row['anomaly_rate']*100:.2f}%")

    fig = px.bar(
        traffic.sort_values("avg_vehicle_count", ascending=False),
        x="road_segment_id",
        y="avg_vehicle_count",
        title="Average Vehicle Count by Road Segment"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        traffic,
        x="avg_speed",
        y="avg_vehicle_count",
        color="anomaly_rate",
        size="avg_lane_occupancy",
        hover_name="road_segment_id",
        title="Traffic Speed vs Vehicle Count",
        color_continuous_scale="Turbo"
    )
    st.plotly_chart(fig, use_container_width=True)

#MACHINE LEARNING
elif page == "Machine Learning":

    st.title("Machine Learning Evaluation")
    st.caption("Comparison of Four Classification Models")

    metric_options = {
        "Accuracy": "accuracy",
        "Weighted Precision": "weighted_precision",
        "Weighted Recall": "weighted_recall",
        "Weighted F1-score": "weighted_f1",
        "AUC-ROC": "auc_roc",
        "Anomaly Recall": "anomaly_recall"
    }

    selected_metric = st.selectbox(
        "Select Evaluation Metric",
        list(metric_options.keys())
    )

    metric = metric_options[selected_metric]

    fig = px.bar(
        ml,
        x="model",
        y=metric,
        color="framework",
        barmode="group",
        title=f"Model Comparison: {selected_metric}",
        text_auto=".4f"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ml, use_container_width=True)

    st.warning(
        "All four models produced zero recall for the anomaly class. "
        "Accuracy alone is therefore insufficient to assess model performance."
    )

#GRAPH ANALYTICS
elif page == "Graph Analytics":

    st.title("Graph Analytics")
    st.caption("Road Segment Similarity Network")

    G = nx.Graph()

    for _, row in nodes.iterrows():
        G.add_node(int(row["road_segment_id"]))

    for _, row in edges.iterrows():
        G.add_edge(
            int(row["source"]),
            int(row["target"]),
            weight=float(row["weight"])
        )

    community_map = communities.set_index(
        "road_segment_id"
    )["community"].to_dict()

    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []

    for source, target in G.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="#64748B"),
        hoverinfo="none"
    )

    node_x, node_y, node_colors, node_text = [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(community_map.get(node, -1))
        node_text.append(f"Road Segment {node}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[str(n) for n in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=13,
            color=node_colors,
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(title="Community")
        )
    )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title="Traffic Road Segment Similarity Graph",
        height=650,
        showlegend=False,
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        font_color="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", G.number_of_nodes())
    c2.metric("Edges", G.number_of_edges())
    c3.metric("Communities", communities["community"].nunique())

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Edges represent similarity between traffic characteristics, "
        "not physical road connectivity."
    )
