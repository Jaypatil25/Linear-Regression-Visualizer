import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Linear Regression Explorer", layout="wide", page_icon=":material/show_chart:")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 8px 16px;
        color: #a0aec0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: white !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a1f35, #1e2540);
        border-left: 4px solid #4f46e5;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9em;
        color: #a0aec0;
    }
</style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def generate_data(n=80, noise=0.5, true_m=2.0, true_b=1.0, n_outliers=0, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, n)
    y = true_m * X + true_b + rng.normal(0, noise * 3, n)
    if n_outliers > 0:
        idx = rng.choice(n, n_outliers, replace=False)
        y[idx] += rng.choice([-1, 1], n_outliers) * rng.uniform(8, 15, n_outliers)
    return X, y

def mse(X, y, m, b):
    return float(np.mean((y - (m * X + b)) ** 2))

def gradient_descent(X, y, lr, n_iter):
    m, b = 0.0, 0.0
    history = [(m, b, mse(X, y, m, b))]
    for _ in range(n_iter):
        pred = m * X + b
        err = pred - y
        dm = (2 / len(X)) * np.dot(err, X)
        db = (2 / len(X)) * np.sum(err)
        m -= lr * dm
        b -= lr * db
        history.append((m, b, mse(X, y, m, b)))
    return history

COLORS = {"scatter": "#818cf8", "line": "#f472b6", "residual": "#fb923c",
          "gd": "#34d399", "surface": "Viridis", "bg": "#0e1117", "grid": "#1e2130"}

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<h2><i class="bi bi-sliders"></i> Controls</h2>', unsafe_allow_html=True)
    st.markdown("---")
    dataset = st.selectbox("Dataset", ["Clean Linear", "Noisy", "Custom"])
    noise_level = st.slider("Noise Level", 0.1, 3.0, 0.8, 0.1)
    add_outliers = st.checkbox("Add Outliers", False)
    n_outliers = st.slider("# Outliers", 1, 15, 5) if add_outliers else 0
    st.markdown("---")
    st.markdown("### Manual Line")
    slope = st.slider("Slope (m)", -5.0, 5.0, 2.0, 0.1)
    intercept = st.slider("Intercept (b)", -10.0, 10.0, 1.0, 0.1)
    st.markdown("---")
    st.markdown("### Gradient Descent")
    lr = st.select_slider("Learning Rate (α)", [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5], value=0.01)
    n_iter = st.slider("Iterations", 10, 500, 100, 10)

noise_map = {"Clean Linear": 0.2, "Noisy": noise_level, "Custom": noise_level}
X, y = generate_data(noise=noise_map[dataset], n_outliers=n_outliers)
X_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 200)

# ── header ────────────────────────────────────────────────────────────────────

st.markdown('<h1><i class="bi bi-graph-up-arrow"></i> Linear Regression Explorer</h1>', unsafe_allow_html=True)
st.markdown("*Interactively visualize how a model learns — from raw data to gradient descent convergence.*")
st.markdown("---")

# ── tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data & Line Fit",
    "Error & Residuals",
    "Loss Landscape",
    "Gradient Descent",
    "Learning Rate",
    "Noise & Outliers",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data & Line Fit
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_info, col_metric = st.columns([3, 1])
    current_mse = mse(X, y, slope, intercept)

    with col_metric:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.8em;color:#a0aec0;">Current MSE</div>
            <div style="font-size:2em;font-weight:700;color:#f472b6;">{current_mse:.2f}</div>
        </div>
        <br>
        <div class="metric-card">
            <div style="font-size:0.8em;color:#a0aec0;">Line Equation</div>
            <div style="font-size:1.1em;font-weight:600;color:#818cf8;">y = {slope:.2f}x + {intercept:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode="markers",
        marker=dict(color=COLORS["scatter"], size=7, opacity=0.8, line=dict(color="white", width=0.5)),
        name="Data Points"))
    fig.add_trace(go.Scatter(x=X_line, y=slope * X_line + intercept,
        line=dict(color=COLORS["line"], width=3), name=f"y = {slope:.2f}x + {intercept:.2f}"))

    # OLS line
    m_ols = np.cov(X, y)[0, 1] / np.var(X)
    b_ols = np.mean(y) - m_ols * np.mean(X)
    fig.add_trace(go.Scatter(x=X_line, y=m_ols * X_line + b_ols,
        line=dict(color="#34d399", width=2, dash="dash"), name="OLS Best Fit", opacity=0.6))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["grid"],
        height=420, legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        xaxis_title="X", yaxis_title="y",
        title=dict(text="Scatter Plot with Adjustable Regression Line", font=dict(size=16))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box">
    <i class="bi bi-search"></i> <b>What to observe:</b> Drag the slope and intercept sliders in the sidebar.
    The <span style="color:#f472b6">pink line</span> is your manual fit; the
    <span style="color:#34d399">green dashed line</span> is the mathematically optimal OLS solution.
    Try to match them by minimizing the MSE shown above!
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Error & Residuals
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    y_pred = slope * X + intercept
    residuals = y - y_pred
    sq_errors = residuals ** 2

    c1, c2, c3 = st.columns(3)
    c1.metric("MSE", f"{np.mean(sq_errors):.3f}")
    c2.metric("RMSE", f"{np.sqrt(np.mean(sq_errors)):.3f}")
    c3.metric("MAE", f"{np.mean(np.abs(residuals)):.3f}")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Residuals on Scatter", "Squared Error per Point"))

    # Left: scatter + residual lines
    fig.add_trace(go.Scatter(x=X, y=y, mode="markers",
        marker=dict(color=COLORS["scatter"], size=7), name="Data", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_line, y=slope * X_line + intercept,
        line=dict(color=COLORS["line"], width=2), name="Fit Line", showlegend=False), row=1, col=1)

    for xi, yi, ypi in zip(X, y, y_pred):
        color = "#ef4444" if abs(yi - ypi) > np.std(residuals) * 1.5 else COLORS["residual"]
        fig.add_trace(go.Scatter(x=[xi, xi], y=[yi, ypi],
            line=dict(color=color, width=1.5), mode="lines", showlegend=False), row=1, col=1)

    # Right: bar chart of squared errors
    sorted_idx = np.argsort(X)
    bar_colors = ["#ef4444" if e > np.percentile(sq_errors, 80) else "#fb923c" for e in sq_errors[sorted_idx]]
    fig.add_trace(go.Bar(x=X[sorted_idx], y=sq_errors[sorted_idx],
        marker_color=bar_colors, name="Squared Error", showlegend=False), row=1, col=2)

    fig.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["grid"], height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box">
    <i class="bi bi-search"></i> <b>What to observe:</b> Each vertical line is a residual (prediction error).
    <span style="color:#ef4444">Red lines</span> are large errors (outliers).
    The right chart shows squared errors — squaring penalizes large mistakes heavily,
    which is why Linear Regression is sensitive to outliers.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Loss Landscape
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    m_range = np.linspace(-1, 5, 60)
    b_range = np.linspace(-5, 10, 60)
    MM, BB = np.meshgrid(m_range, b_range)
    ZZ = np.array([[mse(X, y, mm, bb) for mm in m_range] for bb in b_range])

    view = st.radio("View", ["3D Surface", "Contour Map"], horizontal=True)

    if view == "3D Surface":
        fig = go.Figure(go.Surface(x=MM, y=BB, z=ZZ, colorscale=COLORS["surface"],
            opacity=0.85, showscale=True))
        fig.add_trace(go.Scatter3d(x=[slope], y=[intercept], z=[mse(X, y, slope, intercept)],
            mode="markers", marker=dict(color="#f472b6", size=8, symbol="diamond"),
            name="Current Position"))
        fig.update_layout(scene=dict(
            xaxis_title="Slope (m)", yaxis_title="Intercept (b)", zaxis_title="MSE Loss",
            bgcolor=COLORS["bg"]),
            template="plotly_dark", paper_bgcolor=COLORS["bg"], height=500,
            title="Loss Surface J(m, b)")
    else:
        fig = go.Figure(go.Contour(x=m_range, y=b_range, z=ZZ,
            colorscale=COLORS["surface"], ncontours=25,
            contours=dict(showlabels=True, labelfont=dict(size=9))))
        fig.add_trace(go.Scatter(x=[slope], y=[intercept], mode="markers",
            marker=dict(color="#f472b6", size=12, symbol="star"),
            name="Current Position"))
        m_ols = np.cov(X, y)[0, 1] / np.var(X)
        b_ols = np.mean(y) - m_ols * np.mean(X)
        fig.add_trace(go.Scatter(x=[m_ols], y=[b_ols], mode="markers",
            marker=dict(color="#34d399", size=12, symbol="star"),
            name="Global Minimum (OLS)"))
        fig.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
            plot_bgcolor=COLORS["grid"], height=500,
            xaxis_title="Slope (m)", yaxis_title="Intercept (b)",
            title="Loss Contour Map — Top-Down View")

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class="insight-box">
    <i class="bi bi-search"></i> <b>What to observe:</b> The loss surface is a bowl (convex) — there's one global minimum.
    The <span style="color:#f472b6">pink star</span> is your current (m, b) position.
    The <span style="color:#34d399">green star</span> is the true minimum.
    Gradient descent rolls down this bowl toward the bottom.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Gradient Descent
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    history = gradient_descent(X, y, lr, n_iter)
    ms = [h[0] for h in history]
    bs = [h[1] for h in history]
    losses = [h[2] for h in history]

    step = st.slider("Animation Step", 0, len(history) - 1, len(history) - 1)

    col_a, col_b = st.columns(2)

    with col_a:
        # Contour + path
        m_range = np.linspace(min(ms) - 1, max(ms) + 1, 50)
        b_range = np.linspace(min(bs) - 1, max(bs) + 1, 50)
        MM, BB = np.meshgrid(m_range, b_range)
        ZZ = np.array([[mse(X, y, mm, bb) for mm in m_range] for bb in b_range])

        fig = go.Figure(go.Contour(x=m_range, y=b_range, z=ZZ,
            colorscale="Viridis", ncontours=20, showscale=False))
        fig.add_trace(go.Scatter(x=ms[:step+1], y=bs[:step+1],
            mode="lines+markers", line=dict(color=COLORS["gd"], width=2),
            marker=dict(size=4), name="GD Path"))
        fig.add_trace(go.Scatter(x=[ms[step]], y=[bs[step]], mode="markers",
            marker=dict(color="#f472b6", size=12, symbol="star"), name="Current"))
        fig.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
            plot_bgcolor=COLORS["grid"], height=350,
            xaxis_title="m", yaxis_title="b", title="Gradient Descent Path")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Data + current line
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=X, y=y, mode="markers",
            marker=dict(color=COLORS["scatter"], size=6), name="Data"))
        fig2.add_trace(go.Scatter(x=X_line, y=ms[step] * X_line + bs[step],
            line=dict(color=COLORS["gd"], width=3),
            name=f"Step {step}: y={ms[step]:.2f}x+{bs[step]:.2f}"))
        fig2.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
            plot_bgcolor=COLORS["grid"], height=350,
            xaxis_title="X", yaxis_title="y", title="Line at Current Step")
        st.plotly_chart(fig2, use_container_width=True)

    # Loss curve
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=losses, mode="lines",
        line=dict(color=COLORS["gd"], width=2), name="Loss"))
    fig3.add_trace(go.Scatter(x=[step], y=[losses[step]], mode="markers",
        marker=dict(color="#f472b6", size=10), name="Current Step"))
    fig3.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["grid"], height=220,
        xaxis_title="Iteration", yaxis_title="MSE Loss", title="Loss vs Iterations")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f"""<div class="insight-box">
    <i class="bi bi-search"></i> <b>Step {step}/{n_iter}:</b> m = {ms[step]:.4f}, b = {bs[step]:.4f}, MSE = {losses[step]:.4f}<br>
    Drag the slider to replay the learning process step by step.
    Each step moves parameters in the direction that reduces loss the most.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Learning Rate Experiments
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    lr_configs = {
        "Too Small (0.0001)": 0.0001,
        "Small (0.001)": 0.001,
        "Optimal (0.01)": 0.01,
        "Large (0.1)": 0.1,
        "Too Large (0.5)": 0.5,
    }

    fig = go.Figure()
    palette = ["#818cf8", "#34d399", "#f472b6", "#fb923c", "#ef4444"]

    for (label, lr_val), color in zip(lr_configs.items(), palette):
        hist = gradient_descent(X, y, lr_val, 200)
        loss_curve = [h[2] for h in hist]
        # clip for display
        loss_curve = [min(l, 500) for l in loss_curve]
        fig.add_trace(go.Scatter(y=loss_curve, mode="lines",
            line=dict(color=color, width=2), name=label))

    fig.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["grid"], height=420,
        xaxis_title="Iteration", yaxis_title="MSE Loss (capped at 500)",
        title="Loss Curves for Different Learning Rates",
        legend=dict(bgcolor="rgba(0,0,0,0.4)"))
    st.plotly_chart(fig, use_container_width=True)

    # Final parameter comparison
    rows = []
    for label, lr_val in lr_configs.items():
        hist = gradient_descent(X, y, lr_val, 200)
        rows.append({"Learning Rate": label, "Final MSE": f"{hist[-1][2]:.4f}",
                     "Final m": f"{hist[-1][0]:.4f}", "Final b": f"{hist[-1][1]:.4f}"})

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("""<div class="insight-box">
    <i class="bi bi-search"></i> <b>What to observe:</b>
    <span style="color:#ef4444">Too large</span> → loss explodes (divergence).
    <span style="color:#818cf8">Too small</span> → converges very slowly.
    <span style="color:#f472b6">Optimal</span> → fast, smooth convergence.
    The sweet spot balances speed and stability.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Noise & Outliers
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    col_n, col_o = st.columns(2)
    with col_n:
        noise_compare = st.slider("Noise Level to Compare", 0.1, 3.0, 1.5, 0.1)
    with col_o:
        outlier_compare = st.slider("# Outliers to Inject", 0, 20, 8)

    X_clean, y_clean = generate_data(noise=0.2, n_outliers=0)
    X_noisy, y_noisy = generate_data(noise=noise_compare, n_outliers=0)
    X_out, y_out = generate_data(noise=0.2, n_outliers=outlier_compare)

    def ols(Xd, yd):
        m = np.cov(Xd, yd)[0, 1] / np.var(Xd)
        b = np.mean(yd) - m * np.mean(Xd)
        return m, b

    mc, bc = ols(X_clean, y_clean)
    mn, bn = ols(X_noisy, y_noisy)
    mo, bo = ols(X_out, y_out)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Effect of Noise", "Effect of Outliers"))

    # Noise comparison
    fig.add_trace(go.Scatter(x=X_clean, y=y_clean, mode="markers",
        marker=dict(color="#818cf8", size=5, opacity=0.6), name="Clean Data", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_noisy, y=y_noisy, mode="markers",
        marker=dict(color="#fb923c", size=5, opacity=0.6), name="Noisy Data", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_line, y=mc * X_line + bc,
        line=dict(color="#818cf8", width=2), name="Clean Fit", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_line, y=mn * X_line + bn,
        line=dict(color="#fb923c", width=2, dash="dash"), name="Noisy Fit", showlegend=True), row=1, col=1)

    # Outlier comparison
    fig.add_trace(go.Scatter(x=X_clean, y=y_clean, mode="markers",
        marker=dict(color="#818cf8", size=5, opacity=0.6), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=X_out, y=y_out, mode="markers",
        marker=dict(color="#ef4444", size=5, opacity=0.7), name="With Outliers", showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=X_line, y=mc * X_line + bc,
        line=dict(color="#818cf8", width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=X_line, y=mo * X_line + bo,
        line=dict(color="#ef4444", width=2, dash="dash"), name="Outlier Fit", showlegend=True), row=1, col=2)

    fig.update_layout(template="plotly_dark", paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["grid"], height=430,
        legend=dict(bgcolor="rgba(0,0,0,0.4)"))
    st.plotly_chart(fig, use_container_width=True)

    # MSE comparison table
    import pandas as pd
    comparison = pd.DataFrame({
        "Scenario": ["Clean", f"Noise={noise_compare}", f"{outlier_compare} Outliers"],
        "Slope (m)": [f"{mc:.3f}", f"{mn:.3f}", f"{mo:.3f}"],
        "Intercept (b)": [f"{bc:.3f}", f"{bn:.3f}", f"{bo:.3f}"],
        "MSE": [f"{mse(X_clean, y_clean, mc, bc):.3f}",
                f"{mse(X_noisy, y_noisy, mn, bn):.3f}",
                f"{mse(X_out, y_out, mo, bo):.3f}"]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("""<div class="insight-box">
    <i class="bi bi-search"></i> <b>What to observe:</b> Even a few outliers can dramatically shift the regression line
    because MSE squares the errors — large mistakes get disproportionately large penalties.
    Increase the outlier count and watch the dashed line get pulled away from the true trend.
    </div>""", unsafe_allow_html=True)
