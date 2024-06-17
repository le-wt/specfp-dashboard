"""Analyze Raman spectroscopy band fingerprints.

Run:
    streamlit run raman_spectroscopy.py
"""

from streamlit_plotly_events import plotly_events

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.signal
import specfp.decoders
import streamlit as st


SAVGOL = {"window_length": 11, "polyorder": 3}


@st.cache_data
def load(
        file: str,
        cosmic_rays_filter: bool = True,
        savgol_filter: bool = True,
        bubblefill_filter: bool = True,
        standard_normal_variate: bool = True,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load WDF file path or binary stream."""
    spectra = specfp.decoders.load(file)
    spectra = discretize(spectra.T)
    if cosmic_rays_filter:
        spectra = spectra.apply(remove_cosmic_rays, raw=True)
    if savgol_filter:
        spectra = spectra.apply(scipy.signal.savgol_filter, raw=True, **SAVGOL)
    if bubblefill_filter:
        spectra = spectra.apply(bubblefill)
    if standard_normal_variate:
        spectra = spectra.apply(SNV)
    variance = spectra.var(axis=1).squeeze()
    spectra = spectra.mean(axis=1).squeeze()
    return spectra, variance


def discretize(df: pd.DataFrame, copy: bool = False) -> pd.DataFrame:
    """Discretize the index of a dataframe to the closest integral value."""
    if copy:
        df = df.copy()
    df.index = np.round(df.index).astype(int)
    df = df.loc[~df.index.duplicated()]
    df = df.reindex(range(df.index.min(), df.index.max() + 1))
    df = df.interpolate(method="linear")
    return df


def remove_cosmic_rays(spectrum: np.ndarray, stdfactor: int = 3) -> pd.DataFrame:
    """Remove cosmic rays within the 320-340 and 365-385nm regions."""
    scale = spectrum.max()
    spectrum = spectrum / scale
    d1 = np.diff(spectrum)
    d2 = np.diff(d1)
    threshold = d2.mean() + stdfactor * d2.std()
    cosmic_ray = np.array([False for i in range(len(spectrum))])
    cosmic_ray[1:-1] = np.abs(d2) > threshold
    for i in np.where(cosmic_ray)[0]:
        if 320 < i < 340:
            cosmic_ray[i] = False
            continue
        if 365 < i < 385:
            cosmic_ray[i] = False
            continue
        cosmic_ray[i - 1] = True
        cosmic_ray[i + 1] = True
    x = np.arange(len(spectrum))
    xp = x[~cosmic_ray]
    yp = spectrum[~cosmic_ray]
    spectrum_ = np.interp(x, xp, yp)
    spectrum_ = scale * spectrum_
    return spectrum_


def bubblefill(spectrum, bubblewidths=40, fitorder=1, do_smoothing=True):
    """Compute the Raman shift of a spectral acquisition."""

    def keep_largest(x0, x2, baseline, bubble):
        for j in range(x0, x2 + 1):
            if baseline[j] < bubble[j]:
                baseline[j] = bubble[j]
        return baseline

    def grow_bubble(x, x0, x2, bpos, bwidth, s):
        A = (bwidth / 2)**2 - (x - bpos)**2
        A[A < 0] = 0
        bubble = np.sqrt(A) - bwidth
        x1 = x0 + (s[x0:x2 + 1] - bubble[x0:x2 + 1]).argmin()
        bubble = bubble + (s[x0:x2 + 1] - bubble[x0:x2 + 1]).min()
        return bubble, x1

    def bubbleloop(bubblewidths, x, s, baseline):
        bubblecue = [[0, len(s) - 1]]
        i = 0
        while i < len(bubblecue):
            x0, x2 = bubblecue[i]
            i += 1
            if x0 == x2:
                continue
            if type(bubblewidths) is not int:
                bubblewidth = bubblewidths[(x0 + x2) // 2]
            else:
                bubblewidth = bubblewidths
            if x0 == 0 and x2 != len(s) - 1:
                bwidth = 2 * (x2 - x0)
                bpos = x0
            elif x0 != 0 and x2 == len(s) - 1:
                bwidth = 2 * (x2 - x0)
                bpos = x2
            else:
                if (x2 - x0) < bubblewidth:
                    continue
                bwidth = (x2 - x0)
                bpos = (x0 + x2) / 2
            bubble, x1 = grow_bubble(x, x0, x2, bpos, bwidth, s)
            baseline = keep_largest(x0, x2, baseline, bubble)
            if x1 == x0:
                bubblecue.append([x1 + 1, x2])
            elif x1 == x2:
                bubblecue.append([x0, x1 - 1])
            else:
                bubblecue.append([x0, x1])
                bubblecue.append([x1, x2])
        return baseline

    s = spectrum
    x = np.arange(len(s))
    slope = np.poly1d(np.polyfit(x, spectrum, fitorder))(x)
    s = s - slope
    smin = s.min()
    s = s - smin
    scale = (s.max() / len(s))
    s = s / scale
    baseline = np.zeros(s.shape)
    baseline = bubbleloop(bubblewidths, x, s, baseline)
    baseline = baseline * scale + slope + smin
    if isinstance(bubblewidths, int) and do_smoothing:
        baseline = scipy.signal.savgol_filter(
                baseline, 2 * (bubblewidths // 4) + 1, 3)
    raman = spectrum - baseline
    return raman

def SNV(array) -> np.ndarray:
    """Standard normal variate of a spectrum."""
    if array.ndim == 2:
        array[..., 1:, :] -= array[..., 1:, :].mean(axis=sans_penultimate(array))
        array[..., 1:, :] /= array[..., 1:, :].std(axis=sans_penultimate(array))
    else:
        array -= array.mean()
        array /= array.std()
    return array


def hex2rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return (int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16))


# Configure app
st.set_page_config(layout="wide")
if "peak" not in st.session_state:
    st.session_state["peak"] = {}

# Set page title
st.title("Raman Spectroscopy Band Analysis")

# Select files and the number of spectra to produce
with st.sidebar:
    st.header("1. Upload files")
    files = st.file_uploader(
            label="Spectra files",
            type=["wdf"],
            accept_multiple_files=True)
    st.header("2. Apply signal preprocessing filters")
    filters = {
        "cosmic_rays_filter": st.checkbox("Remove cosmic rays", True),
        "savgol_filter": st.checkbox("Apply the Savgol filter", True),
        "bubblefill_filter":  st.checkbox("Apply the BubbleFill baseline removal", True),
        "standard_normal_variate":  st.checkbox("Apply standard normal variate", True),
    }
    st.header("3. Form new averages of files")
    averages, names, colors = [], [], []
    for selector, default_color in zip(
            range(st.number_input("Number of spectra averages", 0)),
            px.colors.qualitative.Plotly):
        averages.append(st.multiselect(
            "Spectra to be averaged",
            [file.name for file in files],
            key=f"multiselect-{selector}"))
        names.append(st.text_input(
            f"New name for average {selector + 1}",
            key=f"text-{selector}"))
        colors.append(st.color_picker(
            f"Spectrum color for average {selector + 1}",
            key=f"color-{selector}",
            value=default_color))


if files:
    # Read and preprocess files
    spectra, variances = [], []
    for file in files:
        spectrum, variance = load(file, **filters)
        spectrum.name = file.name
        variance.name = file.name
        spectra.append(spectrum)
        variances.append(variance)
    spectra = pd.concat(spectra, axis=1)
    spectra.index.name = "Raman shift (cm⁻¹)"
    variances = pd.concat(variances, axis=1)
    variances.index.name = "Raman shift (cm⁻¹)"
    for name, names, color in zip(names, averages, colors):
        spectra[name] = spectra[names].mean(axis=1)
        variances[name] = variances[names].mean(axis=1)
        spectra.drop(columns=names, inplace=True)
        variances.drop(columns=names, inplace=True)
    # Change the data representation to tidy data
    df = spectra.reset_index().melt(
            "Raman shift (cm⁻¹)",
            var_name="Acquisition",
            value_name="intensity")

    # Visualize the loaded spectra
    fig = px.line(
            df,
            x="Raman shift (cm⁻¹)",
            y="intensity",
            markers=True,
            color="Acquisition",
            color_discrete_sequence=colors,
            hover_name="Acquisition",
            hover_data={"intensity": True,
                        "Raman shift (cm⁻¹)": False,
                        "Acquisition": False})
    fig.update_yaxes(visible=False)
    fig.update_traces(opacity=0.75)
    # Visualize the standard deviation
    std_upper = spectra + variances ** 0.5
    std_lower = spectra - variances ** 0.5
    for trace in fig["data"]:
        col = trace.legendgroup
        if trace.line.color[0] == "#":
            color = *hex2rgb(trace.line.color), 0.2
        else:
            color = f"({trace.line.color[4:-1]}, 0.2)"
        fig.add_trace(go.Scatter(
            x=spectra.index,
            y=std_lower[col],
            showlegend=False,
            legendgroup=col,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=spectra.index,
            y=std_upper[col],
            showlegend=False,
            legendgroup=col,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=f"rgba{color}",
            hoverinfo="skip"))

    # Plot the generated figure
    fig.update_layout(hovermode="x unified", dragmode="select", xaxis={"showgrid": False})
    fig.update_traces(marker={"size": 1})
    # st.plotly_chart(fig, use_container_width=True)
    for coord, enable in st.session_state["peak"].items():
        if enable:
            fig.add_vline(coord, opacity=0.2, line_dash="dot")

    # Update figure on click events
    graph = st.empty()
    st.caption("Usage: Select a region around a peak to place a vertical lines.")
    with st.expander("Configure figure"):
        st.subheader("Control figure size")
        left, right = st.columns(2)
        with left:
            width = st.number_input("Figure width (pixels)", 1, value=1000, step=50)
        with right:
            height = st.number_input("Figure height (pixels)", 1, value=500, step=50)
        refresh = False
        st.subheader("Select peaks")
        for coord, enable in st.session_state["peak"].items():
            value = st.checkbox(
                    f"Enable peak at {coord}nm",
                    enable)
            st.session_state["peak"][coord] = value
            if value != enable:
                refresh = True
        if refresh:
            st.experimental_rerun()
    with graph.container():
        box = plotly_events(
                fig,
                click_event=False,
                select_event=True,
                override_width=width,
                override_height=height)
    if box:
        coord = max(box, key=lambda point: point["y"])["x"]
        st.session_state["peak"][coord] = True
        st.experimental_rerun()
