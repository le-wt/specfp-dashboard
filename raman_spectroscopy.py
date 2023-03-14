"""Analyze Raman spectroscopy band fingerprints.

Run:
    streamlit run raman_spectroscopy.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.signal
import specfp.decoders
import streamlit as st


SAVGOL = {"window_length": 11, "polyorder": 3}


@st.cache_data
def load(file):
    """Load WDF file path or binary stream."""
    spectra = specfp.decoders.load(file)
    spectra = discretize(spectra.T)
    spectra = spectra.apply(remove_cosmic_rays, raw=True)
    spectra = spectra.apply(scipy.signal.savgol_filter, raw=True, **SAVGOL)
    spectra = spectra.apply(bubblefill)
    spectra = spectra.apply(SNV)
    std = spectra.std(axis=1).squeeze()
    spectra = spectra.mean(axis=1).squeeze()
    return spectra, std


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
_, col, _ = st.columns([1, 2, 1])
with col:
    # Set page title
    st.title("Raman Spectroscopy Band Analysis")

    # Upload files
    files = st.file_uploader(
            label="Spectra files",
            type=["wdf"],
            accept_multiple_files=True)


# Read and preprocess files
if files:
    spectra, SD = [], []
    for file in files:
        spectrum, std = load(file)
        spectrum.name, std.name = file.name, file.name
        spectra.append(spectrum)
        SD.append(std)
    spectra = pd.concat(spectra, axis=1)
    spectra.index.name = "wavelength"
    SD = pd.concat(SD, axis=1)
    SD.index.name = "wavelength"

# Visualize the loaded spectra
    df = spectra.reset_index().melt(
            "wavelength",
            var_name="acquisition",
            value_name="Raman shift")
    fig = px.line(
            df,
            x="wavelength",
            y="Raman shift",
            color="acquisition",
            color_discrete_sequence=px.colors.qualitative.Plotly)
    SD_upper = spectra + SD
    SD_lower = spectra - SD
    for trace in fig["data"]:
        col = trace.legendgroup
        if trace.line.color[0] == "#":
            color = *hex2rgb(trace.line.color), 0.2
        else:
            color = f"({trace.line.color[4:-1]}, 0.2)"
        fig.add_trace(go.Scatter(
            x=spectra.index,
            y=SD_lower[col],
            showlegend=False,
            legendgroup=col,
            mode="lines",
            line={"width": 0},
            ))
        fig.add_trace(go.Scatter(
            x=spectra.index,
            y=SD_upper[col],
            showlegend=False,
            legendgroup=col,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=f"rgba{color}",
            ))
    st.plotly_chart(fig, use_container_width=True)
