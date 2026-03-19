from __future__ import annotations

import io
import sys
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from base import Base
from filters import (
    GaussianFilter,
    PrewittFilter,
    CannyFilter,
    HoughLinesFilter,
    HoughCirclesFilter,
    HoughEllipsesFilter,
    ActiveContourFilter,
)
from utils import to_gray, to_bgr


# Shared helpers

def _upload_to_bgr() -> np.ndarray | None:
    """Render an uploader widget and return the image as a BGR ndarray."""
    uploaded = st.file_uploader(
        "Upload an image (JPG / PNG / BMP)", type=["jpg", "jpeg", "png", "bmp"]
    )
    if uploaded is None:
        return None
    pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _show(title: str, image: np.ndarray, *, gray: bool = False) -> None:
    """Display a numpy image in Streamlit (converts BGR→RGB automatically)."""
    if gray or len(image.shape) == 2:
        st.image(image, caption=title, use_container_width=True, clamp=True)
    else:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=title, use_container_width=True)


def _run_filter(flt: Base, image: np.ndarray) -> np.ndarray:
    """Thin wrapper so callers remain decoupled from concrete filter types."""
    return flt.apply(image)


# Edge / shape detection


def _sidebar_task_a() -> dict:
    """Return all Task-A hyper-parameters collected from the sidebar."""
    st.sidebar.header("Task A – Parameters")
    params: dict = {}

    params["canny_low"]  = st.sidebar.slider("Canny – low threshold",  10, 200, 50)
    params["canny_high"] = st.sidebar.slider("Canny – high threshold", 50, 400, 150)

    st.sidebar.markdown("**Hough Lines**")
    params["hl_thresh"]    = st.sidebar.slider("Line threshold",     10, 200, 80)
    params["hl_min_len"]   = st.sidebar.slider("Min line length",     5, 300, 50)
    params["hl_max_gap"]   = st.sidebar.slider("Max line gap",        1,  50, 10)

    st.sidebar.markdown("**Hough Circles**")
    params["hc_dp"]        = st.sidebar.slider("dp",          1.0, 3.0, 1.2)
    params["hc_min_dist"]  = st.sidebar.slider("Min distance", 10, 200,  30)
    params["hc_p1"]        = st.sidebar.slider("param1",       10, 300, 100)
    params["hc_p2"]        = st.sidebar.slider("param2",        5, 100,  30)

    st.sidebar.markdown("**Hough Ellipses**")
    params["he_min_pts"]   = st.sidebar.slider("Min contour points", 5, 50, 5)

    return params


def page_task_a() -> None:
    st.title("Task A – Edge & Shape Detection")
    st.markdown(
        "Detect edges (Canny), lines, circles, and ellipses (Hough), "
        "then superimpose them on the original image."
    )

    image = _upload_to_bgr()
    if image is None:
        st.info("Please upload an image to get started.")
        return

    p = _sidebar_task_a()

    col1, col2 = st.columns(2)
    with col1:
        _show("Original", image)

    #  Gaussian smoothing 
    st.subheader("Step 1 – Gaussian Smoothing")
    smoothed = _run_filter(GaussianFilter(), image)
    col1, col2 = st.columns(2)
    with col1:
        _show("Smoothed (Gaussian 7×7)", smoothed, gray=True)

    #  Prewitt edges 
    st.subheader("Step 2 – Prewitt Gradient")
    prew = PrewittFilter()
    gx, gy = prew.apply_xy(image)
    mag    = prew.apply(image)
    col1, col2, col3 = st.columns(3)
    with col1:
        _show("Gradient X", gx, gray=True)
    with col2:
        _show("Gradient Y", gy, gray=True)
    with col3:
        _show("Magnitude", mag, gray=True)

    #  Canny 
    st.subheader("Step 3 – Canny Edge Detector")
    edges = _run_filter(CannyFilter(p["canny_low"], p["canny_high"]), image)
    _show("Canny edges", edges, gray=True)

    #  Hough lines 
    st.subheader("Step 4 – Hough Lines")
    lines_img = _run_filter(
        HoughLinesFilter(
            threshold      = p["hl_thresh"],
            min_line_length= p["hl_min_len"],
            max_line_gap   = p["hl_max_gap"],
            canny_low      = p["canny_low"],
            canny_high     = p["canny_high"],
        ),
        image,
    )
    _show("Detected lines (red)", lines_img)

    #  Hough circles 
    st.subheader("Step 5 – Hough Circles")
    circles_img = _run_filter(
        HoughCirclesFilter(
            dp       = p["hc_dp"],
            min_dist = p["hc_min_dist"],
            param1   = p["hc_p1"],
            param2   = p["hc_p2"],
        ),
        image,
    )
    _show("Detected circles (green)", circles_img)

    #  Hough ellipses 
    st.subheader("Step 6 – Hough Ellipses")
    ellipses_img = _run_filter(
        HoughEllipsesFilter(
            canny_low  = p["canny_low"],
            canny_high = p["canny_high"],
            min_points = p["he_min_pts"],
        ),
        image,
    )
    _show("Detected ellipses (blue)", ellipses_img)



# Task B – Active Contour (Snake)

def _sidebar_task_b() -> dict:
    st.sidebar.header("Task B – Snake Parameters")
    return {
        "n_points":   st.sidebar.slider("Contour points",   50, 500, 200, step=10),
        "alpha":      st.sidebar.number_input("α – elasticity",  0.0, 1.0, 0.01, step=0.01, format="%.3f"),
        "beta":       st.sidebar.number_input("β – stiffness",   0.0, 1.0, 0.10, step=0.01, format="%.3f"),
        "gamma":      st.sidebar.number_input("γ – step size",   0.0, 1.0, 0.01, step=0.01, format="%.3f"),
        "iterations": st.sidebar.slider("Iterations", 10, 500, 100, step=10),
        "w_size":     st.sidebar.slider("Search window radius", 1, 10, 3),
    }


def page_task_b() -> None:
    st.title("Task B – Active Contour Model (Snake)")
    st.markdown(
        "Initialises an elliptic contour and evolves it using the greedy "
        "algorithm. The final contour is encoded as a **Freeman chain code**; "
        "perimeter and area are computed from it."
    )

    image = _upload_to_bgr()
    if image is None:
        st.info("Please upload an image to get started.")
        return

    p = _sidebar_task_b()

    col1, col2 = st.columns(2)
    with col1:
        _show("Original", image)

    run = st.button("▶  Evolve snake", type="primary")
    if not run:
        st.caption("Adjust parameters in the sidebar, then click **Evolve snake**.")
        return

    with st.spinner("Evolving contour – please wait…"):
        snake = ActiveContourFilter(
            n_points   = p["n_points"],
            alpha      = p["alpha"],
            beta       = p["beta"],
            gamma      = p["gamma"],
            iterations = p["iterations"],
            w_size     = p["w_size"],
        )
        result = snake.apply(image)

    with col2:
        _show("Evolved contour (green)", result)

    # Chain-code & metrics 
    st.subheader("Chain Code")
    cc = snake.chain_code
    st.code(" ".join(map(str, cc)), language="")

    c1, c2, c3 = st.columns(3)
    c1.metric("Chain-code length", len(cc))
    c2.metric("Perimeter (px)",    f"{snake.perimeter:.1f}")
    c3.metric("Area (px²)",        f"{snake.area:.1f}")


# App entry-point

def main() -> None:
    st.set_page_config(
        page_title="CV Assignment 2",
        page_icon="🔍",
        layout="wide",
    )

    # Navigation 
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select task",
        ["Task A – Edge & Shape Detection", "Task B – Active Contour (Snake)"],
    )

    if page.startswith("Task A"):
        page_task_a()
    else:
        page_task_b()


if __name__ == "__main__":
    main()