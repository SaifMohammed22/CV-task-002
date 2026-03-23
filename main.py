from __future__ import annotations

import io
import sys
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from src.base import Base
from src.filters import (
    GaussianFilter,
    SobelFilter,
    CannyFilter,
    HoughLinesFilter,
    HoughCirclesFilter,
    HoughEllipsesFilter,
    ActiveContourFilter,
)
from src.utils import to_gray, to_bgr


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


def _signed_gradient_to_uint8(grad: np.ndarray) -> np.ndarray:
    """Map signed gradients to uint8 for display (0->negative, 128->zero, 255->positive)."""
    grad64 = grad.astype(np.float64)
    max_abs = float(np.max(np.abs(grad64)))
    if max_abs == 0:
        return np.full(grad64.shape, 128, dtype=np.uint8)
    scaled = (grad64 / max_abs) * 127.0 + 128.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _inject_task_a_styles() -> None:
    """Apply lightweight styling for clearer phase comparison blocks."""
    st.markdown(
        """
        <style>
        .phase-card {
            border: 1px solid rgba(120, 134, 160, 0.35);
            border-radius: 14px;
            padding: 0.8rem 0.9rem 0.25rem 0.9rem;
            margin-bottom: 1.1rem;
            background: linear-gradient(180deg, rgba(30, 37, 52, 0.28), rgba(19, 24, 35, 0.18));
        }
        .phase-title {
            font-weight: 700;
            letter-spacing: 0.01em;
            font-size: 1.02rem;
            margin-bottom: 0.4rem;
            color: #EAF1FF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_phase_compare(
    step_title: str,
    original: np.ndarray,
    processed: np.ndarray,
    processed_title: str,
    *,
    processed_gray: bool = False,
) -> None:
    """Render a side-by-side original vs processed view for one phase."""
    st.markdown("<div class='phase-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='phase-title'>{step_title}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        _show("Original", original)
    with c2:
        _show(processed_title, processed, gray=processed_gray)
    st.markdown("</div>", unsafe_allow_html=True)


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
    params["he_min_area"]  = st.sidebar.slider("Min contour area", 20, 2000, 120)

    return params


def page_task_a() -> None:
    _inject_task_a_styles()
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

    st.sidebar.markdown("---")
    st.sidebar.subheader("Original Reference")
    st.sidebar.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Pinned original",
        use_container_width=True,
    )

    #  Gaussian smoothing 
    smoothed = _run_filter(GaussianFilter(), image)
    _show_phase_compare(
        "Step 1 – Gaussian Smoothing",
        image,
        smoothed,
        "Smoothed (Gaussian 7x7)",
        processed_gray=True,
    )

    #  Sobel edges 
    sobel = SobelFilter()
    gx, gy = sobel.apply_xy(smoothed)
    mag = sobel.apply(smoothed)
    gx_vis = _signed_gradient_to_uint8(gx)
    gy_vis = _signed_gradient_to_uint8(gy)

    st.markdown("<div class='phase-card'>", unsafe_allow_html=True)
    st.markdown("<div class='phase-title'>Step 2 – Sobel Gradient</div>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["GX (signed)", "GY (signed)", "Magnitude"])
    with t1:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            _show("Original", image)
        with c2:
            _show("Gradient X (signed)", gx_vis, gray=True)
    with t2:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            _show("Original", image)
        with c2:
            _show("Gradient Y (signed)", gy_vis, gray=True)
    with t3:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            _show("Original", image)
        with c2:
            _show("Magnitude", mag, gray=True)
    st.markdown("</div>", unsafe_allow_html=True)

    #  Canny 
    edges = _run_filter(CannyFilter(p["canny_low"], p["canny_high"]), image)
    _show_phase_compare(
        "Step 3 – Canny Edge Detector",
        image,
        edges,
        "Canny edges",
        processed_gray=True,
    )

    #  Hough lines 
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
    _show_phase_compare(
        "Step 4 – Hough Lines",
        image,
        lines_img,
        "Detected lines (red)",
    )

    #  Hough circles 
    circles_img = _run_filter(
        HoughCirclesFilter(
            dp       = p["hc_dp"],
            min_dist = p["hc_min_dist"],
            param1   = p["hc_p1"],
            param2   = p["hc_p2"],
        ),
        image,
    )
    _show_phase_compare(
        "Step 5 – Hough Circles",
        image,
        circles_img,
        "Detected circles (green)",
    )

    #  Hough ellipses 
    ellipses_img = _run_filter(
        HoughEllipsesFilter(
            canny_low  = p["canny_low"],
            canny_high = p["canny_high"],
            min_points = p["he_min_pts"],
            min_area   = p["he_min_area"],
        ),
        image,
    )
    _show_phase_compare(
        "Step 6 – Hough Ellipses",
        image,
        ellipses_img,
        "Detected ellipses (blue)",
    )



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

    run = st.button("▶  Evolve snake", type="primary")
    if not run:
        c1, c2 = st.columns(2)
        with c1:
            _show("Original", image)
        with c2:
            st.info("Click **Evolve snake** to preview initialized and evolved contours.")
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

    init_overlay = to_bgr(image)
    if snake.initial_contour is not None:
        cv2.polylines(
            init_overlay,
            [snake.initial_contour],
            isClosed=True,
            color=(0, 215, 255),
            thickness=2,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        _show("Original", image)
    with c2:
        _show("Initialized contour (yellow)", init_overlay)
    with c3:
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