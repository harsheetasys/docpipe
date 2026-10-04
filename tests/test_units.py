"""Fast unit tests: no model weights, no Spark, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, ImageDraw

from docpipe.data.labels import assign_field
from docpipe.features.doc_features import FEATURE_NAMES, compute
from docpipe.features.store import FeatureStore, FeatureView
from docpipe.models.unet import UNet, dice_loss, iou_score
from docpipe.monitoring.drift import psi
from docpipe.ocr import RECIPES, cer, exact_match
from docpipe.postprocess import Detection, extract, normalise_date, normalise_total

KEY = {"company": "BOOK TA .K (TAMAN DAYA) SDN BHD", "date": "25/12/2018",
       "address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU, JOHOR.", "total": "9.00"}


# ---- labelling -------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("BOOK TA .K(TAMAN DAYA) SDN BND", "company"),
    ("25/12/2018 8:13:39 PM", "date"),
    ("NO.53 55,57 & 59, JALAN SAGU 18,", "address"),
    ("TOTAL: 9.00", "total"),
    ("ROUNDING ADJ 0.00", "other"),
    ("", "other"),
])
def test_assign_field(text, expected):
    assert assign_field(text, KEY) == expected


def test_total_not_assigned_to_random_matching_number():
    # "9.00" appears in a line that is clearly not the total line
    assert assign_field("ITEM X QTY 1 @ 9.00", KEY) == "other"


# ---- post-processing -------------------------------------------------------------------
def test_normalise_total_fixes_ocr_letters():
    assert normalise_total("TOTAL RM 1O.5O") == "10.50"
    assert normalise_total("TOTAL 1,234.56") == "1234.56"
    assert normalise_total("TOTAL") is None


def test_normalise_date():
    assert normalise_date("DATE: 25/12/2018 20:13") == "25/12/2018"
    assert normalise_date("12 Jan 2019") == "12 Jan 2019"
    assert normalise_date("no date here") is None


def test_extract_prefers_total_line_and_joins_address():
    dets = [
        Detection("total", 0.9, (0, 100, 50, 110), "SUBTOTAL 8.00"),
        Detection("total", 0.7, (0, 120, 50, 130), "TOTAL 9.00"),
        Detection("address", 0.8, (0, 30, 50, 40), "JALAN SAGU 18"),
        Detection("address", 0.8, (0, 10, 50, 20), "NO.53"),
        Detection("date", 0.8, (0, 50, 50, 60), "25/12/2018"),
        Detection("company", 0.8, (0, 0, 50, 8), "BOOK TA"),
    ]
    e = extract(dets)
    assert e.total == "9.00"
    assert e.address == "NO.53 JALAN SAGU 18"
    assert e.date == "25/12/2018" and e.company == "BOOK TA"


# ---- OCR metrics -----------------------------------------------------------------------
def test_cer_and_exact_match():
    assert cer("TOTAL 9.00", "TOTAL 9.00") == 0.0
    assert cer("", "ABCD") == 1.0
    assert 0 < cer("TOTAL 9.0O", "TOTAL 9.00") < 0.2
    assert exact_match(" total  9.00", "TOTAL 9.00")


def test_recipes_return_images():
    im = Image.new("RGB", (120, 30), "white")
    ImageDraw.Draw(im).text((5, 5), "TOTAL 9.00", fill="black")
    for name, fn in RECIPES.items():
        out = fn(im)
        assert out.mode == "L" and out.width >= im.width, name


# ---- features + store ------------------------------------------------------------------
def test_doc_features_complete_and_finite():
    im = Image.fromarray((np.random.rand(200, 100) * 255).astype(np.uint8))
    f = compute(im)
    assert set(f) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in f.values())
    assert f["aspect_ratio"] == pytest.approx(2.0)


def test_feature_store_roundtrip(tmp_path):
    fs = FeatureStore(tmp_path)
    fs.register(FeatureView(name="v", entity="doc_id", features=["a", "b"], ttl_seconds=10**9))
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    fs.write_offline("v", pd.DataFrame({"doc_id": ["x", "y"], "a": [1.0, 2.0], "b": [3.0, 4.0], "event_ts": [t0, t0]}))
    fs.write_offline("v", pd.DataFrame({"doc_id": ["x"], "a": [10.0], "b": [30.0], "event_ts": [t0 + pd.Timedelta(days=1)]}))
    latest = fs.get_historical_features("v").set_index("doc_id")
    assert latest.loc["x", "a"] == 10.0 and latest.loc["y", "a"] == 2.0      # latest wins
    asof = fs.get_historical_features("v", as_of=t0 + pd.Timedelta(hours=1)).set_index("doc_id")
    assert asof.loc["x", "a"] == 1.0                                            # point-in-time correct
    assert fs.materialize("v") == 2
    on = fs.get_online_features("v", ["x", "zzz"])
    assert on["x"]["a"] == 10.0 and on["zzz"] is None


# ---- drift -----------------------------------------------------------------------------
def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    assert psi(ref, rng.normal(0, 1, 500)) < 0.1
    assert psi(ref, rng.normal(1.5, 1, 500)) > 0.25


# ---- model -----------------------------------------------------------------------------
def test_unet_shapes_and_losses():
    m = UNet(base=4)
    x = torch.rand(2, 1, 64, 32)
    y = (torch.rand(2, 1, 64, 32) > 0.5).float()
    lg = m(x)
    assert lg.shape == y.shape
    assert 0 <= float(dice_loss(lg, y)) <= 1
    assert 0 <= iou_score(lg, y) <= 1
