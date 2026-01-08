#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility script to auto-detect the feature dimensions for:
  - Substrate Network (SN)
  - Virtual Network Request (VNR)

This should be run BEFORE creating GNNDQN / GNNA2C to ensure:
  num_inputs_sn  = correct feature dimension
  num_inputs_vnr = correct feature dimension
"""

import numpy as np
import torch


def inspect_features(name, features):
    """Prints a clean diagnostic for a feature matrix."""
    # print(f"\n=== {name.upper()} FEATURES ===")
    # print("Shape:", features.shape)

    if np.isnan(features).any():
        print("❌ ERROR: NaN detected!")
    if np.isinf(features).any():
        print("❌ ERROR: Inf detected!")

    # print("Min:", np.min(features), "Max:", np.max(features))

    for i in range(features.shape[1]):
        col = features[:, i]
        print(f" Feature {i} → min={col.min():.4f}, max={col.max():.4f}")

    print("========================================")


def detect_feature_dims(sn, vnr):
    """
    Detects and prints:
      - num_inputs_sn
      - num_inputs_vnr
    """

    print("\n========================================")
    print("      AUTO-DETECT FEATURE DIMENSIONS     ")
    print("========================================")

    # -------------------------
    # Detect VNR feature dims
    # -------------------------
    vnr_feat = vnr.getFeatures()  # shape = (num_vnfs, k)
    inspect_features("VNR", vnr_feat)
    num_inputs_vnr = vnr_feat.shape[1]

    # -------------------------
    # Detect SN feature dims
    # -------------------------
    example_vnf_cpu = vnr.vnode[0].cpu  # any VNF CPU
    sn_feat = sn.getFeatures(example_vnf_cpu)  # shape = (num_nodes, k)
    inspect_features("SN", sn_feat)
    num_inputs_sn = sn_feat.shape[1]

    print("\n======= SUMMARY =======")
    print(f"Detected num_inputs_vnr = {num_inputs_vnr}")
    print(f"Detected num_inputs_sn  = {num_inputs_sn}")
    print("=======================\n")

    return num_inputs_sn, num_inputs_vnr
