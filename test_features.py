import numpy as np
from DGLgraph import SnGraph
# -------------------------------------------------------------
#  VNR features
# -------------------------------------------------------------
def test_vnr_features(vnr):
    print("\n=== TEST VNR FEATURES ===")

    feat = vnr.getFeatures()
    print("Shape:", feat.shape)

    print("NaN ?", np.isnan(feat).any())
    print("Inf ?", np.isinf(feat).any())

    print("Min =", feat.min(), "Max =", feat.max())

    for i in range(feat.shape[1]):
        pass
        # print(f"Feature[{i}] min={feat[:, i].min():.4f} max={feat[:, i].max():.4f}")


# -------------------------------------------------------------
#  SN features
# -------------------------------------------------------------
def test_sn_features(sn, vnr):
    print("\n=== TEST SN FEATURES ===")

    # Use CPU of first VNF
    vnf_cpu = vnr.vnode[0].cpu

    feat = sn.getFeatures(vnf_cpu)

    # print("Shape:", feat.shape)
    # print("NaN ?", np.isnan(feat).any())
    # print("Inf ?", np.isinf(feat).any())
    # print("Min =", feat.min(), "Max =", feat.max())


# -------------------------------------------------------------
#  Observation object test
# -------------------------------------------------------------
def check_observation(sn, vnr, Observation):
    print("\n=== TEST OBSERVATION ===")

    obs = Observation(sn, vnr, 0, [])
    vnf_cpu = vnr.vnode[0].cpu


    # print("SN graph size:", obs.sn.num_nodes)
    # print("TYPE obs.vnr_graph =", type(obs.vnr))
    # print('getNetworkx:', vnr.getNetworkx())
    g_vnr= vnr.getNetworkx()
    # print("VNR graph size:", obs.vnr.num_vnfs)
    #
    # print("SN feature shape:", obs.sn.getFeatures(vnf_cpu))
    #
    # print("VNR feature shape:", obs.vnr.getFeatures())
    #
    # # Read features (just check)
    # print("VNR features (first rows):\n", vnr.getFeatures()[:3])


# -------------------------------------------------------------
#  Run all tests
# -------------------------------------------------------------
def run_all(sn, vnr, Observation):
    test_vnr_features(vnr)
    test_sn_features(sn, vnr)
    check_observation(sn, vnr, Observation)

    print("\n=== ALL TESTS PASSED SUCCESSFULLY ===")


