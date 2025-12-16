import numpy as np
try:
    sdm = np.load("data/chengdu_directed_shortest_distance_matrix.npy")
    print(f"Shape: {sdm.shape}")
    print(f"Non-zero elements: {np.count_nonzero(sdm)}")
except Exception as e:
    print(e)
