import numpy as np
from gudhi import CubicalComplex

def analyze(vol, name):
    cc = CubicalComplex(top_dimensional_cells=(1-vol).astype(float))
    cc.persistence()
    print(f"\n{name}")
    print("shape:", vol.shape)
    print("unique:", np.unique(vol))
    print("Betti:", cc.betti_numbers())
    print("H0:", cc.persistence_intervals_in_dimension(0)[:10])
    print("H1:", cc.persistence_intervals_in_dimension(1)[:10])
    print("H2:", cc.persistence_intervals_in_dimension(2)[:10])

solid = np.zeros((20, 20, 20), dtype=np.uint8)
solid[5:15, 5:15, 5:15] = 1

two_blocks = np.zeros((20, 20, 20), dtype=np.uint8)
two_blocks[3:8, 3:8, 3:8] = 1
two_blocks[12:17, 12:17, 12:17] = 1

hollow = np.zeros((20, 20, 20), dtype=np.uint8)
hollow[4:16, 4:16, 4:16] = 1
hollow[7:13, 7:13, 7:13] = 0

analyze(solid, "solid")
analyze(two_blocks, "two_blocks")
analyze(hollow, "hollow")