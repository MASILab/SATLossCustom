from gudhi import CubicalComplex
import nibabel as nib
from pathlib import Path

path = Path('/home/local/VANDERBILT/shij18/eye_group_tem/modified_seg/PHOTON-x-13853-x-13853_20161028_MR-x-9-SEG.nii.gz')
img = nib.load(path)
 
data = img.get_fdata()
cc = CubicalComplex(top_dimensional_cells=(1-data).astype(float))
print(f"Cubical complex is of dimension {cc.dimension()} - {cc.num_simplices()} simplices.")

cc.persistence()
print("Betti numbers:", cc.betti_numbers())
print("H0 intervals:", cc.persistence_intervals_in_dimension(0)[:10])
print("H1 intervals:", cc.persistence_intervals_in_dimension(1)[:10])
print("H2 intervals:", cc.persistence_intervals_in_dimension(2)[:10])