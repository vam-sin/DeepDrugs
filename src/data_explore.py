import pandas as pd 


'''About Dataset

The L1000 technique is a low-cost, high-throughput transcriptomics technology
that only directly measure 978 "landmark genes" rather than the full transcriptome.
(Therefore 978 columns, these refer to the 978 genes)
Mechanism of Action (MOA)
_prostate is the smaller dataset (Only 2 cell lines)
Full dataset has 36 cell lines.

A cell line is a permanently established cell culture that will proliferate 
indefinitely given appropriate fresh medium and space.

'''
ds = pd.read_hdf("../Processed_Data/GSE92742_fully_restricted_prostate.hdf")
# ds = pd.DataFrame(ds)
print(ds)