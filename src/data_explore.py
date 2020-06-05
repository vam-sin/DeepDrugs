import pandas as pd 
import numpy as np 

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

# X 
X = np.asarray(ds)
X = pd.DataFrame(X)
# print(X)

# y (moa)
y_p = pd.DataFrame(ds.index)
y_p = np.asarray(y_p)
y = []
for i in y_p:
	y.append(np.asarray(i[0])[5])

y = pd.DataFrame(y)

df = pd.concat([X, y], axis = 1)
print(df)

df.to_csv("GSE92742_fully_restricted_prostate.csv")