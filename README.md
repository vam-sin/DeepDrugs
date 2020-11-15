# DeepDrugs
Deep Learning model to predict which drugs can be repurposed based on the their transcriptome data.

# Requirements

```python3
pip3 install h5py
pip3 install keras
pip3 install tensorflow
```

# Data

The LINCS GSE92742) gene expression dataset was exploited for this model. The following code allows you to explore the dataset.

```python3
python3 data_explore.py
```

Run the above code in the /src folder.

# Model

Two different models, a Graph Neural Network and a Multi Layer Perceptron model were trained on the dataset. The models can be trained using the code below.

```python3
python3 gcn.py
python3 mlp.py
```

The above code is to be run in the /src folder.

Additionally, an autoencoder has been built to reduce the number of input dimensions. The autoencoder can be run using the following code.

```python3
python3 autoencoder.py
```

