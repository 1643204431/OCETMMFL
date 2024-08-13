# Optimization-Framework-in-Joint-Client-and-Modality-Selection-for-Multi-Modal-Federated-Learning
Code for Papaer Optimization Framework in Joint Client and Modality Selection for Multi-Modal Federated Learning

## Environments
You can use requirements.txt

## Dataset
You can download the dataset in https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones, then you can run generate generate_har.py to generate dataset to simulate multi-modal FL environment.

## Train and Evaluation
We have implemented two modality selection methods in client_proposed.py and client_DLF.py, respectively. The random selection method can be directly achieved using the without-replacement sampling function provided by the NumPy library. To change the modality selection method, simply modify the function call in server.py.

```python
Run main.py
