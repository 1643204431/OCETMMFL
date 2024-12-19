# OCETMMFL
Code for Papaer Optimizing Communication Efficiency through TrainingPotential in Multi-Modal Federated Learning

## Preparing

### Environments
You can use requirements.txt

### Dataset
You can download the UCI HAR dataset in https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones, then you can run generate generate_har.py to generate dataset to simulate multi-modal FL environment. Then generate_har dataset via:
```python
Run generate_har.py
```
You can download the Ninapro DB2 and DB7 in https://ninapro.hevs.ch/instructions/DB2.html. Then generate these two datasets via:
```python
Run generate_DB2.py
```
```python
Run generate_DB7.py
```
You can download the ActionSense dataset in https://action-net.csail.mit.edu/. Then generate ActionSense via:
```python
Run generate_ActionSense.py
```

The CG MNIST is from https://github.com/jayaneetha/colorized-MNIST.

## Train and Evaluation

### Select a modality selection method
We have implemented two modality selection methods in client_proposed.py and client_DLF.py, respectively. The random selection method can be directly achieved using the without-replacement sampling function provided by the NumPy library. To change the modality selection method, simply modify the function call in server.py. For example:

```python
# init a DLF client
from clients.client_DLF import client_DLF
clients = list(range(30))
self.clients = []
for client_id in clients:
    client_id += 1
    client = client_DLF(self.args, client_id=client_id)
    self.clients.append(client)
```

```python
# init a Proposed client
from clients.client_proposed import client_proposed
clients = list(range(30))
self.clients = []
for client_id in clients:
    client_id += 1
    client = client_proposed(self.args, client_id=client_id)
    self.clients.append(client)
```

### Run
```python
Run main.py
```
