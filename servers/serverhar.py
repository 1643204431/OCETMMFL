import torch
import os
import numpy as np
import time
import random

from clients.client_proposed import client_proposed
from clients.client_DLF import client_DLF
from trainmodel.har_models import modality_model

class Server(object):
    def __init__(self, args):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.n_modalities = 9
        self.join_counts = [1] * self.n_modalities
        self.history_counts = []
        # init global models
        self.global_models = [modality_model().to(self.device) for _ in range(self.n_modalities)]
        self.modality_size = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.selected_clients = []
        self.save_path = None
        self.method = 'DLF'
        #self.method = 'Proposed'

        # init client
        clients = list(range(30))
        self.clients = []
        for client_id in clients:
            client_id += 1
            client = client_proposed(self.args, client_id=client_id)
            self.clients.append(client)
        self.clients_counts = np.zeros((30, 9))


    def load_model(self, path=None):
        model_path = os.path.join(path, "server.pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def save_model(self, path=None):
        if path is None:
            path = self.save_path
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.global_model, os.path.join(path, "server.pt"))

    def local_train(self):
        for client in self.clients:
            client.local_train()
            client.modality_selection()

    def test(self, is_fuse):
        Acc = []
        if is_fuse:
            for client in self.clients:
                acc = client.test(is_fuse)
                Acc.append(acc)
            return Acc
        else:
            for client in self.clients:
                client.test(is_fuse)

    def receive_aggregate_models(self):
        self.join_counts = [0] * 9
        for client in self.selected_clients:
            for m in range(self.n_modalities):
                self.join_counts[m] += client.is_joins[m]

        self.history_counts.append(self.join_counts)

        for global_model in self.global_models:
            for param in global_model.parameters():
                param.data.zero_()
        #print(self.join_counts)

        # key: modality, value:all the local models
        for client in self.selected_clients:
            m = 0
            for model, is_join, join_counts, global_model in zip(client.models, client.is_joins, self.join_counts, self.global_models):
                if is_join:
                    self.clients_counts[client.client_id-1, m] += 1
                    for server_param, client_param in zip(global_model.parameters(), model.parameters()):
                        server_param.data += client_param.data.clone() * (1 / join_counts)
                m += 1

    def disperse_models(self):
        for client in self.clients:
            for model, join_counts, global_model in zip(client.models, self.join_counts, self.global_models):
                if join_counts > 0:
                    for param, new_param in zip(model.parameters(), global_model.parameters()):
                        param.data = new_param.data.clone()

    def client_selection(self, K1=15, beta_=20):
        self.avg_modalities = np.array(self.join_counts) / len(self.clients)
        total_modality_size = []
        total_training_potential = []
        dist = []
        clientid = []
        for client in self.clients:
            clientid.append(client.client_id)
            # modality selection diversity
            dist.append(np.dot(np.array(client.is_joins), np.array(self.avg_modalities))/\
                          (np.linalg.norm(np.array(client.is_joins)) * np.linalg.norm(np.array(self.avg_modalities))))

            # total_modality_size
            total_modality_size.append(np.sum(self.modality_size[np.array(client.is_joins)==1]))

            # total_training_potential
            total_training_potential.append(np.sum((client.acc2-client.acc1)[np.array(client.is_joins)==1]))

        total_training_potential = np.array(total_training_potential)
        total_modality_size = np.array(total_modality_size)
        dist = np.array(dist)
        total_training_potential = (total_training_potential - np.min(total_training_potential)) / \
                                   (np.max(total_training_potential) - np.min(total_training_potential))

        client_importance = dist + beta_*(total_training_potential) / (total_modality_size)
        print(dist, total_training_potential, total_modality_size)
        selected_clients_id = np.argsort(- client_importance)[0:K1]

        for client in self.clients:
            if client.client_id in selected_clients_id:
                self.selected_clients.append(client)

    def train(self):
        acc_list = []
        for epoch in range(5):
            start = time.time()

            self.disperse_models()
            self.test(is_fuse=False)
            self.local_train()

            acc = self.test(is_fuse=True)
            acc_list.append(np.mean(acc))

            self.client_selection()
            self.receive_aggregate_models()

            end = time.time()
            print(end-start, np.mean(acc))

        return acc_list

    def joins_count(self):
        client_counts = []
        for client in self.clients:
            client_counts.append(client.join_counts)
        return client_counts

