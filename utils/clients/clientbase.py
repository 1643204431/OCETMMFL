import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data

from trainmodel.har_models import modality_model, FusionNet

class ClientBase(object):
    def __init__(self, args, client_id):
        self.dataset = args.dataset
        self.device = args.device
        self.client_id = client_id  # integer
        self.save_folder_name = args.save_folder_name
        self.num_classes = args.num_classes

        self.batch_size = args.batch_size
        self.local_steps = args.local_steps

        self.learning_rate = args.local_learning_rate
        self.train_samples = 0

        self.K2 = 7

        self.n_modalities = 9
        self.is_joins = [1] * self.n_modalities
        self.join_counts = [0] * self.n_modalities

        self.models = [modality_model().to(self.device) for _ in range(self.n_modalities)]
        self.optimizers = [torch.optim.SGD(model.parameters(), lr=self.learning_rate) for model in self.models]

        self.fusion = FusionNet().to(self.device)
        self.fusion_optimizer = torch.optim.SGD(self.fusion.parameters(), lr=self.learning_rate)

        self.loss = nn.CrossEntropyLoss()
        self.acc_m = []

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        train_data = read_client_data(self.dataset, self.client_id, is_train=True)
        #self.train_samples = len(train_data)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.client_id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)


    def update_parameters(self, model, new_model):
        for param, new_param in zip(model.parameters(), new_model.parameters()):
            param.data = new_param.data.clone()

    def save_item(self):
        item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)

        for modality_id, model in zip(self.modalities, self.models):
            torch.save(model, os.path.join(item_path, "client_" + str(self.client_id) +
                                           "modality_" + str(modality_id) + ".pt"))

    def load_item(self):
        item_path = self.save_folder_name
        return [torch.load(os.path.join(item_path, "client_" + str(self.client_id) + "modality_" +
                                        str(modality_id) + ".pt")) for modality_id in self.modalities]

    def test(self, is_fuse=True):
        testloader = self.load_test_data()
        Outputs = []
        acc_modality = []
        for model in self.models:
            m = 0
            outputs = []
            Y = []
            for x, y in testloader:
                x = x[:, m, :, :]
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                outputs.append(output)
                Y.append(y)
            m += 1
            outputs = torch.cat(outputs, dim=0)
            Y = torch.cat(Y, dim=0)
            Y = Y.detach().cpu().numpy()
            acc_modality.append(np.mean(np.argmax(outputs.detach().cpu().numpy(), axis=1) == Y))
            Outputs.append(outputs)

        if is_fuse:
            fusion = self.fusion(Outputs).detach().cpu().numpy()
            acc = np.mean(np.argmax(fusion, axis=1) == Y)
            self.acc2 = np.array(acc_modality)
            return acc

        else:
            self.acc1 = np.array(acc_modality)
            return 0

    def modality_selection(self):
        self.is_joins = [1] * self.n_modalities
        parms = []
        for weight in self.fusion.weights:
            parms.append(weight.parm.detach().cpu().numpy()[0])
        parms = np.array(parms)
        modality_importance_id = np.argsort(- parms)

        for m in range(self.n_modalities):
            if m not in modality_importance_id[0:self.K2]:
                self.is_joins[m] = 0


        for m in range(self.n_modalities):
            if m in modality_importance_id[0:self.K2]:
                self.join_counts[m] += 1