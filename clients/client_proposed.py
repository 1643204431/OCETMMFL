import numpy as np

from clients.clientbase import ClientBase

import torch
from torch.utils.data import Dataset, DataLoader

class client_proposed(ClientBase):
    def __init__(self, args, client_id):
        super(client_proposed, self).__init__(args, client_id)

    def local_train(self):
        # test uni-modal network
        trainloader = self.load_train_data()
        Outputs1 = []
        for model in self.models:
            m = 0
            outputs = []
            Y = []
            for x, y in trainloader:
                x = x[:, m, :, :]
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                outputs.append(output.detach().cpu().numpy())
                Y.append(y.detach().cpu().numpy())
            m += 1
            outputs = np.concatenate(outputs)
            Y = np.concatenate(Y)
            Outputs1.append(outputs)
        Y = torch.from_numpy(Y).to(self.device)
        Outputs1 = [torch.from_numpy(M).to(self.device) for M in Outputs1]

        # train local uni-modal network
        trainloader = self.load_train_data()
        for step in range(self.local_steps):
            for model, optimizer in zip(self.models, self.optimizers):
                m = 0
                for x, y in trainloader:
                    x = x[:, m, :, :]
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    output = model(x)
                    loss = self.loss(output, y)
                    loss.backward()
                    optimizer.step()
                m += 1

        # test local uni-modal network
        trainloader = self.load_train_data()
        Outputs2 = []
        for model in self.models:
            m = 0
            outputs = []
            Y = []
            for x, y in trainloader:
                x = x[:, m, :, :]
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                outputs.append(output.detach().cpu().numpy())
                Y.append(y.detach().cpu().numpy())
            m += 1
            outputs = np.concatenate(outputs)
            Y = np.concatenate(Y)
            Outputs2.append(outputs)
        Y = torch.from_numpy(Y).to(self.device)
        Outputs2 = [torch.from_numpy(M).to(self.device) for M in Outputs2]

        # fusion
        for w in self.fusion.weights:
            w.parm.data = torch.FloatTensor([0.5])
        self.fusion.to(self.device)
        train_set = dataset_prediction_proposed(x1=Outputs1, x2=Outputs2, y=Y)
        train_set_iter = DataLoader(dataset=train_set, batch_size=16, shuffle=True, drop_last=False)
        for step in range(10):
            for batch_index, (x1, x2, y) in enumerate(train_set_iter):
                self.fusion_optimizer.zero_grad()
                fusion1 = self.fusion(x1)
                fusion2 = self.fusion(x2)
                loss = self.loss(fusion2, y) - self.loss(fusion1, y)
                loss.backward()
                self.fusion_optimizer.step()
                for p in self.fusion.parameters():
                    p.data.clamp_(0, 1)

class dataset_prediction_proposed(Dataset):
    def __init__(self, x1, x2, y):
        self.len = len(x1)
        self.features_global = x1
        self.features_local = x2
        self.target = y

    def __getitem__(self, index):
        return [feature[index] for feature in self.features_global], [feature[index] for feature in self.features_local], self.target[index]

    def __len__(self):
        return self.len