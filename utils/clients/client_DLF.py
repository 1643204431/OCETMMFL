import numpy as np

from clients.clientbase import ClientBase

import torch
from torch.utils.data import Dataset, DataLoader

class client_DLF(ClientBase):
    def __init__(self, args, client_id):
        super(client_DLF, self).__init__(args, client_id)

    def local_train(self):
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
        Outputs = []
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
            Outputs.append(outputs)

        Y = torch.from_numpy(Y).to(self.device)
        Outputs = [torch.from_numpy(M).to(self.device) for M in Outputs]
        train_set = dataset_prediction_DLF(x=Outputs, y=Y)
        train_set_iter = DataLoader(dataset=train_set, batch_size=16, shuffle=True, drop_last=False)

        # Fusion
        for w in self.fusion.weights:
            w.parm.data = torch.FloatTensor([0.5])
        self.fusion.to(self.device)
        for step in range(10):
            for batch_index, (x, y) in enumerate(train_set_iter):
                self.fusion_optimizer.zero_grad()
                fusion = self.fusion(x)
                loss = self.loss(fusion, y)
                loss.backward()
                self.fusion_optimizer.step()
                for p in self.fusion.parameters():
                    p.data.clamp_(0, 1)

class dataset_prediction_DLF(Dataset):
    def __init__(self, x, y):
        self.len = len(x)
        self.features = x
        self.target = y

    def __getitem__(self, index):
        return [feature[index] for feature in self.features], self.target[index]

    def __len__(self):
        return self.len