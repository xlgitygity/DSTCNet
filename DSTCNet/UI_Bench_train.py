import numpy as np
import torch
import torch.nn as nn
from Model import DSTCNet
import random
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from pytorchtools import EarlyStopping
from Model import Filterbank
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

s_len = 25
step = 1
fs = 250

### https://bci.med.tsinghua.edu.cn/download.html ###
### Download the dataset from this website ###

def train_data(test_subject, duration):
    all_data = []
    all_labels = []
    all_subject = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                   31, 32, 33, 34, 35]
    train_subject = []

    for s in all_subject:
        if s != test_subject:
            train_subject.append(s)

    for ts in range(len(train_subject)):
        subject_file = '/path/to/your/Dataset/Benchmark/S' + str(test_subject) + '.mat'
        rawdata = loadmat(subject_file)
        rawdata = rawdata['eeg']
        for j in range(4):
            block_index = j
            for k in range(40):
                target_index = k
                for s in range(step):
                    channel_data = rawdata[:, int(160 + s_len * s):int(160 + s_len * s + duration * fs), target_index,
                                   block_index]
                    all_labels.append(target_index)
                    all_data.append(channel_data)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    return all_data, all_labels


def valid_data(test_subject, duration):
    all_data = []
    all_labels = []
    valid_file = '/path/to/your/Dataset/Benchmark/S' + str(test_subject) + '.mat'
    valid_data = loadmat(valid_file)
    valid_data = valid_data['eeg']
    for j in range(4):
        block_index = j
        for k in range(40):
            target_index = k
            for s in range(step):
                channel_data = valid_data[:, int(160 + s_len * s):int(160 + s_len * s + duration * fs), target_index,
                               block_index]
                all_labels.append(target_index)
                all_data.append(channel_data)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    return all_data, all_labels


class TrainDataset(Dataset):

    def __init__(self, duration, test_subject, sampling_frequency, transformer=None):
        super(TrainDataset, self).__init__()
        self.data = []
        self.label = []
        data_list, all_label = train_data(test_subject, duration)
        self.data = data_list
        self.label = all_label
        self.transform = transformer
        self.sampling_frequency = sampling_frequency

    def __len__(self):
        assert len(self.data) == len(self.label)
        return len(self.data)

    def __getitem__(self, item):
        s_data = self.data[item]
        s_label = self.label[item]

        filtered_data = []
        for filterIdx in range(3):
            filtered_band = Filterbank(s_data, self.sampling_frequency, filterIdx)
            filtered_data.append(filtered_band)

        filtered_data = np.array(filtered_data)  # (n_bands, channels, data)

        if self.transform is not None:
            filtered_data = self.transform(filtered_data)

        return filtered_data, s_label


class ValidDataset(Dataset):

    def __init__(self, duration, test_subject, sampling_frequency, transformer=None):
        super(ValidDataset, self).__init__()
        self.data = []
        self.label = []
        data_list, all_label = valid_data(test_subject, duration)
        self.data = data_list
        self.label = all_label
        self.transform = transformer
        self.sampling_frequency = sampling_frequency

    def __len__(self):
        assert len(self.data) == len(self.label)
        return len(self.data)

    def __getitem__(self, item):
        s_data = self.data[item]
        s_label = self.label[item]

        filtered_data = []
        for filterIdx in range(3):
            filtered_band = Filterbank(s_data, self.sampling_frequency, filterIdx)
            filtered_data.append(filtered_band)

        filtered_data = np.array(filtered_data)  # (n_bands, channels, data)

        if self.transform is not None:
            filtered_data = self.transform(filtered_data)

        return filtered_data, s_label

class ToTensor(object):
    def __call__(self, seq):
        return torch.tensor(seq, dtype=torch.float)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr, num_epochs, batch_size = 0.001, 100, 64
    patience = 10
    delta = 1e-5
    n_splits = 5
    foldperf = {}

    for duration in np.arange(0.5, 1.2, 0.1):
        data_length = int(fs * duration)
        print(f'duration: {duration}')
        for subject_id in range(0, 35):
            tic = time.time()
            subject = subject_id + 1
            print('————————————————————————————————————————\n')
            print('Subject-Fold for validation: S{} \n'.format(subject))
            train_dataset = TrainDataset(duration, subject, fs, ToTensor())
            test_dataset = ValidDataset(duration, subject, fs, ToTensor())
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
            train_size = len(train_dataset)
            test_size = len(test_dataset)

            model = DSTCNet(
                num_channel=11,
                num_classes=40,
                signal_length=int(fs * duration),
            )

            model = model.to(device)

            loss_fn = nn.CrossEntropyLoss()
            if torch.cuda.is_available():
                loss_fn = loss_fn.cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            clr = CosineAnnealingLR(optimizer, T_max=10)

            one_fold_path = '/path/to/your/project/model/pretrained/DSTCNet_pretrain_' + str(duration) + 's_S' + str(subject) + '.pth'
            early_stopping = EarlyStopping(patience=patience, verbose=True, path=one_fold_path)

            total_train_step = 0
            total_test_step = 0
            history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
            for epoch in range(num_epochs):
                model.train()
                total_train_loss = 0
                total_train_accuracy = 0
                avg_train_acc = 0
                for data in train_dataloader:
                    s_data, s_label = data
                    info = torch.Tensor(s_data)
                    info = info.to(device)
                    s_label = s_label.to(torch.long)
                    s_label = s_label.to(device)
                    outputs = model(info)
                    loss = loss_fn(outputs, s_label)
                    total_train_loss = total_train_loss + loss.cpu().detach().numpy().item()
                    accuracy = (outputs.argmax(axis=1) == s_label).sum()
                    total_train_accuracy = total_train_accuracy + accuracy

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_train_step = total_train_step + 1
                avg_train_acc = float(total_train_accuracy / train_size * 100)
                clr.step()

                model.eval()
                total_test_loss = 0
                total_accuracy = 0
                with torch.no_grad():
                    for data in test_dataloader:
                        s_data, s_label = data
                        info = torch.Tensor(s_data)
                        info = info.to(device)
                        s_label = s_label.to(torch.long)
                        s_label = s_label.to(device)
                        outputs = model(info)
                        loss = loss_fn(outputs, s_label)
                        total_test_loss = total_test_loss + loss.cpu().detach().numpy().item()
                        accuracy = (outputs.argmax(axis=1) == s_label).sum() 
                        total_accuracy = total_accuracy + accuracy

                if (epoch + 1) == num_epochs:
                    history['train_loss'].append(total_train_loss)
                    history['train_acc'].append(avg_train_acc)
                    history['test_loss'].append(total_test_loss)
                    history['test_acc'].append(float(total_accuracy / test_size * 100))

                early_stopping(total_test_loss, model)
                if early_stopping.early_stop:
                    history['train_loss'].append(total_train_loss)
                    history['train_acc'].append(avg_train_acc)
                    history['test_loss'].append(total_test_loss)
                    history['test_acc'].append(float(total_accuracy / test_size * 100))
                    print("Early stopping")
                    break

            foldperf['fold_S{}'.format(subject)] = history
            print(history)
            print(time.time() - tic)
        time.sleep(50)