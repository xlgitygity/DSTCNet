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

w_len = 25
step_num = 1
fs = 250

### https://bci.med.tsinghua.edu.cn/download.html ###
### Download the dataset from this website ###

def train_data(subject, duration):
    all_data = []
    all_labels = []
    train_block = [0, 1, 2]
    subject_file = '/path/to/your/Dataset/Benchmark/S' + str(subject) + '.mat'
    rawdata = loadmat(subject_file)
    rawdata = rawdata['eeg']
    for k in range(40):
        target_idx = k
        for ts in range(len(train_block)):
            block_idx = train_block[ts]
            for step in range(step_num):
                channel_data = rawdata[:, int(160 + w_len * step):int(160 + w_len * step + duration * fs), target_idx,
                               block_idx]
                all_labels.append(target_idx)
                all_data.append(channel_data)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    return all_data, all_labels


def valid_data(subject, duration):
    all_data = []
    all_labels = []
    subject_file = '/path/to/your/Dataset/Benchmark/S' + str(subject) + '.mat'
    rawdata = loadmat(subject_file)
    rawdata = rawdata['eeg']
    valid_b = [3]
    for k in range(40):
        target_idx = k
        for ts in range(len(valid_b)):
            block_idx = valid_b[ts]
            for step in range(step_num):
                channel_data = rawdata[:, int(160 + w_len * step):int(160 + w_len * step + duration * fs), target_idx,
                               block_idx]
                all_labels.append(target_idx)
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
    choose = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr, num_epochs, batch_size = 0.0005, 150, 64
    patience = 10
    delta = 1e-5
    n_splits = 5
    foldperf = {}
    for duration in np.arange(0.5, 1.2, 0.1):
        data_length = int(fs * duration)
        print(f'duration: {duration}')
        for subject_id in range(35):
            tic = time.time()
            subject = subject_id + 1
            print('————————————————————————————————————————\n')
            print('Subject-Fold for validation: S{} \n'.format(subject))
            for valid_block in range(1):
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

                model_path = '/path/to/your/project/model/pretrained/DSTCNet_pretrain_' + str(duration) + 's_S' + str(subject) + '.pth'
                model.load_state_dict(torch.load(model_path))
                new_model = model.cuda()

                loss_fn = nn.CrossEntropyLoss()
                if torch.cuda.is_available():
                    loss_fn = loss_fn.cuda()

                optimizer = torch.optim.Adam(new_model.parameters(), lr=lr, weight_decay=0.05)
                clr = CosineAnnealingLR(optimizer, T_max=10)

                one_fold_path = '/path/to/your/project/model/benchmark/DSTCNet_block1_' + str(duration) + 's_S' + str(subject) + '.pth'
                early_stopping = EarlyStopping(patience=patience, verbose=True, path=one_fold_path)

                total_train_step = 0
                total_test_step = 0
                history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
                for epoch in range(num_epochs):
                    new_model.train()
                    total_train_loss = 0
                    total_train_accuracy = 0
                    avg_train_acc = 0
                    for data in train_dataloader:
                        s_data, s_label = data
                        info = torch.Tensor(s_data)
                        info = info.to(device)
                        s_label = s_label.to(torch.long)
                        s_label = s_label.to(device)
                        outputs = new_model(info)
                        loss = loss_fn(outputs, s_label)
                        total_train_loss = total_train_loss + loss.sum().item()
                        accuracy = (outputs.argmax(axis=1) == s_label).sum()
                        total_train_accuracy = total_train_accuracy + accuracy

                        optimizer.zero_grad()
                        loss.sum().backward()
                        optimizer.step()

                        total_train_step = total_train_step + 1
                    avg_train_acc = float(total_train_accuracy / train_size * 100)
                    clr.step()

                    new_model.eval()
                    total_test_loss = 0
                    total_accuracy = 0
                    with torch.no_grad():
                        for data in test_dataloader:
                            s_data, s_label = data
                            info = torch.Tensor(s_data)
                            info = info.to(device)
                            s_label = s_label.to(torch.long)
                            s_label = s_label.to(device)
                            outputs = new_model(info)
                            loss = loss_fn(outputs, s_label)
                            total_test_loss = total_test_loss + loss.sum().item()
                            accuracy = (outputs.argmax(axis=1) == s_label).sum()
                            total_accuracy = total_accuracy + accuracy

                    if (epoch + 1) % 25 == 0:
                        print("------- Training Epoch {} -------".format(epoch + 1))
                        # print("output.shape is {}".format(outputs.shape))
                        print("Loss on the entire test set: {}".format(total_test_loss))
                        print("Overall test accuracy: {}%".format(total_accuracy / test_size * 100))
                    total_test_step = total_test_step + 1

                    if (epoch + 1) == num_epochs:
                        history['train_loss'].append(total_train_loss)
                        history['train_acc'].append(avg_train_acc)
                        history['test_loss'].append(total_test_loss)
                        history['test_acc'].append(float(total_accuracy / test_size * 100))
                        torch.save(model.state_dict(), one_fold_path)
                        print("Model saved. Subject:", subject)


                foldperf['fold_S{}'.format(subject)] = history
                print(history)
                print(time.time() - tic)
        time.sleep(50)