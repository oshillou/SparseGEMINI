import itertools
from dataclasses import dataclass

from torch.utils.data import Dataset, DataLoader, TensorDataset, BatchSampler, RandomSampler, SequentialSampler
from torch.utils.data._utils.collate import default_collate
from torchvision import datasets, transforms
import torch
import ot

from io_utils import load_csv


@dataclass
class HistoryItem:
    lambda_: float
    train_gemini: float
    train_reg: float
    val_gemini: float
    val_reg: float
    selected: torch.BoolTensor
    epochs: int

    def log(self):
        print(
            f"{self.epochs} epochs, Total is {self.val_reg - self.val_gemini}\t(GEMINI: {self.val_gemini:.3e}, L1: {self.val_reg:.3e})")


class AffinityDataset(Dataset):
    def __init__(self, X, similarities):
        super().__init__()

        self.X = X
        self.similarities = similarities

    def __getitem__(self, idx):
        if type(idx) == int:
            # When returning a simple index, we only return the similarity to self.
            return self.X[idx], self.similarities[idx, idx]

        batch = []
        for index in idx:
            batch += [(self.X[index], self.similarities[:, index][idx])]
        return default_collate(batch)

    def __len__(self):
        return len(self.X)

    def get_loader(self, batch_size=-1, shuffle=False):
        if batch_size == -1:
            batch_size = len(self)
        if shuffle:
            sampler = BatchSampler(RandomSampler(self), batch_size=batch_size, drop_last=False)
        else:
            sampler = BatchSampler(SequentialSampler(self), batch_size=batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        return batch[0]

    def get_input_shape(self):
        return self.X.shape[-1]


class DynamicAffinityDataset(Dataset):
    def __init__(self, X, affinity_function):
        super().__init__()

        self.X = X
        self.affinity = affinity_function
        sample = self._get_elem_from_X(0)
        self.mask = torch.ones(1, *sample.shape)

        self.pre_compute_similarities()

    def pre_compute_similarities(self):
        print("Pre-computing similarities")
        N = len(self)
        self.similarities = torch.zeros(N, N)

        B = max(100, min(5000, N // 5))  # Clamp to a small division of the dataset for all affinity computations
        if B >= N:
            masked_X = self.X * self.mask
            self.similarities = self.affinity(masked_X, masked_X)
        else:
            # dataloader = DataLoader(TensorDataset(self.X), shuffle=False, batch_size=B)
            # iterator = itertools.combinations_with_replacement(enumerate(dataloader),2)
            #
            # for (i, batch1), (j, batch2) in iterator:
            #     i_min, i_max = B * i, min(B * (i + 1), N)
            #     j_min, j_max = B * j, min(B * (j + 1), N)
            #
            #     # delta is an existing function, so compute similarity
            #     batch_similarity = self.affinity(batch1[0], batch2[0])
            #     # Complete symmetry of this similarities
            #     self.similarities[i_min:i_max, j_min:j_max] = batch_similarity
            #     self.similarities[j_min:j_max, i_min:i_max] = batch_similarity.T
            iterator = itertools.combinations_with_replacement(range(len(self) // B + 1), 2)
            for i, j in iterator:
                i_min, i_max = B * i, min(B * (i + 1), N)
                j_min, j_max = B * j, min(B * (j + 1), N)
                if i_min >= len(self) or j_min >= len(self):
                    continue
                batch1 = self._get_elem_from_X(list(range(i_min, i_max)))
                batch2 = self._get_elem_from_X(list(range(j_min, j_max)))
                batch_similarity = self.affinity(batch1, batch2)
                # Complete symmetry of this similarities
                self.similarities[i_min:i_max, j_min:j_max] = batch_similarity
                self.similarities[j_min:j_max, i_min:i_max] = batch_similarity.T
        print("Similarities computed")

    def update_mask(self, new_mask):
        self.mask = new_mask.clone()
        self.pre_compute_similarities()

    def reset_mask(self):
        self.mask = torch.ones(self.mask.shape)

    def _get_elem_from_X(self, idx):
        if type(idx) == int:
            sub_X = self.X[idx]
            if type(sub_X) == tuple:
                # We are facing an iterable dataset that yields multiple element. Only keep the first one considered to be the data.
                sub_X = sub_X[0]
            return sub_X
        else:
            sub_X = []
            for i in idx:
                sub_X += [torch.unsqueeze(self._get_elem_from_X(i), 0)]
            return torch.cat(sub_X, 0)

    def __getitem__(self, idx):
        sub_X = self._get_elem_from_X(idx) * self.mask
        if type(idx) == int:
            sub_X = sub_X.view((1, -1))
            return sub_X, self.similarities[idx, idx]

        batch = []
        for i in range(len(idx)):
            batch += [(sub_X[i], self.similarities[idx[i], idx])]

        return default_collate(batch)

    def __len__(self):
        return len(self.X)

    def get_loader(self, batch_size=-1, shuffle=False):
        if batch_size == -1:
            batch_size = len(self)
        if shuffle:
            sampler = BatchSampler(RandomSampler(self), batch_size=batch_size, drop_last=False)
        else:
            sampler = BatchSampler(SequentialSampler(self), batch_size=batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        return batch[0]

    def get_input_shape(self):
        X = self._get_elem_from_X(0)
        return X.shape[-1]


def get_affinity(args):
    D = load_csv(args.metric)
    return torch.Tensor(D)


def get_dataset(args):
    if args.csv is not None:
        return torch.Tensor(load_csv(args.csv))
    else:
        # We are facing a deep learning dataset
        if args.data == "mnist":
            return get_mnist(args)
        elif args.data == "fashionmnist":
            return get_fashion_mnist(args)


def get_affinity_function(args):
    if args.dynamic_metric == "sqeuclidean" and args.gemini == "wasserstein":
        return ot.dist
    elif args.dynamic_metric == "euclidean" and args.gemini == "mmd":
        return lambda x, y: x @ y.T
    elif args.dynamic_metric == "euclidean" and args.gemini == "wasserstein":
        return lambda x, y: ot.dist(x, y, metric="euclidean")
    elif args.static_metric == "sqeuclidean" and args.gemini == "wasserstein":
        return ot.dist
    elif args.static_metric == "euclidean" and args.gemini == "mmd":
        return lambda x, y: x @ y.T
    elif args.static_metric == "euclidean" and args.gemini == "wasserstein":
        return lambda x, y: ot.dist(x, y, metric="euclidean")
    else:
        assert False, f"Unavailable combination: mmd and sqeuclidean distance"


def get_mnist(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    return datasets.MNIST(args.data_path, transform=transform, train=True)


def get_fashion_mnist(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    return datasets.FashionMNIST(args.data_path, transform=transform, train=True)
