import multiprocessing
import random

import numpy as np
import torch
from sgfmill import sgf, sgf_moves
from torch.utils.data import Dataset
from tqdm import tqdm

from alphago.data.download_dataset import GoDatasetUtils

from . import _OnePlaneEncoder


class GoDataSet(Dataset):

    def __init__(
        self,
        encoder,
        game="kgs",
        no_of_games=1000,
        dataset_dir="dataset",
        seed=None,
        redownload=False,
        avoid=[],
    ):
        """
        encoder: encoder for the board
        game: use `list_all_datasets` for available datasets
        no_of_games: no of games to sample from
        dataset_dir: location of dataset
        seed: seed
        redownload: to download dataset again
        avoid (list): will avoid the files while sampling, used for test dataset
        """
        random.seed(seed)
        self.game = game
        if encoder == 'oneplane':
            self.encoder = _OnePlaneEncoder
        else:
            raise NotImplementedError(f"Available encoders: {['oneplane']}")
        self.no_of_games = no_of_games
        self.dataset_dir = dataset_dir
        self.total_frames = None
        self.datautils = GoDatasetUtils(name=self.game, dataset_dir=self.dataset_dir)
        if self.datautils.check_dataset_exists() == False or redownload:
            self.datautils.download_dataset()
        sgf_files = self.datautils.get_games(self.no_of_games, avoid)
        random.shuffle(sgf_files)
        self.games = sgf_files[: self.no_of_games]

        # Prepare the multiprocessing pool
        pool = multiprocessing.Pool()

        features = []
        labels = []

        for feature, label in tqdm(
            pool.imap(self.process_sgf_file, self.games), desc="loading games..."
        ):
            features.extend(feature)
            labels.extend(label)

        pool.close()
        pool.join()
        self.features = torch.tensor(np.array(features)).float()
        self.labels = torch.tensor(np.array(labels))

    def process_sgf_file(self, file):
        features = []
        labels = []
        with open(file) as f:
            contents = f.read().encode("ascii")
            game = sgf.Sgf_game.from_bytes(contents)
            board, plays = sgf_moves.get_setup_and_moves(game)
            for color, move in plays:
                if move is None:
                    continue
                row, col = move
                tp = self.encoder(board, move, color)
                features.append(tp[0])
                labels.append(tp[1])
                board.play(row, col, color)
        return features, labels

    @staticmethod
    def list_all_datasets():
        return GoDatasetUtils.DATA_URLS.keys()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
