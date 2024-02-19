import multiprocessing
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from alphago.data.download_dataset import GoDatasetUtils
from alphago.data.sgf import Sgf_game, get_handicap
from alphago.env.go_board import Move
from alphago.env.gotypes import Point


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
        self.encoder = encoder
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
        self.features = torch.tensor(np.array(features))
        self.labels = torch.tensor(np.array(labels))

    def process_sgf_file(self, file):
        features = []
        labels = []
        with open(file, "r") as f:
            game_string = "".join(f.readlines())
        sgf = Sgf_game.from_string(game_string)

        game_state, first_move_done = get_handicap(sgf)

        for item in sgf.main_sequence_iter():
            color, move_tuple = item.get_move()
            point = None
            if color is not None:
                if move_tuple is not None:
                    row, col = move_tuple
                    point = Point(row + 1, col + 1)
                    move = Move.play(point)
                else:
                    move = Move.pass_turn()
                if first_move_done and point is not None:
                    features.append(self.encoder.encode(game_state))
                    labels.append(self.encoder.encode_point(point))
                game_state = game_state.apply_move(move)
                first_move_done = True

        return features, labels

    @staticmethod
    def list_all_datasets():
        return GoDatasetUtils.DATA_URLS.keys()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
