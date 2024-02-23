import gzip
import os
import random
import shutil
import tarfile
import zipfile
from multiprocessing import Pool
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class GoDatasetUtils:
    DATA_URLS = {
        "kgs": "https://u-go.net/gamerecords/",
        "kgs-4D": "https://u-go.net/gamerecords-4d/index.html",
        "cwi-minigo-9x9": "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/Minigo/",
        "cwi-seigen-9x9": "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/Go_Seigen/",
        "cwi-misc-9x9": "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/Misc/",
        # "cwi-nhk-9x9": "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/NHK/",
        # "cwi-propairgo-9x9": "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/ProPairgo/",
        # "cwi-computer-9x9": "https://homepages.cwi.nl/~aeb/go/games/games/other_sizes/9x9/computer/",
        # "cwi-all": "https://homepages.cwi.nl/~aeb/go/games/index.html",
    }
    EXTENSIONS = (".tar.gz", ".sgf", ".tgz")

    def __init__(self, name="kgs", dataset_dir="dataset"):
        self.name = name
        self.dataset_dir = dataset_dir

    def download_file(self, url, save_dir):
        filename = os.path.join(save_dir, os.path.basename(urlparse(url).path))
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return filename

    def get_links(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(self.EXTENSIONS):
                links.append(href)
        return links

    def download_link(self, link_info):
        url, save_dir = link_info
        try:
            filename = self.download_file(url, save_dir)
            print("Downloaded:", url)
            self.extract_file(filename, save_dir)
            os.remove(filename)  # Remove the compressed file after extraction
        except Exception as e:
            print("Failed to download:", url, "Error:", str(e))

    def extract_file(self, filename, save_dir):
        if filename.endswith(".tar.gz") or filename.endswith(".tgz"):
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(path=save_dir)
        elif filename.endswith(".zip"):
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(path=save_dir)

    def download_dataset(self):
        assert (
            self.name in self.DATA_URLS.keys()
        ), f"Available games: {list(self.DATA_URLS.keys())}. You can also paste your custom dataset to the dataset folder"
        url = self.DATA_URLS[self.name]
        save_dir = os.path.join(self.dataset_dir, self.name)
        os.makedirs(save_dir, exist_ok=True)
        links = self.get_links(url)
        link_infos = [(urljoin(url, link), save_dir) for link in links]
        with Pool() as pool:
            pool.map(self.download_link, link_infos)

    def get_games(self, no_of_samples, avoid):
        game_path = os.path.join(self.dataset_dir, self.name)
        sgf_files = []
        exit_loop = False
        paths = list(os.walk(game_path))
        random.shuffle(paths)
        for root, _, files in paths:
            if exit_loop:
                break
            for file in files:
                _file = os.path.join(root, file)
                if _file in avoid:
                    continue
                if file.endswith(".sgf"):
                    sgf_files.append(_file)
                    if len(sgf_files) >= no_of_samples:
                        exit_loop = True
                        break
        return sgf_files

    def check_dataset_exists(self):
        return os.path.isdir(os.path.join(self.dataset_dir, self.name))
