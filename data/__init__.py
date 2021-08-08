import os
import torch
import multiprocessing as mp
import pathlib
from urllib.request import urlopen
from torch.utils.data import Dataset

from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_song_urls(url) -> list:
    """
    It returns a list of Zucchero's song lyrics urls.
    """
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    song_urls = [item["href"] for item in
                 soup.find('div',
                           {'class': 'article-content'}).ul.find_all(
                     href=True)]

    return song_urls


def get_song_text(song_url: str) -> list:
    """
    Given a song's url as input, it returns the text of the song scraped from
    the webpage.
    """
    page = urlopen(song_url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    condition1 = song_url in ['https://www.zucchero.it/eng/testi/alla-fine/',
                              'https://www.zucchero.it/eng/testi/e-un-peccato-morir/',
                              'https://www.zucchero.it/eng/testi/god-bless-the-child/']
    condition2 = song_url == 'https://www.zucchero.it/eng/testi/il-suono-della-domenica/'
    condition3 = song_url in ['https://www.zucchero.it/eng/testi/diamante/',
                              'https://www.zucchero.it/eng/testi/soldati-nella-mia-citta/',
                              'https://www.zucchero.it/eng/testi/spicinfrin-boy/',
                              'https://www.zucchero.it/eng/oltre-le-rive/',
                              'https://www.zucchero.it/eng/alla-fine/',
                              'https://www.zucchero.it/eng/shake/']
    try:
            if condition1:
                song_text = [item.get_text() for item in
                     soup.find('div', {'class': 'article-content'}).find_all("p")[
                     1:] if '<em>' not in str(item)]
            elif condition2:
                song_text = [item.get_text() for item in
                     soup.find('div', {'class': 'article-content'}).find_all("p")[
                     :-1] if '<em>' not in str(item)]
            elif condition3:
                song_text = ['']
            else:
                song_text = [item.get_text() for item in
                     soup.find('div', {'class': 'article-content'}).find_all("p")
                     if '<em>' not in str(item)]
            return song_text
    except:
        return ['']


def get_sugar_lyrics(dataset_dir: pathlib.Path):
    url = "https://www.zucchero.it/eng/lyrics/"
    if (not os.listdir(dataset_dir)) | (all(s == '.ipynb_checkpoints' for s in os.listdir(dataset_dir))):
        song_urls = get_song_urls(url)
        # Step 1: Init multiprocessing.Pool()
        pool = mp.Pool(mp.cpu_count())
        # Step 2: `pool.map` the `get_song_text()`
        songs = pool.map(get_song_text, [url for url in song_urls])
        songs = [song for songs_sublist in songs for song in songs_sublist] # Flatten
        # Step 3: Don't forget to close
        pool.close()
        # Step 4: Save the data into a .txt file
        with open(dataset_dir / 'sugar_lyrics.txt', 'w', encoding='utf-8') as f:
            for song in songs:
                f.write(song)


