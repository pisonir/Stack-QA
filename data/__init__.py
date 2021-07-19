import os
import multiprocessing as mp
from urllib.request import urlopen

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


def get_song_text(song_url: str) -> str:
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
            song_text = '. '.join(
                [item.get_text().replace('\n', '. ') for item in
                 soup.find('div', {'class': 'article-content'}).find_all("p")[
                 1:] if '<em>' not in str(item)])
        elif condition2:
            song_text = '. '.join(
                [item.get_text().replace('\n', '. ') for item in
                 soup.find('div', {'class': 'article-content'}).find_all("p")[
                 :-1] if '<em>' not in str(item)])
        elif condition3:
            song_text = ''
        else:
            song_text = '. '.join(
                [item.get_text().replace('\n', '. ') for item in
                 soup.find('div', {'class': 'article-content'}).find_all("p")
                 if '<em>' not in str(item)])
        return song_text
    except:
        return ''


def build_dataset(songs: list, filename: str):
    """
    Write a .txt file where each song includes the special tokens <BOS> at the
    beginning and <EOS> at the end.
    """
    f = open(f'{filename}.txt', 'w', encoding='utf-8')
    data = ''
    for song in songs:
        song = str(song).strip()
        bos_token = '<BOS>'
        eos_token = '<EOS>'
        data += bos_token + ' ' + song + ' ' + eos_token + '\n'

    f.write(data)


def get_sugar_lyrics(dataset_dir):
    url = "https://www.zucchero.it/eng/lyrics/"
    if not os.listdir(dataset_dir):
        song_urls = get_song_urls(url)
        # Step 1: Init multiprocessing.Pool()
        pool = mp.Pool(mp.cpu_count())
        # Step 2: `pool.map` the `get_song_text()`
        songs = pool.map(get_song_text, [url for url in song_urls])
        # Step 3: Don't forget to close
        pool.close()
        x_train, x_test = train_test_split(songs, test_size=0.1,
                                           random_state=42)
        build_dataset(x_train, dataset_dir / 'train')
        build_dataset(x_test, dataset_dir / 'test')
