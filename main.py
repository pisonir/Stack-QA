from Project import Project
from data import get_sugar_lyrics

if __name__ == '__main__':
    project = Project()
    # get the data and store it in the dataset_dir  as .txt files. Already
    # split into train and test sets
    get_sugar_lyrics(project.dataset_dir)

    # TODO maybe add the hyperparams config with also the train test ratio