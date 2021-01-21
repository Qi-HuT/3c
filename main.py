from pathlib import Path
from process_data import load_data
import torch.utils.data as Data


def main():
    # dir_path = Path('/home/g19tka13/Downloads/data/3C')
    # data_path = dir_path / 'taskA/train.csv'
    dataset = load_data()
    print(dataset)
    data_iter = Data.DataLoader(dataset, batch_size=10, shuffle=True)
    for item in data_iter:
        print(item[0].size())


if __name__ == '__main__':
    main()
