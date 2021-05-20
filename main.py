from os import walk
from pprint import pprint


def create_dict():
    data_dict = {}  # {author: {file: [data]}}
    for (dirpath, dirnames, _) in walk('./scaledata'):
        for name in dirnames:
            data_dict[name] = {}
            for (_, _, files) in walk(dirpath + '/' + name):
                for file in files:
                    data_dict[name][file] = []
        break
    return data_dict


def fill_data(data_dict):
    for (dirpath, dirnames, _) in walk('./scaledata'):
        for name in dirnames:
            for (_, _, files) in walk(dirpath + '/' + name):
                for file in files:
                    print(file)
                    with open(dirpath + '/' + name + '/' + file) as f:
                        data_dict[name][file] = f.read().splitlines()


if __name__ == '__main__':
    data = create_dict()
    fill_data(data)
    print(data['Dennis+Schwartz']['subj.Dennis+Schwartz'])
    #print(data)
