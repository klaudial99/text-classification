from os import walk
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer


def create_dict_authors():
    data_dict = {}  # {author: {file: [data]}}
    for (dirpath, dirnames, _) in walk('./scaledata'):
        for name in dirnames:
            data_dict[name] = {}
            for (_, _, files) in walk(dirpath + '/' + name):
                for file in files:
                    data_dict[name][file] = []
        break
    fill_data_authors(data_dict)
    return data_dict


def fill_data_authors(data_dict):
    for (dirpath, dirnames, _) in walk('./scaledata'):
        for name in dirnames:
            for (_, _, files) in walk(dirpath + '/' + name):
                for file in files:
                    print(file)
                    with open(dirpath + '/' + name + '/' + file) as f:
                        data_dict[name][file] = f.read().splitlines()


def create_dict_all():
    data_dict = {'id': [], 'label_3_class': [], 'label_4_class': [], 'rating': [], 'subj': []}

    for (dirpath, dirnames, _) in walk('./scaledata'):
        for name in dirnames:
            for (_, _, files) in walk(dirpath + '/' + name):
                for file in files:
                    with open(dirpath + '/' + name + '/' + file) as f:
                        data = f.read().splitlines()
                    if file.startswith('id'):
                        data_dict['id'].extend(data)
                    elif file.startswith('label.3'):
                        data_dict['label_3_class'].extend(data)
                    elif file.startswith('label.4'):
                        data_dict['label_4_class'].extend(data)
                    elif file.startswith('rating'):
                        data_dict['rating'].extend(data)
                    elif file.startswith('subj'):
                        data_dict['subj'].extend(data)

    return data_dict


if __name__ == '__main__':
    #data = create_dict_authors()
    #print(len(data['Dennis+Schwartz']['label.4class.Dennis+Schwartz']))
    #print(data)
    dict_all = create_dict_all()

    corpus = dict_all['subj']
    vectorizer = CountVectorizer()
