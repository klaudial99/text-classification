from os import walk
from pprint import pprint
from typing import List

from numpy import take, average
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


def create_dict_for_many_classes(dict_all_data, classes_amount):
    data_class = {}

    for x in range(classes_amount):
        data_class[str(x)] = []

    name = 'label_' + str(classes_amount) + '_class'
    for i in range(len(list(dict_all_data.values())[0])): #len of first value
        data_class[dict_all_data[name][i]].append(dict_all_data['subj'][i])  # {class: [opinions]}

    return data_class


def analyze(analyze_set: List):

    vectorizer = CountVectorizer(stop_words='english')  # collection of text documents -> matrix of token counts
    # print(vectorizer.get_stop_words())
    X = vectorizer.fit_transform(analyze_set)  # learn the vocabulary dictionary and return document-term matrix
    features = vectorizer.get_feature_names()  # array mapping from feature integer indices to feature name
    # print(features)
    print('TOTAL DIFFERENT WORDS AMOUNT:', len(features))
    print('TOTAL OPINIONS AMOUNT:', len(X.toarray()))

    ######################## WORDS COUNTER

    words_counter = {}
    words_len = []
    for i, _ in enumerate(features):  # add keys
        words_counter[i] = 0
    for opinion in X.toarray():
        words_len.append(sum(opinion))  # words in opinion
        for i, word_count in enumerate(opinion):
            words_counter[i] += word_count  # count word occurrences

    for i, feature in enumerate(features):
        words_counter[feature] = words_counter.pop(i)  # change keys to words

    words_counter = {k: v for k, v in sorted(words_counter.items(), key=lambda item: item[1], reverse=True)[:10]}  # sort and take top 10
    print(words_counter)

    ########################## LENGTH COUNTER

    min_len = min(words_len)
    max_len = max(words_len)
    avg_len = average(words_len)

    print('min', min_len)
    print('max', max_len)
    print('avg', avg_len)


if __name__ == '__main__':

    dict_all = create_dict_all()

    classes_3 = create_dict_for_many_classes(dict_all, 3)
    # for k, v in classes_3.items():
    #     print(k, len(v))

    classes_4 = create_dict_for_many_classes(dict_all, 4)
    # for k, v in classes_4.items():
    #     print(k, len(v))

    for k, v in classes_3.items():
        print('---3 CLASSES, CLASS: ' + k + '---')
        analyze(v)
        print()

    for k, v in classes_4.items():
        print('---4 CLASSES, CLASS: ' + k + '---')
        analyze(v)
        print()

    print('---ALL---')
    analyze(dict_all['subj'])
