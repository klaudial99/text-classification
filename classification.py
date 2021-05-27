from numpy import argmax, mean
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from main import create_dict_all

if __name__ == '__main__':

    dict_all = create_dict_all()

    x = dict_all["subj"]
    y_3 = dict_all["label_3_class"]
    y_4 = dict_all["label_4_class"]

    model = make_pipeline(CountVectorizer(stop_words='english'), SelectKBest(chi2, k=5000), MultinomialNB())
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

    val = cross_validate(model, x, y_3, cv=cv)
    print(val['test_score'])
    # for k, v in val.items():
    #     print(k, ':', v)

    best_index = argmax(val['test_score'])
    print("INDEX WITH BEST ACCURACY:", best_index)

    print("BEST ACCURACY: {0:.0%}".format(val['test_score'][best_index]))
    print("AVG ACCURACY: {0:.0%}".format(mean(val['test_score'])))

    generator_list_index = list(cv.split(x, y_3))

    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    for i in generator_list_index[best_index][0]:
        X_train.append(x[i])
        Y_train.append(y_3[i])

    for i in generator_list_index[best_index][1]:
        X_test.append(x[i])
        Y_test.append(y_3[i])

    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)
    # print(y_pred)

    score = model.score(X_test, Y_test)
    print(score)

    print(model.predict(["Peak filmmaking on the grandest scale and THE most monumentally produced, impeccably designed, and harrowingly epic film I have ever seen. 'Titanic' will never not leave me utterly floored. It's been several months since I last watched this, so I'm trying to remain calm…but it just means so damn much when you cherish the commitment and physical craft a production like this takes and how miraculous it is, not only that Cameron's film turned out this spectacularly, but that we will likely never see an undertaking of this caliber ever again. That deafening mechanical roar when the all the lights go out, like a groaning beast from the deep....chills every time."]))
    print(model.predict(["Mr. Bean's Holiday is better than the first film, but only because it focused more on the character and less on trying to have a plot. The film is ridiculous just as it should be, but there is a creepiness about Bean in some moments that I couldn't get past. Rowan Atkinson is still funny though and still keeps the character the same, its just that at some parts I wondered how weird they were trying to make him. I also still feel that they cannot make a 90 minute movie about Mr. Bean and keep it interesting all the way through, because honestly I was bored at many times. Its pretty much the same as the first film, other than doing the smart thing and getting rid of a story and just doing a series of Bean's trouble makings."]))