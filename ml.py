from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import sklearn.svm as svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import itertools

import dataset
from read_data import *
from utils import *
# import warnings
# warnings.filterwarnings("ignore")


def output(results):
    for model_name in results.keys():
        print("_" * 20 + model_name + "_" * 20)
        print(results[model_name])
        print("END" * 20)
        print()


if __name__ == "__main__":
    # word2vec_list = ["feature", "sentence_vector", "text"]  # "word_vector"
    word2vec_list = ["sentence_vector"]  # "word_vector"
    csv_path = "./all_data/features_all_data_0725.csv"
    vec_path = "./all_data/sentence_vectors_all.csv"
    results = {}

    model_list = [
        # MultinomialNB(),
        # svm.SVC(random_state=random_state),
        # RandomForestClassifier(random_state=random_state),
        LogisticRegression(max_iter=2000, random_state=random_state)
    ]
    # max_iter1000时会有警告，见 https://blog.csdn.net/qq_43391414/article/details/113144702

    for word2vec in word2vec_list:
        X, y = ReadData(csv_path, vec_path).run(method=word2vec)
        if word2vec == "text":
            X, _, _ = dataset.text_to_vector(X)
            X = X.numpy()
        datasets_set = dataset_split_kfold(X, y)

        for model in model_list:
            y_predict_res = []
            y_label_res = []
            test_index_lst = []

            # 开始跑代码
            for ind, ds in enumerate(datasets_set):
                model.fit(ds['X_train'], ds['y_train'])
                y_predict = model.predict(ds['X_test'])
                y_predict_res.extend(y_predict.tolist())
                y_label_res.extend(ds['y_test'])
                test_index_lst.extend(ds['test_index'])

            df_res = pd.DataFrame({'test_index': test_index_lst,
                                   'y_label': y_label_res,
                                   'y_pred': y_predict_res})
            df_res.to_excel("./output/{}_{}.xlsx".format(str(word2vec), str(model)), index=False)

            # precision_recall = classification_report(list(itertools.chain(*y_predict_res)), list(itertools.chain(*y_label_res)))
            # results[str(word2vec)+"\t"+str(model)] = precision_recall


    # output(results)




