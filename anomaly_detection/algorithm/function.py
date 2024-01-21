import numpy as np
import anomaly_detection.algorithm.parameter as Para
import csv
import pandas as pd
import tensorflow as tf
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
import sklearn as sk
import joblib
import Graduation_project.settings as settings
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
import h5py

def load_csv(path): #上传csv，数据预处理
    number = "0123456789"
    #path = "./CSVs/Tuesday-WorkingHours.pcap_ISCX.csv"
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        row = next(reader)
        main_labels = list(row)
    main_labels2 = []
    for i in main_labels:
        i = i.strip()
        main_labels2.append(i)

    main_labels = (",".join(i for i in main_labels2))
    main_labels = main_labels + "\n"

    ths = open("processed.csv", "w")
    ths.write(main_labels)
    with open(path, "r") as file:
        while True:
            try:
                line = file.readline()
                if line[0] in number:  # this line eliminates the headers of CSV files and incomplete streams .
                    if " – " in str(
                            line):  ##  if there is "–" character ("–", Unicode code:8211) in the flow ,  it will be chanced with "-" character ( Unicode code:45).
                        line = (str(line).replace(" – ", " - "))
                    line = (str(line).replace("inf", "0"))
                    line = (str(line).replace("Infinity", "0"))

                    line = (str(line).replace("NaN", "0"))

                    ths.write(str(line))
                else:
                    continue
            except:
                break
    ths.close()

    df = pd.read_csv("processed.csv", low_memory=False)
    df = df.fillna(0)

    string_features = ["Flow Bytes/s", "Flow Packets/s"]
    for ii in string_features:  # Some data in the "Flow Bytes / s" and "Flow Packets / s" columns are not numeric. Fixing this bug in this loop
        df[ii] = df[ii].replace('Infinity', -1)
        df[ii] = df[ii].replace('NaN', 0)
        number_or_not = []
        for iii in df[ii]:
            try:
                k = int(float(iii))
                number_or_not.append(int(k))
            except:
                number_or_not.append(iii)
        df[ii] = number_or_not

    string_features = []
    for j in main_labels2:  # In this section, non-numeric (string and / or categorical) properties (columns) are detected.
        if df[j].dtype == "object":
            string_features.append(j)

    labelencoder_X = preprocessing.LabelEncoder()

    for ii in string_features:  ## In this loop, non-numeric (string and/or categorical) properties converted to numeric features.
        try:
            df[ii] = labelencoder_X.fit_transform(df[ii])
        except:
            df[ii] = df[ii].replace('Infinity', -1)
    if 'faulty-Fwd Header Length' in df.columns:
        df = df.drop('faulty-Fwd Header Length', axis=1)
    return df

    # # load image
    # img = skimage.io.imread(path)
    # img = img / 255.0
    # assert (0 <= img).all() and (img <= 1.0).all()
    # # print "Original Image Shape: ", img.shape
    # # we crop image from center
    # short_edge = min(img.shape[:2])
    # yy = int((img.shape[0] - short_edge) / 2)
    # xx = int((img.shape[1] - short_edge) / 2)
    # crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # # resize to 224, 224
    # resized_img = skimage.transform.resize(crop_img, (224, 224))
    # return resized_img
# def extract_feature_vgg19(img_url):
#     with tf.Session() as sess:
#         image = tf.placeholder("float", [None, 224, 224, 3])
#         sess.run(tf.global_variables_initializer())
#         vgg = vgg19.Vgg19()
#
#         with tf.name_scope("content_vgg"):
#             vgg.build(image)
#         try:
#             img = load_image(img_url).reshape((1, 224, 224, 3))
#         except ValueError:
#             print('There is an error occuring for image: ' + img_url)
#         feature = sess.run(vgg.fc7, feed_dict={image: img})
#         feature = np.reshape(feature, [1, -1])
#         return feature

# def  extract_feature_RandomForest(csv_url): #随机森林特征选择
#     try:
#         df = load_csv(csv_url)
#     except ValueError:
#         print('There is an error occuring for csv file: ' + csv_url)
#     attack_or_not = []
#     for i in df[
#         "Label"]:  # it changes the normal label to "1" and the attack tag to "0" for use in the machine learning algorithm
#         if i == "BENIGN":
#             attack_or_not.append(1)
#         else:
#             attack_or_not.append(0)
#     df["Label"] = attack_or_not
#
#     y = df["Label"].values
#     del df["Label"]
#     X = df.values
#
#     X = np.float32(X)
#     X[np.isnan(X)] = 0
#     X[np.isinf(X)] = 0
#
#     forest = sk.ensemble.RandomForestRegressor(n_estimators=250, random_state=0)
#     forest.fit(X, y)
#     importances = forest.feature_importances_
#     std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#                  axis=0)
#     indices = np.argsort(importances)[::-1]
#     refclasscol = list(df.columns.values)
#     impor_bars = pd.DataFrame({'Features': refclasscol[0:20], 'importance': importances[0:20]})
#     impor_bars = impor_bars.sort_values('importance', ascending=False).set_index('Features')
#     plt.rcParams['figure.figsize'] = (10, 5)
#     impor_bars.plot.bar();
#     fea_ture = []
#     count = 0
#     for i in impor_bars.index:
#         fea_ture.append(str(i))
#         count += 1
#         if count == 7:
#             break
#     print("select_feature:"+fea_ture)
#     return fea_ture


#def LSTM_train 显示LSTM模型的训练结果


def predict(input,csv_url): #二分类问题
    path = settings.PARA_PATH
    joblib.dump('pipeline', 'mymodel.pkl')
    model = tf.keras.models.load_model(path + 'lstm.h5', compile = False)
    #model = joblib.load(path + 'lstm.joblib')
    feature_list = input
    try:
        df = load_csv(csv_url)
    except ValueError:
        print('There is an error occuring for csv file: ' + csv_url)

    X = df[feature_list]
    X = np.array(X)
    X_test = X.reshape((X.shape[0], 1, X.shape[1]))

    predict = model.predict(X_test)
    fl = []
    for ii in predict:
        if ii < 0.50:
            fl.append(0)
        else:
            fl.append(1)
    prob = pd.DataFrame(fl, columns=['Anomaly 0/1'])
    prob = np.array(prob)
    print(type(prob))
    print(prob.shape)
    prob = prob.flatten()
    print(prob.shape)
    return prob
    #class_id正常or异常（0/1） 该分类的预测概率






#
#
# def predict(input, task_id):
#     hidden_feature = np.tanh(np.dot(input, Para.input_hidden_weights))
#     temp_hidden_feature = np.concatenate([hidden_feature, Para.task_embedding_vectors[task_id].reshape([1, -1])], 1)
#     probits_softmax = []
#     for j in range(Para.num_class):
#         temp = np.concatenate([temp_hidden_feature, Para.class_embedding_vectors[task_id * Para.num_task + j].reshape([1, -1])], 1)
#         probit_softmax = np_softmax(np.dot(temp, Para.hidden_output_weight[task_id]))
#         probits_softmax.append(probit_softmax)
#     probits_softmax = np.squeeze(np.concatenate([probits_softmax], 0))
#     diagonal = []
#     for j in range(Para.num_class):
#         diagonal.append(probits_softmax[j][j])
#     class_id = np.argmax(diagonal)
#     return class_id, probits_softmax[class_id][class_id]  #样本的预测类别 改分类的预测概率
#
#
#
# def np_softmax(x):
#     """Compute the softmax in a numerically stable way."""
#     x = x - np.max(x)
#     exp_x = np.exp(x)
#     softmax_x = exp_x / np.sum(exp_x)
#     return softmax_x