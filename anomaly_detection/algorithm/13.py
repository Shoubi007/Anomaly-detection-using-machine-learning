import csv
import csv
import pandas as pd
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
import sklearn as sk

number="0123456789"
path = "./CSVs/Tuesday-WorkingHours.pcap_ISCX.csv"
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
                if " – " in str(line):  ##  if there is "–" character ("–", Unicode code:8211) in the flow ,  it will be chanced with "-" character ( Unicode code:45).
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
try:
    string_features.remove('Label')#The "Label" property was removed from the list. Because it has to remain "categorical" for using with different machine learning approach.
except:
    print("error!")

labelencoder_X = preprocessing.LabelEncoder()

for ii in string_features:  ## In this loop, non-numeric (string and/or categorical) properties converted to numeric features.
    try:
        df[ii] = labelencoder_X.fit_transform(df[ii])
    except:
        df[ii] = df[ii].replace('Infinity', -1)
if 'faulty-Fwd Header Length' in df.columns:
    df = df.drop('faulty-Fwd Header Length',axis=1)


attack_or_not = []
for i in df["Label"]:  # it changes the normal label to "1" and the attack tag to "0" for use in the machine learning algorithm
    if i == "BENIGN":
        attack_or_not.append(1)
    else:
        attack_or_not.append(0)
df["Label"] = attack_or_not

y = df["Label"].values
del df["Label"]
X = df.values

X = np.float32(X)
X[np.isnan(X)] = 0
X[np.isinf(X)] = 0

forest = sk.ensemble.RandomForestRegressor(n_estimators=250, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
indices = np.argsort(importances)[::-1]
refclasscol = list(df.columns.values)
impor_bars = pd.DataFrame({'Features':refclasscol[0:20],'importance':importances[0:20]})
impor_bars = impor_bars.sort_values('importance',ascending=False).set_index('Features')
plt.rcParams['figure.figsize'] = (10, 5)
impor_bars.plot.bar();
fea_ture=[]
count=0
for i in impor_bars.index:
    fea_ture.append(str(i))
    count+=1
    if count==7:
        break
print(fea_ture)
print(type(fea_ture))


ml_list = {
    "Naive Bayes": GaussianNB(),
    "QDA": QDA(),
    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "ID3": DecisionTreeClassifier(max_depth=5, criterion="entropy"),
    "AdaBoost": AdaBoostClassifier(),
    "MLP": MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500),
    "Nearest Neighbors": KNeighborsClassifier(3)}

attack_list = ['DoS Hulk','PortScan','DDoS','DoS GoldenEye','FTP-Patator','SSH-Patator','DoS slowloris','DoS Slowhttptest','Bot','Infiltration','Heartbleed']
#load_csv
attack_or_not = []
for i in df["Label"]:  # it changes the normal label to "1" and the attack tag to "0" for use in the machine learning algorithm
    if i == "BENIGN":
        attack_or_not.append(-1)
    else:
        for j in len(attack_list):
            if i == attack_list[j]:
                attack_or_not.append(j)
df["Label"] = attack_or_not
y = df["Label"]  # this section separates the label and the data into two separate pieces, as Label=y Data=X
del df["Label"]
X = df[fea_ture]

#传统机器学习方法predict
for ii in ml_list:
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            # data (X) and labels (y) are divided into 2 parts to be sent to the machine learning algorithm (80% train,%20 test).
                                                            test_size=0.20,
                                                            random_state=repetition)
        clf = ml_list[ii]  # choose algorithm from ml_list dictionary
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
#lstm方法predict
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            # data (X) and labels (y) are divided into 2 parts to be sent to the machine learning algorithm (80% train,%20 test).
                                                            test_size=0.20,
                                                            random_state=repetition)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(1, activation=tf.nn.sigmoid))



