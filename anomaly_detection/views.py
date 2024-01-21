from django.shortcuts import render
from django.conf import settings
import os
import anomaly_detection.algorithm.function as my_F
import warnings
import pandas as pd
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import h5py

# Create your views here.

# task_list = ['amazon', 'caltech', 'dslr', 'webcam']
# label_list = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
#               'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']

feature_list = ["Bwd Packet Length Std", "Flow IAT Min", "Fwd Packet Length Std", "Flow IAT Std",
                         "Total Length of Bwd Packets", "Flow Duration", "Flow IAT Mean"]

def index(request):
    if not request.session.get('is_login', None):
        request.session['is_log'] = '未登录'
    if request.method == 'POST':
        csv = request.FILES['csv']
        path = settings.MEDIA_ROOT
        isExists = os.path.exists(path)
        # 路径存在则返回true，不存在则返回false
        if isExists:
            print("目录已存在")
        else:
            os.mkdir(path)
            print("创建成功")
        csv_url = path + csv.name
        df = my_F.load_csv(csv_url)
        df2 = pd.read_csv(csv_url, usecols=['Flow ID'])
        flow_id = df2['Flow ID']
        prob = my_F.predict(feature_list, csv_url)
        prob = prob.flatten()
        count = 0
        for h in prob:
            if h == 0:
                count += 1
        out_df = pd.DataFrame({'Flow ID': flow_id, 'Anomaly 0/1': prob})
        out_df.to_csv(str(settings.MEDIA_ROOT) + 'out1.csv', index=False)
        return render(request, 'index1.html',  {'out_flag': 1,'anomaly_num':count,'out_num':prob.size})  # {'ano':ano,'att_class':att_class}

    return render(request, 'index1.html',{})