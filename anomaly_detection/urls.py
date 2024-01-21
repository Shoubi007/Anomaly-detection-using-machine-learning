from anomaly_detection import views  # 导入views模块
from django.conf.urls import url
from django.views.static import serve  #图片显示
from django.conf.urls.static import static
from django.urls import path, re_path
from Graduation_project import settings

urlpatterns = [
    path('', views.index,name='index_url'),
]