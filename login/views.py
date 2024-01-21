from django.shortcuts import render
from django.shortcuts import redirect
from . import models
from . import forms
# Create your views here.


def login(request):
    if request.session.get('is_login', None):  # 不允许重复登录
        return redirect('/login/index/')
    if request.method == 'POST':
        login_form = forms.UserForm(request.POST)
        message = '请检查填写的内容！'
        if login_form.is_valid():
            username = login_form.cleaned_data.get('username')
            password = login_form.cleaned_data.get('password')
            try:
                user = models.User.objects.get(name=username)
            except:
                message = '用户不存在！'
                return render(request, 'login.html', locals())
            if user.password == password:
                request.session['is_login'] = True
                request.session['is_log'] = '已登录'
                request.session['user_id'] = user.id
                request.session['user_name'] = user.name
                return redirect('/login/index/')
            else:
                message = '密码不正确！'
                return render(request, 'login.html', locals())

        else:
            return render(request, 'login.html', locals())

    login_form = forms.UserForm()
    return render(request, 'login.html', locals())

def register(request):
    if request.session.get('is_login', None):
        return redirect('/login/index/')

    if request.method == 'POST':
        register_form = forms.RegisterForm(request.POST)
        message = "请检查填写的内容！"
        if register_form.is_valid():
            username = register_form.cleaned_data.get('username')
            email = register_form.cleaned_data.get('email')
            password = register_form.cleaned_data.get('password')
            same_name_user = models.User.objects.filter(name=username)
            if same_name_user:
                message = '用户名已经存在!'
                return render(request, 'register.html', locals())
            same_email_user = models.User.objects.filter(email=email)
            if same_email_user:
                message = '该邮箱已被注册！'
                return render(request, 'register.html', locals())

            new_user = models.User()
            new_user.name = username
            new_user.password = password
            new_user.email = email
            new_user.save()

            return redirect('/login/login/')
        else:
            return render(request, 'register.html', locals())
    register_form = forms.RegisterForm()
    return render(request, 'register.html', locals())

def logout(request):
    if not request.session.get('is_login', None):
        # 如果本来就未登录，也就没有登出一说
        return redirect("/login/login/")
    del request.session['is_login']
    del request.session['user_id']
    del request.session['user_name']
    return redirect("/login/index/")

def index(request):
    if not request.session.get('is_login', None):
        request.session['is_log'] = '未登录'
    return render(request, 'index.html', {})
