from django import forms


class UserForm(forms.Form):
    username = forms.CharField(label="用户名", max_length=128)
    password = forms.CharField(label="密码", max_length=256, widget=forms.PasswordInput)


class RegisterForm(forms.Form):
    username = forms.CharField(label="用户名", max_length=128)
    password = forms.CharField(label="密码", max_length=256)
    email = forms.EmailField(label="邮箱地址")