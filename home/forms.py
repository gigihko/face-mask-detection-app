from django.forms import ModelForm

from . models import *


class FIleForm(ModelForm):
    class Meta:
        model = Image
        fields = ['file']