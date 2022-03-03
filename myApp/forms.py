from django.forms import ModelForm
from myApp.models import Image

class ImageForm(ModelForm):
    class Meta:
        model = Image
        fields = ['image']
