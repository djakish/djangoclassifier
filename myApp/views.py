from django.shortcuts import render
from .forms import ImageForm
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from keras.models import Sequential
import os
from django.conf import settings
import numpy
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Create your views here.
def image_upload_view(request):
    """Process images uploaded by users"""
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            img_obj.save()
            img_obj.result = predict(img_obj.image.path)
            return render(request, 'index.html', {'form': form, 'img_obj': img_obj})
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})

def predict(picturePath):
    model_new = load_model("myModel.h5")
    test_image= image.load_img( picturePath, target_size = (32, 32))
    test_image = image.img_to_array(test_image)
    test_image = numpy.expand_dims(test_image, axis = 0)
    test_image = test_image.reshape(1,32, 32,3)
    return classes[numpy.argmax(model_new.predict(test_image),axis=1)[0]]
