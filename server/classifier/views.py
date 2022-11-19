from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import base64
import numpy as np
from PIL import Image
import io
import ml.testing as ml

model = ml.get_model()
# Create your views here.
@csrf_exempt
def binary_classifier(request):
    base64img = json.loads(request.body)["image"]
    image = Image.open(io.BytesIO(base64.decodebytes(str.encode(base64img[22:]))))
    # image_np = np.array(image)
    
    # Classify the image here
    # ---
    
    output = ml.infer(image)
    
    #####################################
    #### Send the output to the OpenStreet
    #####################################
    
    return HttpResponse("success", json)
