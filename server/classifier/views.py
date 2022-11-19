from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import base64
import numpy as np
from PIL import Image
import io

# Create your views here.
@csrf_exempt
def binary_classifier(request):
    base64img = json.loads(request.body)["image"]
    image = Image.open(io.BytesIO(base64.decodebytes(str.encode(base64img[22:]))))
    image_np = np.array(image)
    # Classify the image here
    # ---
    return HttpResponse("success")
