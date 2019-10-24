# main.py

from flask import Flask, render_template, Response, request

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pathlib
import os


model = ResNet50(weights='imagenet')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(file.filename)
    file.save(f)
    if pathlib.Path(r'static/img/chrt.jpeg').exists():
        os.remove(r'static/img/chrt.jpeg')
    
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
#    
    preds = model.predict(x)
    return render_template('index.html', pre=[list(s)[1] for s in decode_predictions(preds, top=10,)[0]])


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True, threaded=False,)