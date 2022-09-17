from flask import Flask, render_template, request
import os.path
from werkzeug.utils import secure_filename
from imageai.Detection import ObjectDetection
import os
import requests

app = Flask(__name__)

#global variables
detector = None

@app.before_first_request
def load_the_model():
    #download the model file
    model_URL = "https://github.com/OlafenwaMoses/ImageAI/releases/download/essentials-v5/resnet50_coco_best_v2.1.0.h5"
    response = requests.get(URL)
    open("./static/model/resnet50_coco_best_v2.1.0.h5", "wb").write(response.content)
    
    os.chdir('./static/model')
    execution_path = os.getcwd()

    global detector
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
    detector.loadModel()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/processImage', methods=['GET', 'POST'])
def processImage():
    if request.method == 'POST':
        selection_mode = request.form['selection_mode']
        if selection_mode == "2":#file
            
            f = request.files['img_file']
            f.save('./static/images/' + secure_filename(f.filename))
            
            #call the model to make predictions
            os.chdir('./static/images')
            images_path = os.getcwd()
            global detector
            detections = detector.detectObjectsFromImage(input_image=os.path.join(images_path, f.filename), output_image_path=os.path.join(images_path, "imagenew.jpg"))

            for eachObject in detections:
                print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
            
        return render_template('home.html')