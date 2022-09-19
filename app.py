from flask import Flask, render_template, request, redirect, url_for
import os.path
from werkzeug.utils import secure_filename
from imageai.Detection import ObjectDetection
import os
import requests  

app = Flask(__name__)

#global variables
detector = None
input_img_src = None
output_img_src = None
recap = {}

@app.before_first_request
def load_the_model():
    #download the model file
    model_URL = "https://github.com/OlafenwaMoses/ImageAI/releases/download/essentials-v5/resnet50_coco_best_v2.1.0.h5"
    response = requests.get(model_URL)
    open("./static/resnet50_coco_best_v2.1.0.h5", "wb").write(response.content)
    
    os.chdir('./static')
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
  def processImageHelper():
    #call the model to make predictions
    images_path = os.getcwd()
    global detector
    global recap
    detections = detector.detectObjectsFromImage(input_image=os.path.join(images_path, 'image.jpg'), output_image_path=os.path.join(images_path, "imagenew.jpg"))

    recap = {}
    for eachObject in detections:
      name = eachObject["name"]
      probability = eachObject["percentage_probability"]
      if(probability >= 50):
        recap[name] = recap.get(name, 0) + 1

    global input_img_src
    global output_img_src
    input_img_src = url_for('static', filename='image.jpg')
    output_img_src = url_for('static', filename="imagenew.jpg")

  if request.method == 'POST':
    filename = None
    selection_mode = request.form['selection_mode']
    if selection_mode == "1":#URL
      img_URL = request.form['image_url']
      response = requests.get(img_URL)
      open('./image.jpg', 'wb').write(response.content)
      filename = 'image.jpg'

    if selection_mode == "2":#file
      f = request.files['img_file']
      f.save('./image.jpg')
      filename = f.filename
        
    processImageHelper()
      
  return redirect(url_for('result'))

@app.route('/result')
def result():
  global input_img_src
  global output_img_src
  return render_template('result.html', input_img_src=input_img_src, output_img_src=output_img_src, recap=recap)