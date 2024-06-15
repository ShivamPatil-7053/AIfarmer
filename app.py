import os
from flask import Flask, redirect, render_template, request, session
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import pickle
import requests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from keras.preprocessing import image
from PIL import Image

##

app = Flask(__name__)
app.secret_key = 'secret'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)


class User (db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable = False)
    email = db.Column(db.String(100), unique = True)
    message = db.Column(db.Text)
    phone = db.Column(db.String(20), nullable=False)


with app.app_context():
    db.create_all()



disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')
model = CNN.CNN(39)    
model.load_state_dict(torch.load("models/plant_disease_model_1_latest.pt"))
model.eval()
lr = keras.models.load_model('models/weights.hdf5')

class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, img_object):
        return self
    
    def transform(self, img_object):
        img_array = np.array(img_object)
        expanded = np.expand_dims(img_array, axis=0)
        return expanded

class Predictor(BaseEstimator, TransformerMixin):
    def fit(self, img_array):
        return self
    def predict(self, img_array):
        probabilities = lr.predict(img_array)
        predicted_class = ['P_Deficiency', 'Healthy', 'N_Deficiency', 'K_Deficiency'][probabilities.argmax()]
        return predicted_class


full_pipeline = Pipeline([('preprocessor', Preprocessor()), ('predictor', Predictor())])

# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))





def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index




@app.route('/')
def home_page():
    return render_template('home.html')












@app.route('/aboutus', methods = ['GET' , 'POST'])
def aboutus():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        phone = request.form.get('phone')
        user = User(name = name, email = email, message = message, phone= phone)
        db.session.add(user)
        db.session.commit()
        return redirect('/')

    return render_template('aboutus.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/ftips')
def farming_tips():
    return render_template('utube.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))


@app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)



@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        return render_template('crop-result.html', prediction=final_prediction, title=title)
       

    else:

        return render_template('try_again.html', title=title)




# Define route for the homepage
@app.route('/index1')
def home():
    return render_template('index1.html')

def output(full_pipeline, img):
    img = img.resize((224, 224))
    prediction = full_pipeline.predict(img)
    return prediction


# Define route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        img = Image.open(image_file)
        img = img.resize((224, 224))
        prediction = output(full_pipeline, img)
        return prediction







if __name__ == '__main__':
    app.run(debug=True)
