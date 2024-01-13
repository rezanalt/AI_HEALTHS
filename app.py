from flask import Flask,render_template,request , redirect , url_for ,jsonify , session
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import logging
import base64
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,classification_report,precision_score,roc_curve
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from mtranslate import translate


app = Flask('__name__')
CORS(app)

@app.route('/', methods=['GET'])
def op():
    return render_template('app.html')

"""Nevus Detection Part"""

@app.route('/upload_photo', methods=['POST'])
def upload_photo_nevus():
    photo_data = request.json['photo'] 

    
    imgdata = base64.b64decode(photo_data.split(',')[1])
    filename = 'static/uploaded_photo.png'  

    
    with open(filename, 'wb') as f:
        f.write(imgdata)
        
    return render_template('Nevus.html')
@app.route('/Nevus.html', methods = ['POST'])
def predict_nevus():
    IMAGE_SIZE = 224
    model = tf.keras.models.load_model('models/Nevus.h5')
    imagefile = request.files['imagefile']
    
    filename= imagefile.filename
    image_pth = "./static/" + filename
    imagefile.save(image_pth)
    image = tf.keras.preprocessing.image.load_img(image_pth , target_size=(IMAGE_SIZE,IMAGE_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    test_x = np.expand_dims(image , axis=0)
    predictions = model.predict(test_x)
    confidence = np.max(predictions)*100
    
    class_names = ['Melanoma(Kanserli Ben)', 'Nevus(Normal Ben)', 'Seborrheic_Keratosis(İyi Huylu Et Büyümesi)']

    
    prediction = class_names[np.argmax(predictions)]

    if(confidence > 50):
        confidence = confidence
    else:
        confidence = "Tahmin oranı çok düşük"
        prediction = "Tahmin oranı çok düşük"

    image1 = Image.open(image_pth)
    image1 = image1.resize([500,500])
    
    confidence = str(confidence)
    confidence = confidence[:8]
    
    return render_template('/Nevus.html' , prediction = prediction , confidence = confidence, filename = filename) 


@app.route('/upload_photo' , methods=['GET'])
def predict_nevus2():
    image_pth = "./static/uploaded_photo.png" 
    IMAGE_SIZE = 224
    model = tf.keras.models.load_model('models/Nevus.h5')
    image = tf.keras.preprocessing.image.load_img(image_pth , target_size=(IMAGE_SIZE,IMAGE_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    test_x = np.expand_dims(image , axis=0)
    predictions = model.predict(test_x)
    confidence1 = np.max(predictions)*100

    class_names = ['Melanoma(Kanserli Ben)', 'Nevus(Normal Ben)', 'Seborrheic_Keratosis(İyi Huylu Et Büyümesi)']

    prediction1 = class_names[np.argmax(predictions)]
    if(confidence1 > 50):
        confidence1 = confidence1
    else:
        confidence1 = "Tahmin oranı çok düşük"
        prediction1 = "Tahmin oranı çok düşük"

    image1 = Image.open(image_pth)
    image1 = image1.resize([500,500])
    
    confidence1 = str(confidence1)
    confidence1 = confidence1[:8]

    return render_template('Nevus.html' , prediction1=prediction1 , confidence1=confidence1 , image_pth=image_pth)

"""Nevus Detection Part End"""

"""Brain_Tumor Detection Part"""
@app.route('/upload_photo2', methods=['POST'])
def upload_photo_brain():
    photo_data = request.json['photo'] 

    
    imgdata = base64.b64decode(photo_data.split(',')[1])
    filename = 'static/uploaded_photo.png'  


    with open(filename, 'wb') as f:
        f.write(imgdata)
        
    return render_template('Brain_Tumor.html')

@app.route('/Brain_Tumor', methods = ['POST'])
def predict_brain():
    IMAGE_SIZE = 256
    model = tf.keras.models.load_model('models/Brain_tumor.h5')
    imagefile = request.files['imagefile']
    
    filename= imagefile.filename
    image_pth = "./static/" + filename
    imagefile.save(image_pth)
    image = tf.keras.preprocessing.image.load_img(image_pth , target_size=(IMAGE_SIZE,IMAGE_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    test_x = np.expand_dims(image , axis=0)
    predictions = model.predict(test_x)
    confidence = np.max(predictions)*100
    class_names = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']

    prediction = class_names[np.argmax(predictions)]

    if(confidence > 50):
        confidence = confidence
    else:
        confidence = "Tahmin oranı çok düşük"
        prediction = "Tahmin oranı çok düşük"

    image1 = Image.open(image_pth)
    image1 = image1.resize([500,500])
    
    confidence = str(confidence)
    confidence = confidence[:8]
    
    
    return render_template('/Brain_Tumor.html' , prediction = prediction , confidence = confidence, filename = filename) 


@app.route('/upload_photo2' , methods=['GET'])
def predict_brain2():
    image_pth = "./static/uploaded_photo.png" 
    IMAGE_SIZE = 256
    model = tf.keras.models.load_model('C://Users//user//Desktop//AI_HEALTH//models//Brain_tumor.h5')
    image = tf.keras.preprocessing.image.load_img(image_pth , target_size=(IMAGE_SIZE,IMAGE_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    test_x = np.expand_dims(image , axis=0)
    predictions = model.predict(test_x)
    confidence1 = np.max(predictions)*100
    class_names = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']

    prediction1 = class_names[np.argmax(predictions)]

    if(confidence1 > 0.5):
        confidence1 = confidence1
    else:
        confidence1 = "Tahmin oranı çok düşük"
        prediction1 = "Tahmin oranı çok düşük"

    image1 = Image.open(image_pth)
    image1 = image1.resize([500,500])

    confidence1 = str(confidence1)
    confidence1 = confidence1[:8]

    return render_template('Brain_Tumor.html' , prediction1=prediction1 , confidence1=confidence1 , image_pth=image_pth)

"""Brain_Tumor Detection Part end"""

"""Disease Detection Part """

@app.route('/disease', methods=['POST'])
def disease_detect():
    model = joblib.load('C:/Users/user/Desktop/AI_HEALTH/models/Disease_detect.joblib') 
    df_severity = pd.read_csv('C:/Users/user/Desktop/AI_HEALTH/Training_Models/Symptom_Data/Symptom-severity.csv')
    discrp = pd.read_csv('C:/Users/user/Desktop/AI_HEALTH/Training_Models/Symptom_Data/symptom_Description.csv')
    precaution = pd.read_csv('C:/Users/user/Desktop/AI_HEALTH/Training_Models/Symptom_Data/symptom_precaution.csv')

    def translate_to_turkish_mtranslate(text):
        translation = translate(text, "tr")
        return translation
        
    def predd(x, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17):
        try:
            psymptoms = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17]
            a = np.array(df_severity["Symptom"])
            b = np.array(df_severity["weight"])
            for j in range(len(psymptoms)):
                for k in range(len(a)):
                    if psymptoms[j] == a[k]:
                        psymptoms[j] = b[k]
            psy = [psymptoms]
            pred2 = x.predict(psy)
            disp = discrp[discrp['Disease'] == pred2[0]]
            disp = disp.values[0][1]
            recomnd = precaution[precaution['Disease'] == pred2[0]]
            c = np.where(precaution['Disease'] == pred2[0])[0][0]
            precuation_list = []
            for i in range(1, len(precaution.iloc[c])):
                precuation_list.append(precaution.iloc[c, i])

            trans = translate_to_turkish_mtranslate(pred2[0])
            trans2 = translate_to_turkish_mtranslate(disp)
            ilist = []
            for i in precuation_list:
                if i is not None and str(i).lower() != "nan":
                    # Çeviri işlemi yap
                    translated_i = translate_to_turkish_mtranslate(i)

                    # Eğer çeviri başarılı olduysa ilist'e ekle
                    if translated_i is not None and str(translated_i).lower() != "nan":
                        ilist.append(translated_i)

            result = {
                'Hastalık İsmi': trans,
                'Hastalık Açıklaması': trans2,
                'Yapılması Önerilen Şeyler': ilist
            }

            return result
        except Exception as e:
            result = {
                'Hastalık İsmi': 'Hastalık Bulunamadı',
                'Hastalık Açıklaması': 'Hastalık Bulunamadı'
            }
            return result

    
    sympList = df_severity["Symptom"].to_list()
    selected_values = request.form.get('selected_values')
    if selected_values is not None:
        with open('csv.txt', 'w') as file:
             file.write(selected_values.replace(',' , '\n'))
    
    
    with open('csv.txt', 'r') as file:
        lines = file.readlines()
        values_list = [line.strip() for line in lines]


    
    x_values = [0] * 17 

    for i in range(min(17, len(values_list))):
        try:
            x_values[i] = sympList.index(values_list[i])
        except ValueError:
            x_values[i] = 0

    S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17 = x_values
    
    zero = 0
    S1 = sympList[S1]
    S2 = sympList[S2] if S2 != zero else 0
    S3 = sympList[S3] if S3 != zero else 0
    S4 = sympList[S4] if S4 != zero else 0
    S5 = sympList[S5] if S5 != zero else 0
    S6 = sympList[S6] if S6 != zero else 0
    S7 = sympList[S7] if S7 != zero else 0
    S8 = sympList[S8] if S8 != zero else 0
    S9 = sympList[S9] if S9 != zero else 0
    S10 = sympList[S10] if S10 != zero else 0
    S11 = sympList[S11] if S11 != zero else 0
    S12 = sympList[S12] if S12 != zero else 0
    S13 = sympList[S13] if S13 != zero else 0
    S14 = sympList[S14] if S14 != zero else 0
    S15 = sympList[S15] if S15 != zero else 0
    S16 = sympList[S16] if S16 != zero else 0
    S17 = sympList[S17] if S17 != zero else 0


    predict_result = predd(model, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17)
        

    return render_template('symptom.html', predict_result=predict_result, sympList=sympList)

"""Disease Detection Part end"""

"""Eye Detect Part"""

@app.route('/upload_photo3', methods=['POST'])
def upload_photo_eye():
    photo_data = request.json['photo'] 

    
    imgdata = base64.b64decode(photo_data.split(',')[1])
    filename = 'static/uploaded_photo.png'  

    
    with open(filename, 'wb') as f:
        f.write(imgdata)
        
    return render_template('eye.html')
@app.route('/Eye.html', methods = ['POST'])
def predict_eye():
    IMAGE_SIZE = 256
    model = tf.keras.models.load_model('models/Eyes.h5')
    imagefile = request.files['imagefile']
    
    filename= imagefile.filename
    image_pth = "./static/" + filename
    imagefile.save(image_pth)
    image = tf.keras.preprocessing.image.load_img(image_pth , target_size=(IMAGE_SIZE,IMAGE_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    test_x = np.expand_dims(image , axis=0)
    predictions = model.predict(test_x)
    confidence = np.max(predictions)*100
    
    class_names = ['Bulging_Eyes','Cataracts','Crossed_Eyes','Glaucoma','Normal_Eyes','Uveitis']

    
    prediction = class_names[np.argmax(predictions)]

    if(confidence > 50):
        confidence = confidence
    else:
        confidence = "Tahmin oranı çok düşük"
        prediction = "Tahmin oranı çok düşük"

    image1 = Image.open(image_pth)
    image1 = image1.resize([500,500])
    
    confidence = str(confidence)
    confidence = confidence[:8]
    
    return render_template('/eye.html' , prediction = prediction , confidence = confidence, filename = filename) 


@app.route('/upload_photo3' , methods=['GET'])
def predict_eye2():
    image_pth = "./static/uploaded_photo.png" 
    IMAGE_SIZE = 256
    model = tf.keras.models.load_model('models/Eyes.h5')
    image = tf.keras.preprocessing.image.load_img(image_pth , target_size=(IMAGE_SIZE,IMAGE_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    test_x = np.expand_dims(image , axis=0)
    predictions = model.predict(test_x)
    confidence1 = np.max(predictions)*100

    class_names = ['Bulging_Eyes','Cataracts','Crossed_Eyes','Glaucoma','Normal_Eyes','Uveitis']

    prediction1 = class_names[np.argmax(predictions)]
    if(confidence1 > 50):
        confidence1 = confidence1
    else:
        confidence1 = "Tahmin oranı çok düşük"
        prediction1 = "Tahmin oranı çok düşük"

    image1 = Image.open(image_pth)
    image1 = image1.resize([500,500])
    
    confidence1 = str(confidence1)
    confidence1 = confidence1[:8]

    return render_template('eye.html' , prediction1=prediction1 , confidence1=confidence1 , image_pth=image_pth)


"""Eye Detect Part End"""

@app.route('/Nevus.html')
def Nevus_page():
    return render_template('Nevus.html')

@app.route('/Brain_Tumor.html')
def Brain_Tumor_page():
    return render_template('Brain_Tumor.html')

@app.route('/symptom.html')
def symptom_page():
    return render_template('symptom.html')

@app.route('/eye.html')
def eye_page():
    return render_template('eye.html')

@app.route('/app.html')
def main_page():
    return render_template('app.html')


@app.route('/handle_selection', methods=['POST'])
def handle_selection():
    selected_choice = request.form.get('selected_choice')

    if selected_choice == 'brain_tumor.html':
        return redirect(url_for('show_option1_page'))
    elif selected_choice == 'app.html':
        return redirect(url_for('show_option2_page'))
    elif selected_choice == 'Nevus.html':
        return redirect(url_for('show_option3_page'))
    else:
        return redirect(url_for('op'))
    
    
@app.route('/brain_tumor.html')
def show_option1_page():
    return render_template('brain_tumor.html')

@app.route('/app.html')
def show_option2_page():
    return render_template('app.html')

@app.route('/Nevus.html')
def show_option3_page():
    return render_template('Nevus.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0' ,port=8000 , debug=True)