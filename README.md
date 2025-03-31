# **Traffic Sign Recognition with Deep Learning and Web Deployment**

## **Overview**
This project implements a **Deep Learning-based Traffic Sign Recognition System** using an optimized **7-layer CNN** trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The model achieves **98.7% validation accuracy** and is deployed using a **Flask-based web application**.

![Code Structure](https://github.com/rohitcsit/Traffic_Sign_Detection/blob/main/Screenshot%202025-03-31%20195426.jpg)

## **Features**
- **Deep Learning Model**: A **7-layer Convolutional Neural Network (CNN)** optimized with batch normalization and dropout.
- **Data Augmentation**: Uses **12 augmentation techniques** for improved model generalization.
- **Real-Time Web Deployment**: Flask-based web application with image upload, prediction visualization, and geolocation tracking.
- **Efficient Inference**: Achieves **67ms inference time** on standard hardware.
- **Scalability**: Deployed using **Gunicorn** and optimized for cloud deployment.
- **User-Friendly Interface**: Simple and interactive web-based user interface for real-time testing.

![Result Example](https://github.com/rohitcsit/Traffic_Sign_Detection/blob/main/Screenshot%202025-03-31%20195158.jpg)

## **Dataset**
The model is trained on the **GTSRB Dataset**, which includes:
- **43 Traffic Sign Classes**
- **39,209 Training Images**
- **12,630 Test Images**
- **Class Imbalance Handling** using augmentation techniques

## **Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/traffic-sign-detection.git
cd traffic-sign-detection
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Download GTSRB Dataset**
```bash
mkdir dataset
cd dataset
wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip
wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip
unzip '*.zip'
```

## **Model Training**
### **Train the Model**
Run the following command to start training:
```bash
python train.py --epochs 50 --batch_size 64 --lr 0.001
```
### **Evaluate the Model**
Once the model is trained, evaluate it using:
```bash
python evaluate.py --model saved_model.h5
```

## **Web Application Deployment**
### **Run Locally**
Start the Flask web application by running:
```bash
python app.py
```
Now, open your browser and go to **http://127.0.0.1:5000/** to test the application.

### **Deploy with Gunicorn**
To deploy the application in a production environment using **Gunicorn**, run:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## **API Usage**
### **Predict Traffic Sign**
#### **Endpoint:**
```
POST /predict
```
#### **Request:**
```json
{
    "image": "base64_encoded_image"
}
```
#### **Response:**
```json
{
    "prediction": "Stop Sign",
    "confidence": 99.8,
    "class_id": 14
}
```

## **Project Structure**
```
traffic-sign-detection/
│── dataset/                     # Dataset folder
│── models/                      # Trained models
│── static/                      # Static files (CSS, JS, Images)
│── templates/                   # HTML templates for Flask App
│── app.py                        # Flask application script
│── train.py                      # Model training script
│── evaluate.py                   # Model evaluation script
│── requirements.txt              # Required dependencies
│── README.md                     # Documentation
```
