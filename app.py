from flask import Flask, redirect, url_for, render_template, request, send_from_directory, jsonify
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import urllib.request
import zipfile

app = Flask(__name__)

# Define the folder where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the "uploads" directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Download and prepare dataset
def prepare_dataset():
    dataset_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
    dataset_path = "traffic_dataset.zip"
    
    if not os.path.exists("datasets/GTSRB/Final_Training/Images"):
        print("Downloading dataset...")
        urllib.request.urlretrieve(dataset_url, dataset_path)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall("datasets")
        os.remove(dataset_path)
        print("Dataset prepared!")

# Load or train model
def get_traffic_model():
    model_path = "models/traffic_model.h5"
    
    if not os.path.exists(model_path):
        print("Training traffic sign model...")
        prepare_dataset()
        
        # Data preprocessing
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            horizontal_flip=False
        )
        
        train_generator = train_datagen.flow_from_directory(
            'datasets/GTSRB/Final_Training/Images',
            target_size=(64, 64),
            batch_size=64,
            subset='training',
            class_mode='categorical'
        )
        
        validation_generator = train_datagen.flow_from_directory(
            'datasets/GTSRB/Final_Training/Images',
            target_size=(64, 64),
            batch_size=64,
            subset='validation',
            class_mode='categorical'
        )
        
        # Enhanced model architecture
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(43, activation='softmax')
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            train_generator,
            epochs=30,
            validation_data=validation_generator,
            callbacks=[early_stopping]
        )
        
        # Save the model
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        print("Model trained and saved!")
    else:
        model = tf.keras.models.load_model(model_path)
    
    return model

# Initialize the model
traffic_model = get_traffic_model()

# Enhanced traffic sign classification with categories
def predict_traffic_sign(img, model):
    img_pil = Image.open(img)
    img_pil = img_pil.convert("RGB")
    img_resized = img_pil.resize((64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))
    
    # Complete traffic sign classes with categories
    class_info = {
        0: {'name': 'Speed limit (20km/h)', 'category': 'regulatory', 'shape': 'circular'},
        1: {'name': 'Speed limit (30km/h)', 'category': 'regulatory', 'shape': 'circular'},
        2: {'name': 'Speed limit (50km/h)', 'category': 'regulatory', 'shape': 'circular'},
        3: {'name': 'Speed limit (60km/h)', 'category': 'regulatory', 'shape': 'circular'},
        4: {'name': 'Speed limit (70km/h)', 'category': 'regulatory', 'shape': 'circular'},
        5: {'name': 'Speed limit (80km/h)', 'category': 'regulatory', 'shape': 'circular'},
        6: {'name': 'End of speed limit (80km/h)', 'category': 'regulatory', 'shape': 'circular'},
        7: {'name': 'Speed limit (100km/h)', 'category': 'regulatory', 'shape': 'circular'},
        8: {'name': 'Speed limit (120km/h)', 'category': 'regulatory', 'shape': 'circular'},
        9: {'name': 'No passing', 'category': 'prohibitory', 'shape': 'circular'},
        10: {'name': 'No passing for vehicles over 3.5 tons', 'category': 'prohibitory', 'shape': 'circular'},
        11: {'name': 'Right-of-way at intersection', 'category': 'priority', 'shape': 'triangular'},
        12: {'name': 'Priority road', 'category': 'priority', 'shape': 'diamond'},
        13: {'name': 'Yield', 'category': 'priority', 'shape': 'triangular'},
        14: {'name': 'Stop', 'category': 'priority', 'shape': 'octagonal'},
        15: {'name': 'No vehicles', 'category': 'prohibitory', 'shape': 'circular'},
        16: {'name': 'No trucks', 'category': 'prohibitory', 'shape': 'circular'},
        17: {'name': 'No entry', 'category': 'prohibitory', 'shape': 'circular'},
        18: {'name': 'General caution', 'category': 'warning', 'shape': 'triangular'},
        19: {'name': 'Dangerous curve left', 'category': 'warning', 'shape': 'triangular'},
        20: {'name': 'Dangerous curve right', 'category': 'warning', 'shape': 'triangular'},
        21: {'name': 'Double curve', 'category': 'warning', 'shape': 'triangular'},
        22: {'name': 'Bumpy road', 'category': 'warning', 'shape': 'triangular'},
        23: {'name': 'Slippery road', 'category': 'warning', 'shape': 'triangular'},
        24: {'name': 'Road narrows on right', 'category': 'warning', 'shape': 'triangular'},
        25: {'name': 'Road work', 'category': 'warning', 'shape': 'triangular'},
        26: {'name': 'Traffic signals', 'category': 'warning', 'shape': 'triangular'},
        27: {'name': 'Pedestrians', 'category': 'warning', 'shape': 'triangular'},
        28: {'name': 'Children crossing', 'category': 'warning', 'shape': 'triangular'},
        29: {'name': 'Bicycles crossing', 'category': 'warning', 'shape': 'triangular'},
        30: {'name': 'Beware of ice/snow', 'category': 'warning', 'shape': 'triangular'},
        31: {'name': 'Wild animals crossing', 'category': 'warning', 'shape': 'triangular'},
        32: {'name': 'End all speed and passing limits', 'category': 'regulatory', 'shape': 'circular'},
        33: {'name': 'Turn right ahead', 'category': 'mandatory', 'shape': 'circular'},
        34: {'name': 'Turn left ahead', 'category': 'mandatory', 'shape': 'circular'},
        35: {'name': 'Ahead only', 'category': 'mandatory', 'shape': 'circular'},
        36: {'name': 'Go straight or right', 'category': 'mandatory', 'shape': 'circular'},
        37: {'name': 'Go straight or left', 'category': 'mandatory', 'shape': 'circular'},
        38: {'name': 'Keep right', 'category': 'mandatory', 'shape': 'circular'},
        39: {'name': 'Keep left', 'category': 'mandatory', 'shape': 'circular'},
        40: {'name': 'Roundabout mandatory', 'category': 'mandatory', 'shape': 'circular'},
        41: {'name': 'End of no passing', 'category': 'regulatory', 'shape': 'circular'},
        42: {'name': 'End of no passing by vehicles over 3.5 tons', 'category': 'regulatory', 'shape': 'circular'}
    }

    # Detailed descriptions for each sign type
    category_descriptions = {
        'regulatory': 'Regulatory signs indicate traffic laws that must be obeyed.',
        'warning': 'Warning signs alert drivers to potential hazards or changes in road conditions.',
        'priority': 'Priority signs indicate who has the right of way at intersections.',
        'prohibitory': 'Prohibitory signs indicate actions that are not allowed.',
        'mandatory': 'Mandatory signs indicate actions that drivers must take.'
    }

    sign = class_info.get(predicted_class, {
        'name': 'Unknown traffic sign',
        'category': 'unknown',
        'shape': 'unknown'
    })

    description = f"{sign['name']}. This is a {sign['category']} sign ({category_descriptions.get(sign['category'], '')}. "
    description += f"It has a {sign['shape']} shape. "

    # Additional specific information for certain signs
    if 'Speed limit' in sign['name']:
        speed = sign['name'].split('(')[1].split(')')[0]
        description += f"Drivers must not exceed {speed} in this area."
    elif sign['name'] == 'Stop':
        description += "Drivers must come to a complete stop at this sign."
    elif sign['name'] == 'Yield':
        description += "Drivers must slow down and yield to other vehicles or pedestrians."
    elif sign['name'] == 'No entry':
        description += "Vehicles are not allowed to enter this road or area."

    return {
        'class_name': sign['name'],
        'category': sign['category'],
        'shape': sign['shape'],
        'description': description,
        'confidence': confidence,
        'action_required': get_action_required(sign['category'])
    }

def get_action_required(category):
    actions = {
        'regulatory': 'Must obey the regulation',
        'warning': 'Proceed with caution',
        'priority': 'Follow right-of-way rules',
        'prohibitory': 'Do not perform the prohibited action',
        'mandatory': 'Must perform the indicated action',
        'unknown': 'No specific action defined'
    }
    return actions.get(category, 'No specific action defined')

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        location = request.form.get('location')
        img = request.files['img']

        if img:
            # Save the uploaded file
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
            img.save(img_path)
            
            try:
                # Make prediction
                result = predict_traffic_sign(img, traffic_model)
                return render_template('index.html', location=location, img=img.filename, result=result)
            except Exception as e:
                return render_template('index.html', error=str(e))
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")