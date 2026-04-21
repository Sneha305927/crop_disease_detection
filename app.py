from flask import Flask, render_template, request
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from deep_translator import GoogleTranslator
import gdown
import os

app = Flask(__name__)

# Paths
MODEL_PATH = "model/crop-disease_model.h5"
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=qft7bV4vh4Q-8F-59IsOf_unDv9AmO6W"
    gdown.download(url, MODEL_PATH, quiet=False)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load model
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# Image size
IMG_SIZE = (224, 224)

# Get class names from dataset folder
DATASET_DIR = "dataset"
class_names = sorted(os.listdir(DATASET_DIR))

disease_info = {

# ================= APPLE =================
"Apple - Apple scab": {
    "description": "Apple scab is a fungal disease causing dark lesions on leaves and fruits.",
    "symptoms": "Olive-green spots, cracked fruits, leaf fall.",
    "treatment": "Spray captan or mancozeb fungicide.",
    "prevention": "Remove fallen leaves, ensure good air circulation."
},

"Apple - Black rot": {
    "description": "Black rot affects apple leaves, fruits, and bark.",
    "symptoms": "Purple spots on leaves, black rotting fruit.",
    "treatment": "Use copper fungicide and prune infected parts.",
    "prevention": "Orchard sanitation and pruning."
},

"Apple - Cedar apple rust": {
    "description": "Fungal disease requiring both apple and cedar trees.",
    "symptoms": "Yellow-orange spots on leaves and fruits.",
    "treatment": "Spray myclobutanil fungicide.",
    "prevention": "Remove nearby cedar trees, plant resistant varieties."
},

"Apple - healthy": {
    "description": "The apple plant is healthy.",
    "symptoms": "No visible symptoms.",
    "treatment": "No treatment required.",
    "prevention": "Maintain proper care."
},

# ================= BLUEBERRY =================
"Blueberry - healthy": {
    "description": "The blueberry plant is healthy.",
    "symptoms": "No disease symptoms.",
    "treatment": "No treatment required.",
    "prevention": "Good agricultural practices."
},

# ================= CHERRY =================
"Cherry (including sour) - Powdery mildew": {
    "description": "Fungal disease producing white powdery growth.",
    "symptoms": "White patches, leaf curling.",
    "treatment": "Spray sulfur fungicide.",
    "prevention": "Ensure good airflow and sunlight."
},

"Cherry (including sour) - healthy": {
    "description": "Cherry plant is healthy.",
    "symptoms": "No symptoms.",
    "treatment": "No treatment needed.",
    "prevention": "Regular maintenance."
},

# ================= CORN =================
"Corn (maize) - Cercospora leaf spot Gray leaf spot": {
    "description": "Fungal disease causing rectangular leaf lesions.",
    "symptoms": "Gray rectangular spots.",
    "treatment": "Apply fungicide.",
    "prevention": "Crop rotation and residue management."
},

"Corn (maize) - Common rust": {
    "description": "Rust disease causing brown pustules.",
    "symptoms": "Brown rust spots.",
    "treatment": "Apply fungicide if severe.",
    "prevention": "Resistant varieties."
},

"Corn (maize) - Northern Leaf Blight": {
    "description": "Fungal disease causing long cigar-shaped lesions.",
    "symptoms": "Long gray lesions.",
    "treatment": "Use fungicide spray.",
    "prevention": "Crop rotation."
},

"Corn (maize) - healthy": {
    "description": "Corn plant is healthy.",
    "symptoms": "No disease symptoms.",
    "treatment": "No treatment needed.",
    "prevention": "Good soil nutrition."
},

# ================= GRAPE =================
"Grape - Black rot": {
    "description": "Fungal disease causing fruit rot.",
    "symptoms": "Shriveled black fruits, brown leaf spots.",
    "treatment": "Spray mancozeb.",
    "prevention": "Remove infected fruits."
},

"Grape - Esca (Black Measles)": {
    "description": "Trunk disease of grapevine.",
    "symptoms": "Black spots, leaf discoloration.",
    "treatment": "No complete cure, remove infected vines.",
    "prevention": "Avoid vine injury."
},

"Grape - Leaf blight (Isariopsis Leaf Spot)": {
    "description": "Fungal leaf disease.",
    "symptoms": "Brown leaf spots.",
    "treatment": "Copper fungicide.",
    "prevention": "Proper pruning."
},

"Grape - healthy": {
    "description": "Grape plant is healthy.",
    "symptoms": "No disease signs.",
    "treatment": "No treatment required.",
    "prevention": "Regular care."
},

# ================= ORANGE =================
"Orange - Haunglongbing (Citrus greening)": {
    "description": "Bacterial disease spread by psyllids.",
    "symptoms": "Yellow shoots, small green fruits.",
    "treatment": "No cure. Remove infected trees.",
    "prevention": "Control insect vectors."
},

# ================= PEACH =================
"Peach - Bacterial spot": {
    "description": "Bacterial disease affecting leaves and fruits.",
    "symptoms": "Small dark spots, leaf drop.",
    "treatment": "Spray copper-based bactericide.",
    "prevention": "Use resistant varieties."
},

"Peach - healthy": {
    "description": "Peach plant is healthy.",
    "symptoms": "No symptoms.",
    "treatment": "No treatment required.",
    "prevention": "Good maintenance."
},

# ================= PEPPER =================
"Pepper bell - Bacterial spot": {
    "description": "Bacterial disease causing leaf and fruit spots.",
    "symptoms": "Water-soaked spots, yellowing.",
    "treatment": "Copper sprays.",
    "prevention": "Use certified seeds."
},

"Pepper bell - healthy": {
    "description": "Pepper plant is healthy.",
    "symptoms": "No disease.",
    "treatment": "No treatment needed.",
    "prevention": "Proper watering."
},

# ================= POTATO =================
"Potato - Early blight": {
    "description": "Fungal disease causing concentric spots.",
    "symptoms": "Brown leaf spots with rings.",
    "treatment": "Spray chlorothalonil.",
    "prevention": "Crop rotation."
},

"Potato - Late blight": {
    "description": "Serious fungal disease causing plant decay.",
    "symptoms": "Water-soaked lesions, white fungus.",
    "treatment": "Use mancozeb.",
    "prevention": "Remove infected plants."
},

"Potato - healthy": {
    "description": "Potato plant is healthy.",
    "symptoms": "No disease.",
    "treatment": "No treatment needed.",
    "prevention": "Proper care."
},

# ================= RASPBERRY =================
"Raspberry - healthy": {
    "description": "Raspberry plant is healthy.",
    "symptoms": "No disease.",
    "treatment": "No treatment needed.",
    "prevention": "Proper maintenance."
},

# ================= SOYBEAN =================
"Soybean - healthy": {
    "description": "Soybean plant is healthy.",
    "symptoms": "No symptoms.",
    "treatment": "No treatment required.",
    "prevention": "Good agricultural practices."
},

# ================= SQUASH =================
"Squash - Powdery mildew": {
    "description": "Fungal disease producing white powder.",
    "symptoms": "White powdery coating on leaves.",
    "treatment": "Spray sulfur fungicide.",
    "prevention": "Avoid overcrowding."
},

# ================= STRAWBERRY =================
"Strawberry - Leaf scorch": {
    "description": "Fungal disease causing leaf burn.",
    "symptoms": "Purple spots, scorched leaves.",
    "treatment": "Remove infected leaves.",
    "prevention": "Good airflow."
},

"Strawberry - healthy": {
    "description": "Strawberry plant is healthy.",
    "symptoms": "No disease.",
    "treatment": "No treatment needed.",
    "prevention": "Regular care."
},

# ================= TOMATO =================
"Tomato - Bacterial spot": {
    "description": "Bacterial disease affecting leaves and fruits.",
    "symptoms": "Small dark lesions.",
    "treatment": "Copper sprays.",
    "prevention": "Use disease-free seeds."
},

"Tomato - Early blight": {
    "description": "Fungal disease causing concentric rings.",
    "symptoms": "Brown spots with rings.",
    "treatment": "Chlorothalonil spray.",
    "prevention": "Crop rotation."
},

"Tomato - Late blight": {
    "description": "Severe fungal disease causing plant collapse.",
    "symptoms": "Water soaked lesions, white mold.",
    "treatment": "Mancozeb spray.",
    "prevention": "Destroy infected plants."
},

"Tomato - Leaf Mold": {
    "description": "Fungal disease in humid environments.",
    "symptoms": "Yellow spots, mold under leaves.",
    "treatment": "Copper fungicide.",
    "prevention": "Ventilation and spacing."
},

"Tomato - Septoria leaf spot": {
    "description": "Fungal disease causing small circular spots.",
    "symptoms": "Small brown spots with gray centers.",
    "treatment": "Fungicide spray.",
    "prevention": "Remove infected leaves."
},

"Tomato - Spider mites Two-spotted spider mite": {
    "description": "Pest infestation causing leaf damage.",
    "symptoms": "Yellow speckles, webbing.",
    "treatment": "Use insecticidal soap.",
    "prevention": "Maintain humidity."
},

"Tomato - Target Spot": {
    "description": "Fungal disease forming target-like spots.",
    "symptoms": "Brown spots with rings.",
    "treatment": "Apply fungicide.",
    "prevention": "Avoid wet leaves."
},

"Tomato - Tomato Yellow Leaf Curl Virus": {
    "description": "Viral disease spread by whiteflies.",
    "symptoms": "Yellow curled leaves, stunted growth.",
    "treatment": "No cure. Remove infected plants.",
    "prevention": "Control whiteflies."
},

"Tomato - Tomato mosaic virus": {
    "description": "Viral disease causing mottled leaves.",
    "symptoms": "Mosaic patterns, leaf distortion.",
    "treatment": "No cure. Remove infected plants.",
    "prevention": "Disinfect tools."
},

"Tomato - healthy": {
    "description": "Tomato plant is healthy.",
    "symptoms": "No disease.",
    "treatment": "No treatment needed.",
    "prevention": "Proper care."
}

}

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "pa": "Punjabi",
    "as": "Assamese",
    "hr": "Haryanvi"   # Google supports it via hi/auto mostly
}

UI_TEXT = {
    "en": {
        "title": "Crop Disease Detection System",
        "welcome": "Welcome!",
        "upload_text": "Upload a leaf image and let our AI detect the disease and explain the result.",
        "button": "Analyze Leaf",
        "footer": "Powered by Deep Learning & Explainable AI (Grad-CAM)",
        "prediction_result": "Prediction Result",
        "disease_detected": "Disease Detected",
        "confidence": "Confidence",
        "original": "Original Image",
        "attention": "AI Attention (Grad-CAM)",
        "disease_info": "Disease Information",
        "description": "Description",
        "symptoms": "Symptoms",
        "treatment": "Treatment",
        "prevention": "Prevention",
        "analyze_again": "Analyze Another Leaf",
        "select_language": "Select Language"
    },

    "hi": {
        "title": "फसल रोग पहचान प्रणाली",
        "welcome": "स्वागत है!",
        "upload_text": "पत्ते की तस्वीर अपलोड करें और AI से रोग का पता लगाएँ।",
        "button": "पत्ता जाँचें",
        "footer": "डीप लर्निंग और AI द्वारा संचालित",
        "prediction_result": "जाँच का परिणाम",
        "disease_detected": "पाई गई बीमारी",
        "confidence": "विश्वास",
        "original": "मूल चित्र",
        "attention": "AI का ध्यान",
        "disease_info": "रोग की जानकारी",
        "description": "विवरण",
        "symptoms": "लक्षण",
        "treatment": "इलाज",
        "prevention": "रोकथाम",
        "analyze_again": "दूसरा पत्ता जाँचें",
        "select_language": "भाषा चुनें"
    },

    "mr": {
        "title": "पीक रोग ओळख प्रणाली",
        "welcome": "स्वागत आहे!",
        "upload_text": "पानाचा फोटो अपलोड करा आणि AI कडून रोग ओळख करून घ्या.",
        "button": "पान तपासा",
        "footer": "डीप लर्निंग आणि AI द्वारा समर्थित",
        "prediction_result": "तपासणीचा निकाल",
        "disease_detected": "आढळलेला रोग",
        "confidence": "विश्वास",
        "original": "मूळ प्रतिमा",
        "attention": "AI लक्ष",
        "disease_info": "रोग माहिती",
        "description": "वर्णन",
        "symptoms": "लक्षणे",
        "treatment": "उपचार",
        "prevention": "प्रतिबंध",
        "analyze_again": "दुसरे पान तपासा",
        "select_language": "भाषा निवडा"
    },

    "pa": {
        "title": "ਫਸਲ ਬਿਮਾਰੀ ਪਛਾਣ ਪ੍ਰਣਾਲੀ",
        "welcome": "ਸੁਆਗਤ ਹੈ!",
        "upload_text": "ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ ਅਤੇ AI ਨਾਲ ਬਿਮਾਰੀ ਪਤਾ ਕਰੋ।",
        "button": "ਪੱਤਾ ਜਾਂਚੋ",
        "footer": "ਡੀਪ ਲਰਨਿੰਗ ਅਤੇ AI ਨਾਲ ਚਲਾਇਆ ਗਿਆ",
        "prediction_result": "ਨਤੀਜਾ",
        "disease_detected": "ਪਾਈ ਗਈ ਬਿਮਾਰੀ",
        "confidence": "ਭਰੋਸਾ",
        "original": "ਅਸਲੀ ਤਸਵੀਰ",
        "attention": "AI ਧਿਆਨ",
        "disease_info": "ਬਿਮਾਰੀ ਦੀ ਜਾਣਕਾਰੀ",
        "description": "ਵੇਰਵਾ",
        "symptoms": "ਲੱਛਣ",
        "treatment": "ਇਲਾਜ",
        "prevention": "ਬਚਾਅ",
        "analyze_again": "ਦੁਬਾਰਾ ਜਾਂਚੋ",
        "select_language": "ਭਾਸ਼ਾ ਚੁਣੋ"
    },

    "as": {
        "title": "শস্য ৰোগ চিনাক্তকৰণ প্ৰণালী",
        "welcome": "স্বাগতম!",
        "upload_text": "পাতৰ ছবি আপলোড কৰক আৰু AI এ ৰোগ চিনাক্ত কৰক।",
        "button": "পাত পৰীক্ষা কৰক",
        "footer": "ডীপ লাৰ্নিং আৰু AI দ্বাৰা চালিত",
        "prediction_result": "ফলাফল",
        "disease_detected": "পোৱা ৰোগ",
        "confidence": "বিশ্বাস",
        "original": "মূল ছবি",
        "attention": "AI দৃষ্টি",
        "disease_info": "ৰোগৰ তথ্য",
        "description": "বিৱৰণ",
        "symptoms": "লক্ষণ",
        "treatment": "চিকিৎসা",
        "prevention": "প্ৰতিৰোধ",
        "analyze_again": "আকৌ পৰীক্ষা কৰক",
        "select_language": "ভাষা বাছনি কৰক"
    },

    # Haryanvi → use Hindi UI (dialect)
    "hr": {}  # will fallback to Hindi
}


def generate_gradcam(img_path, model, predicted_class):
    # Find last conv layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            last_conv_layer_name = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Load original image
    orig = cv2.imread(img_path)
    orig = cv2.resize(orig, IMG_SIZE)

    # Resize heatmap
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    superimposed_img = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)

    return superimposed_img

@app.route("/", methods=["GET", "POST"])
def index():
    # Get selected language (default English)
    lang = request.form.get("language") or request.args.get("lang") or "en"

    # Haryanvi fallback to Hindi UI
    if lang == "hr":
        ui = UI_TEXT["hi"]
    else:
        ui = UI_TEXT.get(lang, UI_TEXT["en"])

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filename = file.filename
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            # Predict
            img = image.load_img(upload_path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            preds = model.predict(img_array)
            predicted_class = np.argmax(preds[0])
            confidence = float(np.max(preds[0]))

            raw_name = class_names[predicted_class]

            # Clean the class name for display
            class_name = raw_name.replace("___", " - ").replace("_", " ")

            # Get disease info (English base)
            info = disease_info.get(class_name, {
                "description": "No description available.",
                "symptoms": "No symptoms information available.",
                "treatment": "No treatment information available.",
                "prevention": "No prevention information available."
            })

            # Translate disease info if not English
            if lang != "en":
                try:
                    from deep_translator import GoogleTranslator
                    target_lang = "hi" if lang == "hr" else lang
                    translator = GoogleTranslator(source="en", target=target_lang)
                    info = {
                        "description": translator.translate(info["description"]),
                        "symptoms": translator.translate(info["symptoms"]),
                        "treatment": translator.translate(info["treatment"]),
                        "prevention": translator.translate(info["prevention"])
                    }
                except Exception as e:
                    print("❌ TRANSLATION ERROR:", e)
 # fallback to English

            # Generate heatmap
            heatmap_img = generate_gradcam(upload_path, model, predicted_class)
            result_path = os.path.join(RESULT_FOLDER, "heatmap_" + filename)
            cv2.imwrite(result_path, heatmap_img)

            return render_template(
                "result.html",
                ui=ui,
                uploaded_image="uploads/" + filename,
                heatmap_image="results/" + "heatmap_" + filename,
                prediction=class_name,
                confidence=round(confidence * 100, 2),
                description=info["description"],
                symptoms=info["symptoms"],
                treatment=info["treatment"],
                prevention=info["prevention"],
                 
            )

    return render_template("index.html", ui=ui, selected_lang=lang)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

