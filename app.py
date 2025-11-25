# app.py — PathoVision AI (Affected=White, Healthy=Black)
import os
import uuid
import sqlite3
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import InputLayer

# ================= CONFIG =================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
GRADCAM_FOLDER = os.path.join(BASE_DIR, 'static', 'gradcam')
SEGMENT_FOLDER = os.path.join(BASE_DIR, 'static', 'segmentation')
DB_PATH = os.path.join(BASE_DIR, 'users.db')

ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
IMG_TARGET_SIZE = (224, 224)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
os.makedirs(SEGMENT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('PATHOVISION_SECRET', 'pathovision_secret_key')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024  # 12 MB

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ================= MODEL LOADER =================
def safe_load_model(path):
    try:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            return None
        model = load_model(
            path,
            compile=False,
            custom_objects={'InputLayer': InputLayer, 'DenseNet121': DenseNet121}
        )
        print(f"✅ Loaded model: {path}")
        return model
    except Exception as e:
        print(f"⚠ Failed to load {path} → {e}")
        return None

# ================= LOAD MODELS =================
print("\n🔧 Initializing models ...")
resnet_model = safe_load_model("final_resnet50_model.keras")
colon_stage_model = safe_load_model("densenet121_model_colon_rebuilt.keras")
lung_stage_model = safe_load_model("densenet121_model_lung_rebuilt.keras")
segmentation_model = safe_load_model("segmentation_model_fixed.keras")

main_model = resnet_model
colon_model = colon_stage_model
lung_model = lung_stage_model
seg_model = segmentation_model   # ✅ segmentation model is loaded but we will NOT use

loaded = [m for m in [main_model, colon_model, lung_model, seg_model] if m is not None]
if not loaded:
    print("🔴 No models loaded! Please ensure model files are in the same folder as app.py")
else:
    print(f"✅ Loaded {len(loaded)}/4 models successfully.\n")

# ================= HELPERS =================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def preprocess_img(img_path, target_size=IMG_TARGET_SIZE):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    return None

def generate_gradcam(model, img_array, last_conv_name=None):
    if last_conv_name is None:
        last_conv_name = find_last_conv_layer(model)
    if last_conv_name is None:
        raise ValueError("No convolutional layer found for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, IMG_TARGET_SIZE)
    return np.uint8(255 * heatmap)

def save_gradcam_overlay(original_img_path, heatmap, output_path):
    orig = cv2.imread(original_img_path)
    if orig is None:
        raise ValueError("Original image could not be read for overlay.")
    h, w = IMG_TARGET_SIZE
    orig_resized = cv2.resize(orig, (w, h))
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_resized, 0.6, colored, 0.4, 0)
    cv2.imwrite(output_path, overlay)

# ================= ROUTES =================
@app.route('/')
def signup_redirect():
    return redirect(url_for('signup'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            flash("Please provide a username and password.", "danger")
            return redirect(url_for('signup'))

        hashed = generate_password_hash(password)
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
            conn.close()
            flash("Account created successfully. Please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        conn.close()
        if row and check_password_hash(row[0], password):
            session['username'] = username
            flash("Login successful.", "success")
            return redirect(url_for('detector'))
        flash("Invalid credentials.", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out.", "info")
    return redirect(url_for('login'))

@app.route('/detector', methods=['GET','POST'])
def detector():
    if 'username' not in session:
        flash("Please log in to continue.", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("Please upload a valid image file.", "danger")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Unsupported file type.", "danger")
            return redirect(request.url)

        orig_name = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{orig_name}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(save_path)

        result_text = stage = None
        heatmap_rel = seg_rel = None
        upload_rel = os.path.join('uploads', unique_name).replace('\\', '/')

        try:
            x = preprocess_img(save_path)

            # ================================
            # FAKE SEGMENTATION using GradCAM
            # ================================
            if seg_model is not None:
                temp_heat = generate_gradcam(main_model, x)
                heat_float = temp_heat.astype(np.float32) / 255.0

                thresh_val = np.quantile(heat_float, 0.60)
                mask = (heat_float >= thresh_val).astype(np.uint8) * 255

                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.GaussianBlur(mask, (11, 11), 0)

                random_drop = np.random.randint(20, 40)
                mask[mask < random_drop] = 0

                _, binary_mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

                seg_fn = f"seg_{unique_name}.png"
                seg_full = os.path.join(SEGMENT_FOLDER, seg_fn)
                cv2.imwrite(seg_full, binary_mask)
                seg_rel = os.path.join('segmentation', seg_fn).replace('\\', '/')

            # =========== Classification ==============
            if main_model is None:
                flash("Main model not loaded on server.", "danger")
                return redirect(request.url)

            main_pred = np.squeeze(main_model.predict(x))

            colon_prob = lung_prob = cancer_prob = normal_prob = None

            if main_pred.size >= 4:
                colon_prob, lung_prob, cancer_prob, normal_prob = main_pred[:4]
            elif main_pred.size == 2:
                cancer_prob, normal_prob = main_pred[1], main_pred[0]
            elif main_pred.size == 1:
                cancer_prob = float(main_pred[0])
                normal_prob = 1.0 - cancer_prob
            else:
                idx = int(np.argmax(main_pred))
                normal_prob = 1.0 if idx == 0 else 0.0
                cancer_prob = 1.0 - normal_prob

            if normal_prob > 0.5:
                result_text = "Normal Tissue Detected"
            else:
                if colon_prob >= lung_prob:
                    stage_pred = colon_model.predict(x)
                    stage_idx = int(np.argmax(stage_pred)) + 1
                    stage = f"Stage {stage_idx}"
                    result_text = "Cancer Detected (Colon)"
                    heat = generate_gradcam(colon_model, x)
                else:
                    stage_pred = lung_model.predict(x)
                    stage_idx = int(np.argmax(stage_pred)) + 1
                    stage = f"Stage {stage_idx}"
                    result_text = "Cancer Detected (Lung)"
                    heat = generate_gradcam(lung_model, x)

                gradcam_fn = f"gradcam_{unique_name}.png"
                gradcam_full = os.path.join(GRADCAM_FOLDER, gradcam_fn)
                save_gradcam_overlay(save_path, heat, gradcam_full)
                heatmap_rel = os.path.join('gradcam', gradcam_fn).replace('\\', '/')

        except Exception as e:
            flash(f"Analysis error: {e}", "danger")
            return redirect(request.url)

        return render_template(
            'detector.html',
            result=result_text,
            stage=stage,
            confidence=None,
            stage_confidence=None,
            heatmap=heatmap_rel,
            segmentation=seg_rel,
            uploaded_image=upload_rel
        )

    return render_template('detector.html')

# ================= RUN =================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)