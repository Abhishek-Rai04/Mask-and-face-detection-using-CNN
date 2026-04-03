import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform as BaseGlorotUniform, Zeros as BaseZeros
import detect_mask_image

# Keras 3 compatibility wrappers for old model initializer config
class GlorotUniform(BaseGlorotUniform):
    def __init__(self, seed=None, dtype=None):
        super().__init__(seed=seed)

    def get_config(self):
        return {"seed": self.seed}

class Zeros(BaseZeros):
    def __init__(self, dtype=None):
        super().__init__()

    def get_config(self):
        return {}


def register_legacy_initializers():
    try:
        import keras.src.initializers.random_initializers as random_inits
        import keras.src.initializers.constant_initializers as constant_inits
    except (ImportError, ModuleNotFoundError):
        return

    def make_compat_initializer(base_class):
        class Compat(base_class):
            def __init__(self, *args, dtype=None, **kwargs):
                kwargs.pop("dtype", None)
                super().__init__(*args, **kwargs)

            @classmethod
            def from_config(cls, config):
                config = dict(config)
                config.pop("dtype", None)
                return super().from_config(config)

        Compat.__name__ = base_class.__name__
        Compat.__module__ = base_class.__module__
        return Compat

    initializer_names = ["GlorotUniform", "Zeros", "Ones", "Constant"]

    for name in initializer_names:
        if hasattr(random_inits, name):
            setattr(random_inits, name, make_compat_initializer(getattr(random_inits, name)))
        if hasattr(constant_inits, name):
            setattr(constant_inits, name, make_compat_initializer(getattr(constant_inits, name)))

    try:
        import keras
        for name in initializer_names:
            if hasattr(keras.initializers, name):
                setattr(keras.initializers, name, make_compat_initializer(getattr(keras.initializers, name)))
    except ImportError:
        pass

    try:
        import tensorflow.keras.initializers as tf_init
        for name in initializer_names:
            if hasattr(tf_init, name):
                setattr(tf_init, name, make_compat_initializer(getattr(tf_init, name)))
    except ImportError:
        pass

    # Compatibility patching is already applied through make_compat_initializer invocation above.


def build_mask_detector():
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(224, 224, 3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = False
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-4), metrics=["accuracy"])
    return model


def load_model_compat(model_path):
    register_legacy_initializers()
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"[WARN] load_model failed, retrying with compatibility wrapper: {e}")
        from tensorflow.keras.utils import get_custom_objects
        get_custom_objects().update({"GlorotUniform": GlorotUniform, "Zeros": Zeros})
        try:
            return load_model(model_path, custom_objects={"GlorotUniform": GlorotUniform, "Zeros": Zeros})
        except Exception as e2:
            print(f"[WARN] load_model with custom objects also failed: {e2}")

    print("[WARN] Falling back to building model architecture and loading weights by name")
    model = build_mask_detector()
    try:
        model.load_weights(model_path, by_name=True)
        return model
    except Exception as e3:
        raise RuntimeError(f"Failed to load model or weights from {model_path}: {e3}")


# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(page_title='Face Mask Detector', page_icon='😷', layout='centered', initial_sidebar_state='expanded')


def local_css(file_name):
    """ Method for reading styles.css and applying necessary changes to HTML"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def mask_image():
    # path to models
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])

    # load face detector
    print("[INFO] loading face detector model...")
    if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
        raise FileNotFoundError("Face detector model files not found in face_detector folder")
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load mask detector model
    print("[INFO] loading face mask detector model...")
    model_path = "mask_detector.model"

    if not os.path.exists(model_path):
        model_h5_path = model_path.replace(".model", ".h5")
        if os.path.exists(model_h5_path):
            model_path = model_h5_path
        else:
            raise FileNotFoundError(f"Mask model not found: {model_path}")

    if model_path.endswith(".model") and os.path.exists(model_path):
        model_h5_path = model_path + ".h5"
        if not os.path.exists(model_h5_path):
            print(f"[INFO] converting {model_path} to {model_h5_path} for Keras 3 compatibility")
            import shutil
            shutil.copyfile(model_path, model_h5_path)
        model_path = model_h5_path

    model = load_model_compat(model_path)

    # image input path
    image_path = "./images/out.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass through face detector
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence <= 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        (mask, withoutMask) = model.predict(face)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

        cv2.putText(image, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return RGB_img

# mask_image()  # Remove this - function should only run when called

def mask_detection():
    local_css("css/styles.css")
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    activities = ["Image", "Webcam"]
    st.sidebar.markdown("# Mask Detection on?")
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    if choice == 'Image':
        st.markdown('<h2 align="center">Detection on Image</h2>', unsafe_allow_html=True)
        st.markdown("### Upload your image here ⬇")
        image_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])  # support more formats
        if image_file is not None:
            our_image = Image.open(image_file)  # making compatible to PIL
            im = our_image.save('./images/out.jpg')
            saved_image = st.image(image_file, caption='', use_column_width=True)
            st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
            if st.button('Process'):
                RGB_img = mask_image()  # Call function here when Process is clicked
                st.image(RGB_img, use_column_width=True)

    if choice == 'Webcam':
        st.markdown('<h2 align="center">Detection on Webcam</h2>', unsafe_allow_html=True)
        st.markdown('<h3 align="center">This feature will be available soon!</h3>', unsafe_allow_html=True)


if __name__ == '__main__':
    mask_detection()
