# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform as BaseGlorotUniform, Zeros as BaseZeros
import numpy as np
import argparse
import cv2
import os

# Keras 3 compatibility wrappers for old initializer config
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


def mask_image():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	ap.add_argument("-f", "--face", type=str,
		default="face_detector",
		help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str,
		default="mask_detector.model",
		help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	print(f"[DEBUG] Starting mask detection for image: {args['image']}")
	print(f"[DEBUG] Using face detector: {args['face']}")
	print(f"[DEBUG] Using mask model: {args['model']}")

	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
	weightsPath = os.path.sep.join([args["face"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
	print(f"[DEBUG] Face detector paths: {prototxtPath}, {weightsPath}")
	net = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	model_path = args["model"]
	if not os.path.exists(model_path):
		# prefer .h5 model output from training if original path missing
		alternative = model_path.replace(".model", ".h5")
		if os.path.exists(alternative):
			model_path = alternative
		else:
			raise FileNotFoundError(f"Model not found: {model_path}")

	if model_path.endswith(".model") and os.path.exists(model_path):
		model_h5_path = model_path + ".h5"
		if not os.path.exists(model_h5_path):
			print(f"[INFO] converting {model_path} to {model_h5_path} for Keras 3 compatibility")
			import shutil
			shutil.copyfile(model_path, model_h5_path)
		model_path = model_h5_path

	if model_path.endswith(".h5") and os.path.exists(model_path):
		# keep the .h5 path directly, prefer Keras 3-friendly file
		pass

	model = load_model_compat(model_path)

	# load the input image from disk, clone it, and grab the image spatial
	# dimensions
	print(f"[INFO] loading input image: {args['image']}")
	image = cv2.imread(args["image"])
	if image is None:
		print(f"[ERROR] Could not load image: {args['image']}")
		return
	print(f"[DEBUG] Image shape: {image.shape}")
	orig = image.copy()
	(h, w) = image.shape[:2]

	# construct a blob from the image
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	print("[INFO] computing face detections...")
	net.setInput(blob)
	detections = net.forward()

	print(f"[DEBUG] Found {detections.shape[2]} detections")

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# pass the face through the model to determine if the face
			# has a mask or not
			(mask, withoutMask) = model.predict(face)[0]

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(image, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

	# show the output image
	output_path = args["image"].replace(".jpg", "_detected.jpg").replace(".jpeg", "_detected.jpeg").replace(".png", "_detected.png")
	cv2.imwrite(output_path, image)
	print(f"[INFO] Output saved to: {output_path}")
	print(f"[INFO] Processing completed for {args['image']}")
	
	# GUI display is disabled in this automated mode
	print("[INFO] GUI display skipped; output saved to file.")
	
if __name__ == "__main__":
	mask_image()
