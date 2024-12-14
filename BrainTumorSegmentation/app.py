from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Load the pre-trained UNet model
model = load_model('model/final_UNET_Brain_Tumor.keras', compile=False)

def preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess the input image."""
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image) / 255.0  # Normalize the image
    return image_array

def preprocess_mask(mask_path, target_size=(256, 256)):
    """Load and preprocess the mask."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_resized = cv2.resize(mask, target_size)
    mask_normalized = mask_resized.astype('float32') / 255.0
    return mask_normalized

def save_visualization(image, predicted_mask, ground_truth_mask, output_path):
    """Visualize segmentation results and save as an image."""
    # Unnormalize image (multiply by 255)
    image = (image * 255).astype(np.uint8)
    
    # Generate the binary mask for the predicted mask
    binary_pred_mask = (predicted_mask > 0.5).astype(np.uint8)

    # Create the segmented image by applying the binary mask to the original image
    segmented_image = cv2.merge((binary_pred_mask, binary_pred_mask, binary_pred_mask)) * image

    # Plot the results
    plt.figure(figsize=(12, 4))  # Adjusted width for three images
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(segmented_image)
    plt.title('Segmented Image')

    # Save the result
    plt.savefig(output_path)
    plt.close()

@app.route('/')
def index():
    """Render the upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image uploads and perform segmentation."""
    if 'image' not in request.files:
        return "No image uploaded", 400
    image_file = request.files['image']
    mask_file = request.files.get('mask')  # Optional ground truth mask

    # Save uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    mask_path = None
    if mask_file:
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_file.filename)
        mask_file.save(mask_path)

    # Preprocess image
    image_array = preprocess_image(image_path)
    ground_truth_array = preprocess_mask(mask_path) if mask_path else None

    # Predict segmentation mask
    predicted_mask = model.predict(np.expand_dims(image_array, axis=0))[0].squeeze()

    # Save the visualization
    result_path = os.path.join(app.config['RESULT_FOLDER'], 'segmentation_result.png')
    save_visualization(image_array, predicted_mask, ground_truth_array, result_path)

    return render_template('result.html', result_image='results/segmentation_result.png')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(debug=True)
