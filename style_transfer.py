"""
Neural Style Transfer Implementation
Optimized for GPU acceleration
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import os

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Model Configuration
CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e-2
TOTAL_VARIATION_WEIGHT = 30
IMG_SIZE = 400

def preprocess_image(image_path):
    """Load and preprocess image for VGG19."""
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def deprocess_image(tensor):
    """Convert tensor back to image."""
    tensor = tensor.numpy()
    tensor = tensor.reshape((IMG_SIZE, IMG_SIZE, 3))
    tensor[:, :, 0] += 103.939
    tensor[:, :, 1] += 116.779
    tensor[:, :, 2] += 123.68
    tensor = tensor[:, :, ::-1]  # BGR to RGB
    return np.clip(tensor, 0, 255).astype('uint8')

def gram_matrix(input_tensor):
    """Compute Gram matrix for style representation."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def compute_loss(feature_extractor, generated_image, content_target, style_targets):
    """Compute total loss for style transfer."""
    with tf.device('/GPU:0'):  # Explicit GPU placement
        generated_features = feature_extractor(generated_image)
        gen_content_feature = generated_features[0]
        gen_style_features = generated_features[1:]
        
        # Content loss
        content_loss = CONTENT_WEIGHT * tf.reduce_mean(
            tf.square(gen_content_feature - content_target))
        
        # Style loss
        style_loss = 0
        for gen_style, style_target in zip(gen_style_features, style_targets):
            style_loss += STYLE_WEIGHT * tf.reduce_mean(
                tf.square(gram_matrix(gen_style) - gram_matrix(style_target)))
        style_loss /= len(style_targets)
        
        # Variation loss
        x_diff = generated_image[:, :, 1:, :] - generated_image[:, :, :-1, :]
        y_diff = generated_image[:, 1:, :, :] - generated_image[:, :-1, :, :]
        variation_loss = TOTAL_VARIATION_WEIGHT * (
            tf.reduce_mean(tf.square(x_diff)) + tf.reduce_mean(tf.square(y_diff)))
        
        return content_loss + style_loss + variation_loss

def main(content_path, style_path, output_path, epochs=10, steps=100):
    """Main style transfer function with GPU optimization."""
    # Verify GPU availability
    if not tf.config.list_physical_devices('GPU'):
        print("Warning: No GPU detected! Falling back to CPU.")
    
    # Load and preprocess images
    with tf.device('/GPU:0'):
        content_image = preprocess_image(content_path)
        style_image = preprocess_image(style_path)
        generated_image = tf.Variable(content_image, dtype=tf.float32)
        
        # Build feature extractor
        vgg = vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        content_layer = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1', 
                       'block3_conv1', 'block4_conv1', 'block5_conv1']
        outputs = [vgg.get_layer(name).output for name in (content_layer + style_layers)]
        feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=outputs)
        
        # Cache targets
        content_target = feature_extractor(content_image)[0]
        style_targets = feature_extractor(style_image)[1:]
    
    # Optimization loop
    opt = tf.optimizers.Adam(learning_rate=0.02)
    for epoch in range(epochs):
        for step in range(steps):
            with tf.GradientTape() as tape:
                loss = compute_loss(
                    feature_extractor, generated_image, content_target, style_targets)
            
            gradients = tape.gradient(loss, [generated_image])
            opt.apply_gradients(zip(gradients, [generated_image]))
            generated_image.assign(tf.clip_by_value(generated_image, 0.0, 255.0))
            
            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {step}/{steps}, Loss: {loss.numpy():.2f}")
    
    # Save result
    final_image = deprocess_image(generated_image[0])
    Image.fromarray(final_image).save(output_path)
    print(f"Styled image saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Neural Style Transfer with GPU Acceleration")
    parser.add_argument('--content', required=True, help='Content image path')
    parser.add_argument('--style', required=True, help='Style image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--steps', type=int, default=100, help='Steps per epoch')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    main(args.content, args.style, args.output, args.epochs, args.steps)