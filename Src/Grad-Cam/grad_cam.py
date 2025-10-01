import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def preprocess_image(path_or_array, target_size=(224,224), to_gray=True):
    """
    Load an image from path or accept a numpy array.
    Returns a preprocessed batch tensor shape (1, H, W, C) dtype float32 scaled [0,1].
    If to_gray True, returns single channel (H, W, 1) to match your model.
    """
    # If path given -> load
    if isinstance(path_or_array, str):
        img = Image.open(path_or_array)
        img = img.convert("L") if to_gray else img.convert("RGB")
        img = img.resize(target_size, Image.BILINEAR)
        arr = np.array(img)
    else:
        # assume numpy array image, resize if needed
        arr = path_or_array
        if arr.ndim == 3 and arr.shape[:2] != target_size:
            arr = np.array(Image.fromarray(arr).resize(target_size, Image.BILINEAR))

        if to_gray and arr.ndim == 3:
            # convert RGB->Gray by averaging channels (simple)
            arr = np.mean(arr, axis=2).astype(arr.dtype)

    # ensure shape H,W,C
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]   # (H,W,1)
    if not to_gray and arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)

    arr = arr.astype("float32") / 255.0
    batch = np.expand_dims(arr, axis=0)  # (1,H,W,C)
    return batch, arr  # return batch and the single image array (unbatched)


def find_last_conv_layer(model):
    """Return the last Conv2D layer OBJECT in the model (or None). Recurses into nested Sequential."""
    # Iterate over all layers in the model recursively
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer  # RETURN OBJECT HERE, NOT layer.name
        elif isinstance(layer, tf.keras.Sequential):
            # Recursively check inner sequential layers
            inner_layer = find_last_conv_layer(layer)
            if inner_layer:
                return inner_layer
    return None


def make_gradcam_heatmap(img_batch, model, last_conv_layer=None, pred_index=None):
    """
    img_batch: a numpy array with shape (1,H,W,C), dtype float32, values [0,1]
    model: a tf.keras.Model (already loaded)
    last_conv_layer: optional layer OBJECT; if None we try to auto-find it
    pred_index: integer class index to visualize; if None we use the predicted class.
                For binary models with sigmoid (output shape (1,1)) pred_index ignored,
                and we compute gradients of the single logit/probability.
    Returns: heatmap (H_conv, W_conv) upsampled to the input image shape via tf.image.resize.
    """

    if last_conv_layer is None:
        last_conv_layer = find_last_conv_layer(model)
        if last_conv_layer is None:
            raise ValueError("No Conv2D layer found in the model!")

    # Build a model that maps the input image to the activations of the last conv layer
    # and the final predictions. Uses layer.output directly (no get_layer needed).
    conv_model = tf.keras.Model([model.inputs], [last_conv_layer.output, model.output])

    # 1) Run forward pass and get conv outputs + predictions
    with tf.GradientTape() as tape:
        # We need tape to watch conv outputs
        inputs = tf.cast(img_batch, tf.float32)
        tape.watch(inputs)
        conv_outputs, preds = conv_model(inputs)   # conv_outputs: (1, h, w, channels)

        if preds.shape[-1] == 1:
            # binary sigmoid model -> scalar output
            class_channel = preds[:, 0]  # shape (1,)
        else:
            # multiclass softmax/logits
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]  # shape (1,)

    # 2) Compute gradients of the class score w.r.t. conv feature maps
    # Note: we need to watch conv_outputs (or inputs) so tape.gradient works
    grads = tape.gradient(class_channel, conv_outputs)  # shape (1, h, w, channels)

    # 3) Pool the gradients over the spatial dimensions to get importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (channels,)

    # 4) Multiply each channel by its weight and sum over channels to get the CAM
    conv_outputs = conv_outputs[0]         # shape (h, w, channels)
    weights = pooled_grads.numpy()         # (channels,)
    cam = np.zeros(shape=conv_outputs.shape[:2], dtype=np.float32)  # (h,w)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i].numpy()

    # 5) Relu on CAM (keep only positive signals) and normalize
    cam = np.maximum(cam, 0)
    if cam.max() != 0:
        cam = cam / (cam.max() + 1e-8)   # scale to [0,1]

    # 6) Upsample CAM to input image size
    input_h, input_w = img_batch.shape[1], img_batch.shape[2]
    cam_resized = tf.image.resize(cam[..., np.newaxis], (input_h, input_w))
    heatmap = tf.squeeze(cam_resized).numpy()  # final heatmap shape (H_input, W_input)
    heatmap = np.clip(heatmap, 0, 1)

    return heatmap


def overlay_gradcam(original_img, heatmap, alpha=0.4, cmap='jet'):
    """
    original_img: single image array H,W,C dtype float32 in [0,1] (C==1 or 3)
    heatmap: H,W in [0,1]
    Returns matplotlib figure showing overlay.
    """
    # ensure 3 channels for display
    if original_img.ndim == 2:
        original_rgb = np.stack([original_img]*3, axis=-1)
    elif original_img.shape[-1] == 1:
        original_rgb = np.concatenate([original_img]*3, axis=-1)
    else:
        original_rgb = original_img

    # Convert heatmap to colormap
    colormap = cm.get_cmap(cmap)
    heatmap_colored = colormap(heatmap)[:, :, :3]  # H,W,3

    # Overlay
    overlayed = (1 - alpha) * original_rgb + alpha * heatmap_colored
    overlayed = np.clip(overlayed, 0, 1)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 3, 1)
    plt.title("Input (gray)")
    plt.imshow(original_rgb if original_rgb.shape[-1] == 3 else original_rgb[...,0], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM heatmap")
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlayed)
    plt.axis('off')
    plt.tight_layout()  # Better spacing
    plt.show()
    return overlayed


