import cv2
import numpy as np

def create_overlay(image_bgr, heatmap_tensor, config):
    """
    Overlays the predicted heatmap onto the original image.
    image_bgr: numpy array of shape (H, W, 3) in BGR format
    heatmap_tensor: torch Tensor or numpy array of shape (1, H, W) or (H, W), values in [0, 1]
    """
    if hasattr(heatmap_tensor, 'cpu'):
        heatmap = heatmap_tensor.squeeze().cpu().numpy()
    else:
        heatmap = heatmap_tensor.squeeze()
        
    # Resize heatmap to match image if needed
    if heatmap.shape != image_bgr.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
        
    # Normalize to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Apply colormap
    cmap_name = config.get('inference', {}).get('heatmap_colormap', 'jet').upper()
    try:
        cmap = getattr(cv2, f"COLORMAP_{cmap_name}")
    except AttributeError:
        cmap = cv2.COLORMAP_JET
        
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cmap)
    
    # Blend
    alpha = config.get('inference', {}).get('heatmap_alpha', 0.4)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    
    return overlay
