import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

# --- IMPORT LITE-UNET ---
from LiteUNet import Lite_UNet
# from EfficientLiteUNet import Lite_UNet

# --- USER CONFIGURATION SECTION ---
MODEL_WEIGHTS_PATH = r"C:\Users\User\Desktop\Lite-UNet\lite-unet-best.h5" 
BASE_DATA_PATH = r"C:\Users\User\Desktop\Paddy_Dataset"
MAIN_OUTPUT_DIR = r"C:\Users\User\Desktop\Lite-UNet\testing_result_1"

# The 7 disease folders
DISEASES = ["Bacterial Leaf Blight", "Bacterial Leaf Streak", "Blast", "Brown Spot", "DownyMildew", "Hispa", "Tungro"]

# Model Config
IMG_HEIGHT = 640
IMG_WIDTH = 480

# Mock arguments class to pass into Lite_UNet
class MockArgs:
    img_height = IMG_HEIGHT
    img_width = IMG_WIDTH
    batch_size = 1
    lr = 0.0001
    epochs = 1
    output_dir = ''
    result_dir = ''
# ----------------------------------

def calculate_complexity(model):
    """Calculates model complexity (Parameters and FLOPs) in TensorFlow 2.x."""
    # 1. Calculate Parameters
    trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params

    # 2. Calculate FLOPs using TF Profiler
    try:
        # Create a tf.function to represent the model
        @tf.function
        def model_fn(inputs):
            return model(inputs)
        
        # Get the concrete function matching the exact input shape
        input_signature = tf.TensorSpec(shape=(1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)
        concrete_func = model_fn.get_concrete_function(input_signature)
        
        # Convert to a frozen graph
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)
        
        # Run the profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
        # Suppress the massive profiler terminal output by routing it to a dummy file/null
        opts['output'] = 'none' 
        
        flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                              run_meta=run_meta, 
                                              cmd='op', 
                                              options=opts)
        total_flops = flops.total_float_ops
    except Exception as e:
        print(f"[WARNING] FLOPs calculation failed: {e}")
        total_flops = 0

    return total_params, total_flops

def calculate_metrics(pred_mask, true_mask):
    """Calculates evaluation metrics using NumPy."""
    pred = pred_mask.flatten()
    true = true_mask.flatten()
    
    pred_bin = (pred > 0.5).astype(np.uint8)
    true_bin = (true > 0.5).astype(np.uint8)

    tp = np.sum((pred_bin == 1) & (true_bin == 1))
    tn = np.sum((pred_bin == 0) & (true_bin == 0))
    fp = np.sum((pred_bin == 1) & (true_bin == 0))
    fn = np.sum((pred_bin == 0) & (true_bin == 1))

    epsilon = 1e-7
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)

    return {"Dice": dice, "IoU": iou, "Precision": precision, "Recall": recall, "Accuracy": accuracy, "F1_Score": f1}

def save_visual_result(image_np, true_mask_np, pred_mask_np, filename, dice_score, output_dir):
    """Saves side-by-side comparison of Original, GT, and Prediction."""
    # For Lite-UNet only
    # # Convert image back from [-1, 1] (Dataloader format) to [0, 1] for visualization
    # img_display = (image_np + 1.0) / 2.0

    # For EfficientNet-based Lite-UNet
    # Convert image from [0, 255] (EfficientNet format) to [0, 1] for visualization
    img_display = image_np / 255.0
    img_display = np.clip(img_display, 0, 1) # Ensure safe bounds
    
    true_bin = (true_mask_np > 0.5).astype(np.uint8)
    pred_bin = (pred_mask_np > 0.5).astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img_display)
    ax[0].set_title(f"Original: {filename}")
    ax[0].axis("off")
    
    ax[1].imshow(true_bin, cmap='gray')
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")
    
    ax[2].imshow(pred_bin, cmap='gray')
    ax[2].set_title(f"Pred (Dice: {dice_score:.2f})")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}_eval.png", dpi=150)
    plt.close(fig)

def read_image_and_mask(img_path, mask_path):
    """Reads and preprocesses image/mask EXACTLY like Dataloader.py"""
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    
    # OpenCV uses (Width, Height) for resize
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
    img = np.expand_dims(img, axis=0) # Add batch dimension: (1, H, W, 3)

    graymask = cv2.imread(mask_path, 0)
    if graymask is None:
        return None, None
        
    graymask = cv2.resize(graymask, (IMG_WIDTH, IMG_HEIGHT))
    (_, mask) = cv2.threshold(graymask, 1, 255, cv2.THRESH_BINARY)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=0) # Add batch dimension
    
    return img, mask

def run_test_on_disease(disease_name, model, params, flops):
    img_dir = os.path.join(BASE_DATA_PATH, disease_name, "Infer_Ori")
    mask_dir = os.path.join(BASE_DATA_PATH, disease_name, "Infer_GT")
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"[WARNING] Skipping {disease_name}: Path not found.")
        return

    disease_output_dir = Path(MAIN_OUTPUT_DIR) / disease_name
    img_output_dir = disease_output_dir / "predictions"
    disease_output_dir.mkdir(parents=True, exist_ok=True)
    img_output_dir.mkdir(parents=True, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    results = []
    
    for filename in tqdm(img_files, desc=f"Testing {disease_name}"):
        img_path = os.path.join(img_dir, filename)
        mask_name = filename if os.path.exists(os.path.join(mask_dir, filename)) else os.path.splitext(filename)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)

        img_np, true_mask_np = read_image_and_mask(img_path, mask_path)
        if img_np is None or true_mask_np is None:
            continue

        pred_mask_np = model.predict(img_np, verbose=0)
        
        metrics = calculate_metrics(pred_mask_np[0], true_mask_np[0])
        metrics['Filename'] = filename
        results.append(metrics)
        
        save_visual_result(img_np[0], true_mask_np[0], pred_mask_np[0], filename, metrics['Dice'], img_output_dir)

    if results:
        df = pd.DataFrame(results)
        metric_cols = ['Dice', 'IoU', 'Precision', 'Recall', 'Accuracy', 'F1_Score']
        means = df[metric_cols].mean().to_dict()
        
        # Divide by 10^9 to format FLOPs into GFLOPs if preferred, but leaving as exact count here
        summary_df = pd.DataFrame([{'Metric': k, 'Value': v} for k, v in means.items()] + 
                                  [{'Metric': 'Params', 'Value': params}, {'Metric': 'FLOPs', 'Value': flops}])

        excel_path = disease_output_dir / f'{disease_name}_metrics.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            df[['Filename'] + metric_cols].to_excel(writer, sheet_name='Detailed', index=False)
            
        return means
    return None

if __name__ == '__main__':
    args = MockArgs()
    
    print("Building Lite-UNet architecture...")
    lite_unet_builder = Lite_UNet(args)
    model = lite_unet_builder.build_model()
    
    print(f"Loading weights from: {MODEL_WEIGHTS_PATH}")
    try:
        model.load_weights(MODEL_WEIGHTS_PATH)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}")
        exit(1)

    print("Calculating Model Complexity (This may take a moment)...")
    params, flops = calculate_complexity(model)
    print(f"Model Parameters: {params:,}")
    print(f"Model FLOPs: {flops:,}")

    all_disease_means = []
    for disease in DISEASES:
        disease_mean = run_test_on_disease(disease, model, params, flops)
        if disease_mean:
            disease_mean['Disease'] = disease
            all_disease_means.append(disease_mean)
            
    if all_disease_means:
        calc_mean_dir = os.path.join(MAIN_OUTPUT_DIR, "calculated_mean")
        os.makedirs(calc_mean_dir, exist_ok=True)
        means_df = pd.DataFrame(all_disease_means)
        cols = ['Disease'] + [c for c in means_df.columns if c != 'Disease']
        means_df = means_df[cols]
        
        # Calculate the overall mean and append it to the DataFrame
        overall_mean = means_df.mean(numeric_only=True).to_dict()
        overall_mean['Disease'] = 'OVERALL MEAN'
        means_df = pd.concat([means_df, pd.DataFrame([overall_mean])], ignore_index=True)
        
        save_path = os.path.join(calc_mean_dir, "calculated_mean.xlsx")
        means_df.to_excel(save_path, index=False)
        print(f"\nCalculated means for all diseases saved to: {save_path}")

    print("\n--- All Testing Completed ---")