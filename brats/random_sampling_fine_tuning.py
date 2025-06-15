import os
import numpy as np
import pandas as pd
from PIL import Image
import time
import random
import copy
import matplotlib
matplotlib.use('Agg')
import nibabel as nib
import glob
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report

# --- Utility functions ---
def sanitize_filename(name):
    """Sanitize filename by replacing problematic characters with underscores"""
    return name.replace('/', '_').replace(' ', '_').replace('\\', '_')

# --- Configuration for BraTS Random Baseline ---
class Config:
    # Dataset paths
    TRAIN_DIR = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    VAL_DIR = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    
    IMG_SIZE = 224
    BATCH_SIZE = 8  # Smaller batch size for medical data
    ALPHA = 0.05  # weight for explanation loss
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # BraTS tumor classes
    CLASSES = [
        'Edema Dominant',      # Peritumoral edema (label 2)
        'Necrotic Dominant',   # Necrotic core (label 1)
        'Enhancing Dominant'   # Enhancing tumor (label 4)
    ]
    NUM_CLASSES = len(CLASSES)
    
    # Few-shot parameters
    N_SHOT = 3   # 3-shot learning
    N_QUERY = 3  # 3 query samples per class
    EMBEDDING_SIZE = 512
    
    # Random Sampling Configuration (matching Active Learning)
    TOTAL_TRAINING_SAMPLES = 600  # Total samples for random training (200 per class)
    TRAINING_EPOCHS = 15  # More epochs since we're training once vs multiple AL iterations
    TRAINING_EPISODES = 50  # Episodes per epoch
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Test Set Configuration (must match Active Learning)
    TEST_SET_SIZE = 90  # Fixed test set size (30 per class) - SAME as AL
    
    # Minimum tumor voxels to consider a slice valid
    MIN_TUMOR_VOXELS = 100
    
    # Baseline model path
    BASELINE_MODEL_PATH = '/kaggle/input/brats_model_inital/pytorch/default/1/best_brats_explainable_model.pt'
    
    # Output directories
    OUTPUT_DIR = 'brats_random_baseline_output'
    HEATMAP_DIR = os.path.join(OUTPUT_DIR, 'heatmaps')
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    LOG_FILE = os.path.join(LOG_DIR, 'random_baseline_log.csv')
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, 'final_random_baseline_model.pt')
    
    SEED = 42

# Create directories
os.makedirs(Config.HEATMAP_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(Config.HEATMAP_DIR, 'training'), exist_ok=True)
os.makedirs(os.path.join(Config.HEATMAP_DIR, 'validation'), exist_ok=True)

# Set seeds
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)

# --- BraTS Dataset Class (Updated for Random Baseline) ---
class BraTSDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', require_segmentation=True):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.require_segmentation = require_segmentation
        
        # Class to BraTS label mapping
        self.class_to_brats_label = {
            'Edema Dominant': 2,      # Peritumoral edema
            'Necrotic Dominant': 1,   # Necrotic core
            'Enhancing Dominant': 4   # Enhancing tumor
        }
        
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(Config.CLASSES)}
        
        print(f"🧠 Loading BraTS {split} dataset from {data_dir}")
        
        # Find all case directories
        case_dirs = glob.glob(os.path.join(data_dir, "BraTS20_*"))
        print(f"Found {len(case_dirs)} cases")
        
        # Process each case to extract valid slices
        self.valid_slices = []
        self.slice_info = {}  # Store metadata for each slice
        
        for case_dir in case_dirs:
            self.process_case(case_dir)
        
        print(f"✅ Found {len(self.valid_slices)} valid slices")
        
        # Print class distribution
        if self.require_segmentation:
            self.print_class_distribution()
            
            # Group slices by class for balanced sampling
            self.class_slices = {cls_idx: [] for cls_idx in range(Config.NUM_CLASSES)}
            for slice_path in self.valid_slices:
                cls_idx = self.slice_info[slice_path]['class_idx']
                self.class_slices[cls_idx].append(slice_path)
        else:
            print("  (No segmentation - for testing only)")
    
    def process_case(self, case_dir):
        """Process a single BraTS case and extract valid slices"""
        case_name = os.path.basename(case_dir)
        
        try:
            # Check if we need segmentation
            if self.require_segmentation:
                # Load segmentation to find slices with tumors
                seg_path = os.path.join(case_dir, f"{case_name}_seg.nii")
                if not os.path.exists(seg_path):
                    return
                
                seg_nii = nib.load(seg_path)
                seg_data = seg_nii.get_fdata()
                
                # Check each slice (axial dimension is typically the last)
                for slice_idx in range(seg_data.shape[2]):
                    slice_seg = seg_data[:, :, slice_idx]
                    
                    # Count tumor voxels for each type
                    tumor_counts = {
                        1: np.sum(slice_seg == 1),  # Necrotic
                        2: np.sum(slice_seg == 2),  # Edema
                        4: np.sum(slice_seg == 4)   # Enhancing
                    }
                    
                    # Only keep slices with significant tumor
                    total_tumor = sum(tumor_counts.values())
                    if total_tumor < Config.MIN_TUMOR_VOXELS:
                        continue
                    
                    # Determine dominant tumor type
                    dominant_label = max(tumor_counts.keys(), key=lambda x: tumor_counts[x])
                    
                    # Map to class name and index
                    label_to_class = {v: k for k, v in self.class_to_brats_label.items()}
                    class_name = label_to_class[dominant_label]
                    class_idx = self.class_to_idx[class_name]
                    
                    # Create unique identifier for this slice
                    slice_id = f"{case_name}_slice_{slice_idx}"
                    
                    # Store slice information
                    self.slice_info[slice_id] = {
                        'case_dir': case_dir,
                        'case_name': case_name,
                        'slice_idx': slice_idx,
                        'class_name': class_name,
                        'class_idx': class_idx,
                        'dominant_label': dominant_label,
                        'tumor_counts': tumor_counts
                    }
                    
                    self.valid_slices.append(slice_id)
            else:
                # For testing dataset (no segmentation required)
                # Load one modality to get volume dimensions
                t1_path = os.path.join(case_dir, f"{case_name}_t1.nii")
                if not os.path.exists(t1_path):
                    return
                
                t1_nii = nib.load(t1_path)
                t1_data = t1_nii.get_fdata()
                
                # Add all slices (we'll use them for testing)
                for slice_idx in range(t1_data.shape[2]):
                    slice_id = f"{case_name}_slice_{slice_idx}"
                    
                    # Store slice information (without class info)
                    self.slice_info[slice_id] = {
                        'case_dir': case_dir,
                        'case_name': case_name,
                        'slice_idx': slice_idx,
                        'class_name': None,  # Unknown for testing
                        'class_idx': None,   # Unknown for testing
                    }
                    
                    self.valid_slices.append(slice_id)
                    
        except Exception as e:
            print(f"Error processing {case_name}: {e}")
    
    def print_class_distribution(self):
        """Print distribution of classes in the dataset"""
        class_counts = {name: 0 for name in Config.CLASSES}
        for slice_id in self.valid_slices:
            class_name = self.slice_info[slice_id]['class_name']
            class_counts[class_name] += 1
        
        print("Class distribution:")
        total_samples = sum(class_counts.values())
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"  {class_name}: {count} slices ({percentage:.1f}%)")
    
    def normalize_channel(self, channel):
        """Normalize single MRI channel to [0, 1]"""
        channel = channel.astype(np.float32)
        if channel.max() > channel.min():
            channel = (channel - channel.min()) / (channel.max() - channel.min())
        return channel
    
    def load_mri_slice(self, case_dir, case_name, slice_idx):
        """Load T1CE, FLAIR, T2 channels for a specific slice"""
        # Load the three modalities we want
        modalities = {
            't1ce': os.path.join(case_dir, f"{case_name}_t1ce.nii"),
            'flair': os.path.join(case_dir, f"{case_name}_flair.nii"),
            't2': os.path.join(case_dir, f"{case_name}_t2.nii")
        }
        
        channels = []
        for modality, file_path in modalities.items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing {modality} file: {file_path}")
            
            nii_img = nib.load(file_path)
            data = nii_img.get_fdata()
            channel_slice = data[:, :, slice_idx]
            
            # Normalize channel
            normalized_channel = self.normalize_channel(channel_slice)
            channels.append(normalized_channel)
        
        # Stack channels: T1CE, FLAIR, T2
        rgb_image = np.stack(channels, axis=-1)  # Shape: (H, W, 3)
        
        # Convert to PIL Image
        rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
        pil_image = Image.fromarray(rgb_image_uint8)
        
        return pil_image
    
    def load_segmentation_slice(self, case_dir, case_name, slice_idx, target_label):
        """Load binary segmentation mask for target tumor type"""
        seg_path = os.path.join(case_dir, f"{case_name}_seg.nii")
        seg_nii = nib.load(seg_path)
        seg_data = seg_nii.get_fdata()
        
        # Extract slice and create binary mask
        slice_seg = seg_data[:, :, slice_idx]
        binary_mask = (slice_seg == target_label).astype(np.float32)
        
        return binary_mask
    
    def __len__(self):
        return len(self.valid_slices)
    
    def __getitem__(self, idx):
        slice_id = self.valid_slices[idx]
        slice_info = self.slice_info[slice_id]
        
        # Load MRI image (T1CE + FLAIR + T2)
        image = self.load_mri_slice(
            slice_info['case_dir'],
            slice_info['case_name'],
            slice_info['slice_idx']
        )
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.require_segmentation:
            # Load binary segmentation mask for dominant tumor type
            binary_mask = self.load_segmentation_slice(
                slice_info['case_dir'],
                slice_info['case_name'],
                slice_info['slice_idx'],
                slice_info['dominant_label']
            )
            
            # Resize binary mask
            mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((Config.IMG_SIZE, Config.IMG_SIZE), resample=Image.NEAREST)
            mask_tensor = TF.to_tensor(mask_pil).squeeze(0)  # (H, W)
            
            return image, torch.tensor(slice_info['class_idx'], dtype=torch.long), mask_tensor
        else:
            # For testing dataset (no segmentation)
            dummy_mask = torch.zeros(Config.IMG_SIZE, Config.IMG_SIZE)
            dummy_label = torch.tensor(0, dtype=torch.long)  # Dummy label
            return image, dummy_label, dummy_mask
    
    def sample_balanced_few_shot_batch(self, n_shot, n_query, labeled_indices):
        """Sample balanced few-shot batch for training"""
        support_images, support_labels, support_masks = [], [], []
        query_images, query_labels, query_masks = [], [], []
        
        # Group labeled indices by class
        class_labeled_indices = {cls_idx: [] for cls_idx in range(Config.NUM_CLASSES)}
        for idx in labeled_indices:
            slice_id = self.valid_slices[idx]
            cls_idx = self.slice_info[slice_id]['class_idx']
            class_labeled_indices[cls_idx].append(idx)
        
        # Sample EXACTLY n_shot + n_query from each class
        for cls_idx in range(Config.NUM_CLASSES):
            available_indices = class_labeled_indices[cls_idx]
            
            if len(available_indices) == 0:
                print(f"    Warning: No samples for class {Config.CLASSES[cls_idx]} - skipping")
                continue
                
            total_needed = n_shot + n_query
            
            if len(available_indices) < total_needed:
                # Sample with replacement if needed
                sampled_indices = random.choices(available_indices, k=total_needed)
            else:
                # Sample without replacement
                sampled_indices = random.sample(available_indices, total_needed)
            
            # Split into support and query
            support_indices = sampled_indices[:n_shot]
            query_indices = sampled_indices[n_shot:n_shot + n_query]
            
            # Get support examples
            for idx in support_indices:
                img, label, mask = self[idx]
                support_images.append(img)
                support_labels.append(label)
                support_masks.append(mask)
            
            # Get query examples
            for idx in query_indices:
                img, label, mask = self[idx]
                query_images.append(img)
                query_labels.append(label)
                query_masks.append(mask)
        
        if not support_images or not query_images:
            print("  ERROR: Not enough data for balanced few-shot episode")
            return None
        
        # Convert to tensors
        support_images = torch.stack(support_images)
        support_labels = torch.stack(support_labels)
        support_masks = torch.stack(support_masks)
        
        query_images = torch.stack(query_images)
        query_labels = torch.stack(query_labels)
        query_masks = torch.stack(query_masks)
        
        return (support_images, support_labels, support_masks, 
                query_images, query_labels, query_masks)

# --- Prototypical Network with GradCAM (Same as Active Learning) ---
class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_size=Config.EMBEDDING_SIZE, pretrained=True):
        super(PrototypicalNetwork, self).__init__()
        # Load pretrained DenseNet as feature extractor
        densenet = models.densenet121(pretrained=pretrained)
        self.features = densenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(1024, embedding_size)
        
        # Target layer for GradCAM
        self.target_layer = self.features.denseblock4.denselayer16.conv2
    
    def forward(self, x):
        features = self.features(x)
        pooled = self.avgpool(features)
        flat = torch.flatten(pooled, 1)
        embeddings = self.embedding(flat)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def compute_gradcam(self, image, prototype):
        """Compute GradCAM for explainability guidance"""
        original_mode = self.training
        self.eval()
        
        gradcam = torch.zeros(1, 1, Config.IMG_SIZE, Config.IMG_SIZE, device=image.device)
        
        try:
            image_for_grad = image.clone().detach().requires_grad_(True)
            
            activations = None
            gradients = None
            
            def forward_hook(module, input, output):
                nonlocal activations
                activations = output.clone()
                
            def backward_hook(module, grad_input, grad_output):
                nonlocal gradients
                gradients = grad_output[0].clone()
            
            forward_handle = self.target_layer.register_forward_hook(forward_hook)
            backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
            
            embedding = self(image_for_grad)
            prototype = prototype.to(embedding.device)
            distance = torch.norm(embedding - prototype, p=2, dim=1)
            score = -distance
            
            self.zero_grad()
            if image_for_grad.grad is not None:
                image_for_grad.grad.zero_()
            score.backward(retain_graph=True)
            
            if activations is None or gradients is None:
                raise ValueError("Failed to capture activations or gradients")
            
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            
            cam = F.interpolate(cam, size=(Config.IMG_SIZE, Config.IMG_SIZE), 
                              mode='bilinear', align_corners=False)
            
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max > cam_min:
                gradcam[0] = (cam - cam_min) / (cam_max - cam_min)
            else:
                gradcam[0] = cam
            
            forward_handle.remove()
            backward_handle.remove()
            self.train(original_mode)
            
        except Exception as e:
            self.train(original_mode)
            torch.cuda.empty_cache()
            print(f"GradCAM computation failed: {str(e)}")
        
        return gradcam

# --- Loss functions (Same as Active Learning) ---
def dice_loss(pred, target, epsilon=1e-6):
    pred_flat = pred.clone().view(-1)
    target_flat = target.clone().view(-1)
    
    intersection = torch.sum(pred_flat * target_flat)
    pred_sum = torch.sum(pred_flat)
    target_sum = torch.sum(target_flat)
    
    return 1 - (2 * intersection + epsilon) / (pred_sum + target_sum + epsilon)

def prototypical_loss(query_embeddings, support_embeddings, query_labels, support_labels, n_classes):
    prototypes = torch.zeros(n_classes, support_embeddings.shape[1], device=support_embeddings.device)
    for c in range(n_classes):
        mask = support_labels == c
        if mask.sum() > 0:
            prototypes[c] = support_embeddings[mask].mean(0)
    
    dists = torch.cdist(query_embeddings, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)
    target_inds = query_labels
    loss = -log_p_y.gather(1, target_inds.unsqueeze(1)).squeeze().mean()
    
    return loss, prototypes

# --- Fixed Test Set Creation (Same as Active Learning) ---
def create_fixed_test_set(full_dataset, test_size=Config.TEST_SET_SIZE):
    """Create a stratified, fixed test set for consistent evaluation - SAME as Active Learning"""
    samples_per_class = test_size // Config.NUM_CLASSES
    test_indices = []
    
    print(f"Creating fixed test set with {samples_per_class} samples per class...")
    
    for class_idx in range(Config.NUM_CLASSES):
        class_slice_ids = full_dataset.class_slices[class_idx]
        
        if len(class_slice_ids) >= samples_per_class:
            selected_ids = random.sample(class_slice_ids, samples_per_class)
        else:
            selected_ids = class_slice_ids
        
        for slice_id in selected_ids:
            idx = full_dataset.valid_slices.index(slice_id)
            test_indices.append(idx)
        
        print(f"Test set - Class {Config.CLASSES[class_idx]}: {len(selected_ids)} samples")
    
    print(f"Total fixed test set size: {len(test_indices)} samples")
    return test_indices

# --- Balanced Random Training Set Creation ---
def create_balanced_random_training_set(full_dataset, training_size=Config.TOTAL_TRAINING_SAMPLES, test_indices=None):
    """Create balanced random training set for fair comparison with Active Learning"""
    if test_indices is None:
        test_indices = set()
    else:
        test_indices = set(test_indices)
    
    samples_per_class = training_size // Config.NUM_CLASSES
    training_indices = []
    
    print(f"Creating balanced random training set with {samples_per_class} samples per class...")
    
    for class_idx in range(Config.NUM_CLASSES):
        class_slice_ids = full_dataset.class_slices[class_idx]
        
        # Get available samples (excluding test set)
        available_indices = []
        for slice_id in class_slice_ids:
            idx = full_dataset.valid_slices.index(slice_id)
            if idx not in test_indices:  # Exclude test set
                available_indices.append(idx)
        
        if len(available_indices) >= samples_per_class:
            selected_indices = random.sample(available_indices, samples_per_class)
        else:
            selected_indices = available_indices
            print(f"Warning: Only {len(available_indices)} samples available for class {Config.CLASSES[class_idx]}")
        
        training_indices.extend(selected_indices)
        print(f"Training set - Class {Config.CLASSES[class_idx]}: {len(selected_indices)} samples")
    
    print(f"Total balanced random training set size: {len(training_indices)} samples")
    return training_indices

# --- Visualization (Same as Active Learning) ---
def save_brats_heatmap_comparison(image, mask, gradcam, class_name, save_path, slice_info=None):
    """Save BraTS-specific heatmap comparison"""
    plt.figure(figsize=(20, 5))
    
    # Process image for display
    img = image.detach().cpu().permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    # T1CE channel (first channel - most important for tumors)
    t1ce_channel = img[:, :, 0]
    
    # Original composite image
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    title = f"Brain MRI (T1CE+FLAIR+T2)\nClass: {class_name}"
    if slice_info:
        title += f"\nCase: {slice_info.get('case_name', 'Unknown')}"
    plt.title(title)
    plt.axis('off')
    
    # T1CE with expert annotation
    plt.subplot(1, 4, 2)
    plt.imshow(t1ce_channel, cmap='gray')
    mask_np = mask.detach().cpu().numpy()
    plt.imshow(mask_np, cmap='Reds', alpha=0.6)
    plt.title(f"Expert Annotation\n(Radiologist {class_name})")
    plt.axis('off')
    
    # T1CE with model attention
    plt.subplot(1, 4, 3)
    plt.imshow(t1ce_channel, cmap='gray')
    gradcam_cpu = gradcam.detach().cpu()
    gradcam_np = gradcam_cpu.numpy()
    while gradcam_np.ndim > 2:
        gradcam_np = np.squeeze(gradcam_np, axis=0)
    plt.imshow(gradcam_np, cmap='jet', alpha=0.7)
    plt.title("Model Attention\n(GradCAM)")
    plt.axis('off')
    
    # Side-by-side comparison
    plt.subplot(1, 4, 4)
    plt.imshow(t1ce_channel, cmap='gray')
    plt.imshow(mask_np, cmap='Reds', alpha=0.4, label='Expert')
    plt.imshow(gradcam_np, cmap='Blues', alpha=0.4, label='Model')
    plt.title("Expert vs Model\n(Red=Expert, Blue=Model)")
    plt.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

# --- Test Evaluation (Same as Active Learning) ---
def evaluate_model_properly(model, prototypes, test_loader):
    """Proper test evaluation using fixed test set and stable prototypes"""
    model.eval()
    correct = 0
    total = 0
    losses = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            # Get embeddings
            embeddings = model(images)
            
            # Calculate distances to prototypes
            dists = torch.cdist(embeddings, prototypes)
            
            # Log probabilities for loss
            log_p_y = F.log_softmax(-dists, dim=1)
            loss = F.nll_loss(log_p_y, labels)
            losses.append(loss.item())
            
            # Predictions (minimum distance)
            predicted = torch.argmin(dists, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate metrics
    test_loss = np.mean(losses)
    test_acc = correct / total
    
    return test_loss, test_acc

# --- Random Baseline Training ---
def train_random_baseline_model(model, training_indices, test_loader, full_dataset):
    """Train model with random balanced samples using dual-objective loss"""
    print(f"🚀 Training Random Baseline Model with {len(training_indices)} balanced samples...")
    print(f"Training set composition: {Config.TOTAL_TRAINING_SAMPLES // Config.NUM_CLASSES} samples per class")
    
    # Set up optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create validation set from remaining data (not in training or test)
    all_indices = set(range(len(full_dataset)))
    training_set = set(training_indices)
    test_set = set()  # Assuming test_loader uses separate indices
    
    available_for_val = list(all_indices - training_set - test_set)
    val_indices = random.sample(available_for_val, min(50, len(available_for_val)))
    val_subset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Initialize training log
    with open(Config.LOG_FILE, 'w') as f:
        f.write("Epoch,Episode,Proto_Loss,Exp_Loss,Total_Loss,Train_Acc,Val_Loss,Val_Acc,Test_Loss,Test_Acc\n")
    
    best_test_acc = 0.0
    model.train()
    
    # Training loop
    for epoch in range(Config.TRAINING_EPOCHS):
        print(f"\n📊 Epoch {epoch+1}/{Config.TRAINING_EPOCHS}")
        print("-" * 60)
        
        epoch_proto_loss = 0.0
        epoch_exp_loss = 0.0
        epoch_total_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_episodes = 0
        
        # Create epoch heatmap directory
        epoch_heatmap_dir = os.path.join(Config.HEATMAP_DIR, 'training', f"epoch_{epoch+1}")
        os.makedirs(epoch_heatmap_dir, exist_ok=True)
        
        for episode in range(Config.TRAINING_EPISODES):
            try:
                # Sample balanced few-shot episode from training set
                episode_data = full_dataset.sample_balanced_few_shot_batch(
                    Config.N_SHOT, Config.N_QUERY, training_indices
                )
                
                if episode_data is None:
                    continue
                
                support_images, support_labels, support_masks, query_images, query_labels, query_masks = episode_data
                
                # Move to device
                support_images = support_images.to(Config.DEVICE)
                support_labels = support_labels.to(Config.DEVICE)
                support_masks = support_masks.to(Config.DEVICE)
                query_images = query_images.to(Config.DEVICE)
                query_labels = query_labels.to(Config.DEVICE)
                query_masks = query_masks.to(Config.DEVICE)
                
                # Get embeddings
                support_embeddings = model(support_images)
                query_embeddings = model(query_images)
                
                # Prototypical loss
                proto_loss, prototypes = prototypical_loss(
                    query_embeddings, support_embeddings,
                    query_labels, support_labels, Config.NUM_CLASSES
                )
                
                # Explanation loss using expert annotations
                exp_losses = []
                save_heatmap = (episode % 20 == 0)  # Save every 20 episodes
                saved_heatmap = False
                
                for i in range(query_images.size(0)):
                    try:
                        img = query_images[i:i+1]
                        label_idx = query_labels[i].item()
                        mask = query_masks[i]
                        
                        prototype = prototypes[label_idx]
                        gradcam = model.compute_gradcam(img, prototype)
                        exp_loss = dice_loss(gradcam.squeeze(), mask)
                        exp_losses.append(exp_loss)
                        
                        # Save visualization
                        if save_heatmap and not saved_heatmap:
                            class_name_safe = sanitize_filename(Config.CLASSES[label_idx])
                            save_path = os.path.join(
                                epoch_heatmap_dir, 
                                f"episode_{episode+1}_class_{label_idx}_{class_name_safe}.png"
                            )
                            save_brats_heatmap_comparison(
                                img.squeeze(0), mask, gradcam.squeeze(0),
                                Config.CLASSES[label_idx], save_path
                            )
                            saved_heatmap = True
                            
                    except Exception as e:
                        exp_losses.append(torch.tensor(0.0, device=Config.DEVICE))
                
                if exp_losses:
                    exp_loss = torch.stack(exp_losses).mean()
                else:
                    exp_loss = torch.tensor(0.0, device=Config.DEVICE)
                
                # Total loss with explanation guidance
                total_loss = proto_loss + Config.ALPHA * exp_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                
                torch.cuda.empty_cache()
                
                # Compute accuracy
                with torch.no_grad():
                    dists = torch.cdist(query_embeddings, prototypes)
                    pred_labels = torch.argmin(dists, dim=1)
                    correct = (pred_labels == query_labels).sum().item()
                    total = query_labels.size(0)
                
                # Track metrics
                epoch_proto_loss += proto_loss.item()
                epoch_exp_loss += exp_loss.item()
                epoch_total_loss += total_loss.item()
                epoch_correct += correct
                epoch_total += total
                num_episodes += 1
                
                # Print progress
                if (episode + 1) % 10 == 0:
                    batch_acc = correct / total if total > 0 else 0.0
                    print(f"Episode {episode+1:3d}: Loss {total_loss.item():.4f} "
                          f"(Proto: {proto_loss.item():.4f}, Exp: {exp_loss.item():.4f}) "
                          f"Acc: {batch_acc:.4f}")
                
            except Exception as e:
                print(f"    Error in episode {episode+1}: {e}")
                continue
        
        # Calculate epoch averages
        if num_episodes > 0:
            avg_proto_loss = epoch_proto_loss / num_episodes
            avg_exp_loss = epoch_exp_loss / num_episodes  
            avg_total_loss = epoch_total_loss / num_episodes
            train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        else:
            avg_proto_loss = avg_exp_loss = avg_total_loss = train_acc = 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                embeddings = model(images)
                dists = torch.cdist(embeddings, prototypes)
                pred_labels = torch.argmin(dists, dim=1)
                
                # Compute loss
                log_p_y = F.log_softmax(-dists, dim=1)
                batch_loss = F.nll_loss(log_p_y, labels)
                
                accuracy = (pred_labels == labels).float().mean().item()
                val_loss += batch_loss.item()
                val_acc += accuracy
                val_total += 1
        
        if val_total > 0:
            val_loss /= val_total
            val_acc /= val_total
        
        # Test evaluation
        test_loss, test_acc = evaluate_model_properly(model, prototypes, test_loader)
        
        scheduler.step(val_loss)
        model.train()
        
        # Print epoch summary
        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"Train - Proto: {avg_proto_loss:.4f}, Exp: {avg_exp_loss:.4f}, "
              f"Total: {avg_total_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        
        # Save best model based on test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'prototypes': prototypes.cpu().clone(),
                'test_acc': test_acc,
                'training_indices': training_indices,
                # 'config': {
                #     'num_classes': Config.NUM_CLASSES,
                #     'embedding_size': Config.EMBEDDING_SIZE,
                #     'classes': Config.CLASSES,
                #     'total_training_samples': Config.TOTAL_TRAINING_SAMPLES,
                #     'training_epochs': Config.TRAINING_EPOCHS,
                #     'learning_rate': Config.LEARNING_RATE,
                #     'alpha': Config.ALPHA,
                # },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, Config.FINAL_MODEL_PATH)
            print(f"💾 Saved new best model! Test Acc: {test_acc:.4f}")
        
        # Log results
        with open(Config.LOG_FILE, 'a') as f:
            f.write(f"{epoch+1},summary,{avg_proto_loss:.4f},{avg_exp_loss:.4f},"
                   f"{avg_total_loss:.4f},{train_acc:.4f},{val_loss:.4f},"
                   f"{val_acc:.4f},{test_loss:.4f},{test_acc:.4f}\n")
    
    return model, prototypes, best_test_acc

# --- Main execution ---
def run_brats_random_baseline_experiment():
    """Run BraTS random baseline experiment for fair comparison with Active Learning"""
    print("🧠 BraTS Random Baseline Experiment")
    print("=" * 70)
    print("🎯 Purpose: Fair comparison baseline for Active Learning")
    print(f"📊 Training samples: {Config.TOTAL_TRAINING_SAMPLES} (balanced)")
    print(f"🔬 Test samples: {Config.TEST_SET_SIZE} (same as AL)")
    
    # Define transforms (same as Active Learning)
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create training dataset
    print("📚 Loading training dataset...")
    full_dataset = BraTSDataset(Config.TRAIN_DIR, transform=transform, split='train', require_segmentation=True)
    
    if len(full_dataset) == 0:
        print("❌ No training samples found! Check your data directory.")
        return None
    
    # Create fixed test set (SAME as Active Learning)
    print("🔧 Creating fixed test set (same as Active Learning)...")
    test_indices = create_fixed_test_set(full_dataset, Config.TEST_SET_SIZE)
    test_subset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Create balanced random training set
    print("🎲 Creating balanced random training set...")
    training_indices = create_balanced_random_training_set(
        full_dataset, Config.TOTAL_TRAINING_SAMPLES, test_indices
    )
    
    # Load baseline model
    print(f"📥 Loading baseline model from: {Config.BASELINE_MODEL_PATH}")
    if not os.path.exists(Config.BASELINE_MODEL_PATH):
        print("❌ Baseline model not found! Please check the path.")
        return None
    
    checkpoint = torch.load(Config.BASELINE_MODEL_PATH, map_location=Config.DEVICE)
    model = PrototypicalNetwork(embedding_size=Config.EMBEDDING_SIZE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    
    # Initial prototypes from checkpoint or random initialization
    if 'prototypes' in checkpoint:
        prototypes = checkpoint['prototypes'].to(Config.DEVICE)
    else:
        prototypes = torch.randn(Config.NUM_CLASSES, Config.EMBEDDING_SIZE, device=Config.DEVICE)
        prototypes = F.normalize(prototypes, p=2, dim=1)
    
    # Evaluate baseline model
    print("📏 Evaluating baseline model performance...")
    baseline_test_loss, baseline_test_acc = evaluate_model_properly(model, prototypes, test_loader)
    print(f"Baseline test accuracy: {baseline_test_acc:.4f}")
    print(f"Baseline test loss: {baseline_test_loss:.4f}")
    
    # Train with random balanced samples
    print("\n🏋️ Training with random balanced samples...")
    model, final_prototypes, best_test_acc = train_random_baseline_model(
        model, training_indices, test_loader, full_dataset
    )
    
    # Final evaluation
    print("\n🏆 Final Evaluation...")
    final_test_loss, final_test_acc = evaluate_model_properly(model, final_prototypes, test_loader)
    
    # Save final results
    final_results = {
        'baseline_test_accuracy': baseline_test_acc,
        'baseline_test_loss': baseline_test_loss,
        'final_test_accuracy': final_test_acc,
        'final_test_loss': final_test_loss,
        'best_test_accuracy': best_test_acc,
        'improvement': final_test_acc - baseline_test_acc,
        'training_indices': training_indices,
        'test_indices': test_indices,
        # 'config': {
        #     'total_training_samples': Config.TOTAL_TRAINING_SAMPLES,
        #     'samples_per_class': Config.TOTAL_TRAINING_SAMPLES // Config.NUM_CLASSES,
        #     'test_set_size': Config.TEST_SET_SIZE,
        #     'training_epochs': Config.TRAINING_EPOCHS,
        #     'learning_rate': Config.LEARNING_RATE,
        #     'alpha': Config.ALPHA,
        #     'random_baseline': True,
        #     'balanced_sampling': True
        # },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, 'random_baseline_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)
    
    print("\n" + "="*80)
    print("RANDOM BASELINE EXPERIMENT COMPLETED!")
    print("="*80)
    print(f"📊 Training set: {Config.TOTAL_TRAINING_SAMPLES} balanced samples ({Config.TOTAL_TRAINING_SAMPLES // Config.NUM_CLASSES} per class)")
    print(f"🧪 Test set: {Config.TEST_SET_SIZE} samples (same as Active Learning)")
    print(f"📈 Baseline → Final: {baseline_test_acc:.4f} → {final_test_acc:.4f}")
    print(f"🚀 Improvement: {final_test_acc - baseline_test_acc:+.4f}")
    print(f"🏆 Best test accuracy: {best_test_acc:.4f}")
    print(f"💾 Results saved to: {Config.OUTPUT_DIR}")
    print("\n🎯 Ready for comparison with Active Learning results!")
    
    return final_results

if __name__ == "__main__":
    results = run_brats_random_baseline_experiment()