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
from scipy.stats import entropy

# --- Utility functions ---
def sanitize_filename(name):
    """Sanitize filename by replacing problematic characters with underscores"""
    return name.replace('/', '_').replace(' ', '_').replace('\\', '_')

# --- Configuration for BraTS Active Learning ---
class Config:
    # Dataset paths
    TRAIN_DIR = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    VAL_DIR = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    
    IMG_SIZE = 224
    BATCH_SIZE = 16  # Smaller batch size for medical data
    ALPHA = 0.1  # weight for explanation loss
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dataset size limitation
    MAX_TRAINING_SAMPLES = 5000  # Limit to first 5000 samples
    
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
    
    # Active Learning Configuration
    INITIAL_LABELED_SIZE = 150  # Start with small labeled set (50 per class)
    ACQUISITION_BATCH_SIZE = 120  # K samples to query per iteration (20 per class)
    MAX_AL_ITERATIONS = 6  # Number of active learning rounds
    LAMBDA_UNCERTAINTY = 0.3  # λ parameter balancing uncertainty vs misalignment
    
    # Fine-tuning Configuration
    FINETUNE_EPOCHS = 3  # Epochs per AL iteration
    FINETUNE_EPISODES = 10  # Episodes per epoch
    FINETUNE_LR = 1e-4  # Learning rate for fine-tuning
    WEIGHT_DECAY = 1e-5  
    
    # Test Set Configuration
    TEST_SET_SIZE = 90  # Fixed test set size (30 per class)
    
    # Minimum tumor voxels to consider a slice valid
    MIN_TUMOR_VOXELS = 100
    
    # Output directories
    OUTPUT_DIR = 'brats_active_learning_output'
    HEATMAP_DIR = os.path.join(OUTPUT_DIR, 'heatmaps')
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    # Model paths - Initialize with baseline prototypical model if available
    BASELINE_MODEL_PATH = None  # Set to your baseline model path if available
    LOG_FILE = os.path.join(LOG_DIR, 'active_learning_log.csv')
    FINETUNE_LOG_FILE = os.path.join(LOG_DIR, 'fine_tuning_log.csv')
    
    SEED = 42

# Create directories
os.makedirs(Config.HEATMAP_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(Config.HEATMAP_DIR, 'samples'), exist_ok=True)
os.makedirs(os.path.join(Config.HEATMAP_DIR, 'episodes'), exist_ok=True)
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

# --- BraTS Dataset Class (Updated for Active Learning) ---
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
        
        # LIMIT DATASET SIZE if specified
        if hasattr(Config, 'MAX_TRAINING_SAMPLES') and Config.MAX_TRAINING_SAMPLES > 0:
            if len(self.valid_slices) > Config.MAX_TRAINING_SAMPLES:
                print(f"🔄 Limiting dataset from {len(self.valid_slices)} to {Config.MAX_TRAINING_SAMPLES} samples...")
                
                # Shuffle to ensure random selection
                random.shuffle(self.valid_slices)
                
                # Keep only first MAX_TRAINING_SAMPLES
                selected_slices = self.valid_slices[:Config.MAX_TRAINING_SAMPLES]
                
                # Remove unused slices from slice_info to save memory
                all_slice_ids = set(self.valid_slices)
                selected_slice_ids = set(selected_slices)
                unused_slice_ids = all_slice_ids - selected_slice_ids
                
                for unused_id in unused_slice_ids:
                    if unused_id in self.slice_info:
                        del self.slice_info[unused_id]
                
                # Update valid_slices
                self.valid_slices = selected_slices
                
                print(f"✅ Dataset limited to {len(self.valid_slices)} samples")
        
        # Print class distribution
        if self.require_segmentation:
            self.print_class_distribution()
            
            # Group slices by class for few-shot sampling
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
        
        if hasattr(Config, 'MAX_TRAINING_SAMPLES'):
            print(f"Total samples used: {total_samples} / {Config.MAX_TRAINING_SAMPLES} (limit)")
    
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
        """Sample balanced few-shot batch for active learning"""
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

# --- Prototypical Network with GradCAM ---
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

# --- Loss functions ---
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

# --- Test Set Creation ---
def create_fixed_test_set(full_dataset, test_size=Config.TEST_SET_SIZE):
    """Create a stratified, fixed test set for consistent evaluation"""
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

# --- Visualization ---
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

# --- Test Evaluation ---
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

# --- Active Learning Framework ---
class BraTSActiveLearningFramework:
    def __init__(self, model, prototypes, full_dataset):
        self.model = model
        self.prototypes = prototypes
        self.full_dataset = full_dataset
        self.lambda_param = Config.LAMBDA_UNCERTAINTY
        
        # Create fixed test set at initialization
        print("Setting up fixed test set for proper evaluation...")
        self.fixed_test_indices = create_fixed_test_set(full_dataset, Config.TEST_SET_SIZE)
        self.fixed_test_subset = Subset(full_dataset, self.fixed_test_indices)
        self.fixed_test_loader = DataLoader(self.fixed_test_subset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        print(f"Fixed test set created with {len(self.fixed_test_indices)} samples")
        
        # Remove test indices from available pool to prevent data leakage
        available_indices = set(range(len(full_dataset))) - set(self.fixed_test_indices)
        
        # Initialize labeled/unlabeled sets from remaining data
        self.labeled_indices = set()
        self.unlabeled_indices = available_indices
        
        print(f"Available data pool (excluding test set): {len(available_indices)} samples")
        
        # Metrics tracking
        self.metrics = {
            'iterations': [],
            'labeled_sizes': [],
            'accuracies': [],
            'test_losses': [],
            'uncertainties': [],
            'misalignments': [],
            'composite_scores': []
        }
        
        # Initialize fine-tuning log
        with open(Config.FINETUNE_LOG_FILE, 'w') as f:
            f.write("AL_Iteration,Epoch,Episode,Proto_Loss,Exp_Loss,Total_Loss,Train_Acc,Val_Loss,Val_Acc,Test_Loss,Test_Acc\n")
    
    def initialize_labeled_set(self):
        """Create initial stratified labeled set from remaining data (excluding test set)"""
        initial_indices = []
        samples_per_class = Config.INITIAL_LABELED_SIZE // Config.NUM_CLASSES
        
        print(f"Creating initial labeled set with {samples_per_class} samples per class (excluding test set)...")
        
        for class_idx in range(Config.NUM_CLASSES):
            # Get class samples that are NOT in test set
            class_slice_ids = self.full_dataset.class_slices[class_idx]
            available_class_indices = []
            
            for slice_id in class_slice_ids:
                idx = self.full_dataset.valid_slices.index(slice_id)
                if idx not in self.fixed_test_indices:  # Exclude test set
                    available_class_indices.append(idx)
            
            if len(available_class_indices) >= samples_per_class:
                selected_indices = random.sample(available_class_indices, samples_per_class)
            else:
                selected_indices = available_class_indices
            
            initial_indices.extend(selected_indices)
            print(f"Class {Config.CLASSES[class_idx]}: {len(selected_indices)} initial samples")
        
        self.labeled_indices = set(initial_indices)
        self.unlabeled_indices -= self.labeled_indices
        
        print(f"Initial labeled set: {len(self.labeled_indices)} samples")
        print(f"Unlabeled pool: {len(self.unlabeled_indices)} samples")
        print(f"Test set: {len(self.fixed_test_indices)} samples")

    def compute_classification_uncertainty(self, unlabeled_indices):
        """Compute H(x) = -∑ p(y=k|x) log p(y=k|x) for uncertainty estimation"""
        self.model.eval()
        uncertainties = []
        
        unlabeled_subset = Subset(self.full_dataset, list(unlabeled_indices))
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=8, shuffle=False)
        
        with torch.no_grad():
            for images, _, _ in unlabeled_loader:
                images = images.to(Config.DEVICE)
                
                # Get embeddings
                embeddings = self.model(images)
                
                # Compute distances to prototypes
                dists = torch.cdist(embeddings, self.prototypes)
                
                # Convert to probabilities
                log_probs = F.log_softmax(-dists, dim=1)
                probs = torch.exp(log_probs)
                
                # Compute entropy for each sample
                for i in range(probs.shape[0]):
                    prob_dist = probs[i].cpu().numpy() + 1e-10
                    h_x = entropy(prob_dist)
                    uncertainties.append(h_x)
        
        return uncertainties
    
    def compute_explanation_misalignment(self, unlabeled_indices):
        """Compute D_exp(x) = 1 - (2|CAM∩PM|)/(|CAM|+|PM|) using Dice-style loss"""
        self.model.eval()
        misalignments = []
        
        # Process in smaller batches to avoid memory issues
        unlabeled_list = list(unlabeled_indices)
        batch_size = 4  # Smaller batch for BraTS due to memory constraints
        
        for i in range(0, len(unlabeled_list), batch_size):
            batch_indices = unlabeled_list[i:i+batch_size]
            
            for idx in batch_indices:
                try:
                    image, label, expert_mask = self.full_dataset[idx]
                    image = image.unsqueeze(0).to(Config.DEVICE)
                    
                    # Get prediction
                    with torch.no_grad():
                        embedding = self.model(image)
                        dists = torch.cdist(embedding, self.prototypes)
                        pred_class = torch.argmin(dists, dim=1).item()
                    
                    # Generate GradCAM for predicted class
                    pred_prototype = self.prototypes[pred_class]
                    gradcam = self.model.compute_gradcam(image, pred_prototype)
                    gradcam_2d = gradcam.squeeze().cpu()
                    
                    # Compute Dice misalignment with expert annotation
                    intersection = torch.sum(gradcam_2d * expert_mask)
                    union = torch.sum(gradcam_2d) + torch.sum(expert_mask)
                    
                    if union > 0:
                        dice_similarity = (2.0 * intersection) / union
                        misalignment = 1.0 - dice_similarity
                    else:
                        misalignment = 1.0
                    
                    misalignments.append(misalignment.item())
                    
                except Exception as e:
                    print(f"Error computing misalignment for sample {idx}: {e}")
                    misalignments.append(1.0)  # Maximum misalignment on error
        
        return misalignments
    
    def compute_composite_acquisition_scores(self, unlabeled_indices):
        """Compute Score(x) = λ·H(x) + (1-λ)·D_exp(x)"""
        print("Computing classification uncertainties...")
        uncertainties = self.compute_classification_uncertainty(unlabeled_indices)
        
        print("Computing explanation misalignments...")
        misalignments = self.compute_explanation_misalignment(unlabeled_indices)
        
        # Ensure same number of samples
        min_samples = min(len(uncertainties), len(misalignments))
        uncertainties = uncertainties[:min_samples]
        misalignments = misalignments[:min_samples]
        
        # Normalize to [0, 1]
        uncertainties = np.array(uncertainties)
        misalignments = np.array(misalignments)
        
        # Compute composite scores
        composite_scores = self.lambda_param * uncertainties + (1 - self.lambda_param) * misalignments
        
        return composite_scores, uncertainties, misalignments
    
    def select_balanced_top_k_samples(self, composite_scores, unlabeled_indices, k):
        """Select top-K samples ensuring BALANCED representation"""
        k_per_class = k // Config.NUM_CLASSES
        
        # Group unlabeled indices by class
        class_unlabeled = {cls_idx: [] for cls_idx in range(Config.NUM_CLASSES)}
        unlabeled_list = list(unlabeled_indices)
        
        for i, idx in enumerate(unlabeled_list):
            slice_id = self.full_dataset.valid_slices[idx]
            cls_idx = self.full_dataset.slice_info[slice_id]['class_idx']
            class_unlabeled[cls_idx].append((i, idx))
        
        selected_indices = []
        selected_scores = []
        
        # Select top k_per_class from each class
        for cls_idx in range(Config.NUM_CLASSES):
            class_samples = class_unlabeled[cls_idx]
            
            if len(class_samples) == 0:
                print(f"Warning: No unlabeled samples for class {Config.CLASSES[cls_idx]}")
                continue
            
            # Get scores for this class
            class_scores = [(composite_scores[score_idx], dataset_idx) 
                           for score_idx, dataset_idx in class_samples]
            
            # Sort by score (descending - higher scores first)
            class_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Select top k_per_class
            num_to_select = min(k_per_class, len(class_scores))
            for i in range(num_to_select):
                score, dataset_idx = class_scores[i]
                selected_indices.append(dataset_idx)
                selected_scores.append(score)
        
        return selected_indices, selected_scores
    
    def visualize_selected_samples(self, selected_indices, al_iteration, num_samples=3):
        """Visualize selected samples with their expert annotations"""
        print(f"Visualizing {num_samples} selected samples from AL iteration {al_iteration}...")
        
        # Create output directory for visualizations
        viz_dir = os.path.join(Config.HEATMAP_DIR, f'visualizations_iter_{al_iteration}')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Select first num_samples samples
        samples_to_viz = selected_indices[:min(num_samples, len(selected_indices))]
        
        for i, idx in enumerate(samples_to_viz):
            try:
                # Get sample data
                image, label, expert_mask = self.full_dataset[idx]
                slice_id = self.full_dataset.valid_slices[idx]
                slice_info = self.full_dataset.slice_info[slice_id]
                class_name = Config.CLASSES[label]
                
                # Save visualization
                save_path = os.path.join(viz_dir, f'selected_sample_{i+1}_slice_{slice_id}_class_{label}.png')
                save_brats_heatmap_comparison(
                    image, expert_mask, torch.zeros_like(expert_mask),  # Dummy gradcam for now
                    class_name, save_path, slice_info
                )
                
                print(f"  Saved: {save_path}")
                
            except Exception as e:
                print(f"Error visualizing sample {i+1} (idx {idx}): {e}")
                continue
        
        print(f"Visualizations saved to: {viz_dir}")

    def query_expert_annotations(self, selected_indices):
        """Simulate expert providing class labels and diagnostic masks"""
        new_labels = []
        new_masks = []
        
        for idx in selected_indices:
            # Get true annotations from dataset (simulates expert)
            _, true_label, true_mask = self.full_dataset[idx]
            new_labels.append(true_label.item())
            new_masks.append(true_mask)
        
        return new_labels, new_masks
    
    def evaluate_current_performance(self):
        """Use fixed test set for consistent, reliable evaluation"""
        test_loss, test_acc = evaluate_model_properly(
            self.model, self.prototypes, self.fixed_test_loader
        )
        return test_loss, test_acc

    def validate_model(self, val_loader):
        """Validate model performance"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                embeddings = self.model(images)
                dists = torch.cdist(embeddings, self.prototypes)
                
                # Loss
                log_probs = F.log_softmax(-dists, dim=1)
                batch_loss = F.nll_loss(log_probs, labels)
                val_loss += batch_loss.item()
                
                # Accuracy  
                preds = torch.argmin(dists, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_val_acc = correct / total if total > 0 else 0.0
        
        self.model.train()
        return avg_val_loss, avg_val_acc

    def retrain_with_dual_objective(self, al_iteration):
        """Fine-tune model with balanced episodes and dual-objective loss"""
        print(f"Retraining model with dual-objective loss (AL Iteration {al_iteration})...")
        print(f"Using balanced few-shot episodes from {len(self.labeled_indices)} labeled samples")
        
        # Set up optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=Config.FINETUNE_LR,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Create validation set from unlabeled data
        val_indices = random.sample(list(self.unlabeled_indices), 
                                   min(30, len(self.unlabeled_indices)))
        val_subset = Subset(self.full_dataset, val_indices)
        val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        self.model.train()
        
        # Training loop
        for epoch in range(Config.FINETUNE_EPOCHS):
            print(f"  Epoch {epoch+1}/{Config.FINETUNE_EPOCHS}")
            
            epoch_proto_loss = 0.0
            epoch_exp_loss = 0.0
            epoch_total_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            num_episodes = 0
            
            max_episodes = Config.FINETUNE_EPISODES
            
            for episode in range(max_episodes):
                try:
                    # Sample balanced few-shot episode
                    episode_data = self.full_dataset.sample_balanced_few_shot_batch(
                        Config.N_SHOT, Config.N_QUERY, list(self.labeled_indices)
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
                    support_embeddings = self.model(support_images)
                    query_embeddings = self.model(query_images)
                    
                    # Prototypical loss
                    proto_loss, prototypes = prototypical_loss(
                        query_embeddings, support_embeddings,
                        query_labels, support_labels, Config.NUM_CLASSES
                    )
                    self.prototypes = prototypes.detach()
                    
                    # Explanation loss
                    exp_losses = []
                    for i in range(query_images.size(0)):
                        try:
                            img = query_images[i:i+1]
                            label_idx = query_labels[i].item()
                            mask = query_masks[i]
                            
                            prototype = prototypes[label_idx]
                            gradcam = self.model.compute_gradcam(img, prototype)
                            exp_loss = dice_loss(gradcam.squeeze(), mask)
                            exp_losses.append(exp_loss)
                        except Exception as e:
                            exp_losses.append(torch.tensor(0.0, device=Config.DEVICE))
                    
                    if exp_losses:
                        exp_loss = torch.stack(exp_losses).mean()
                    else:
                        exp_loss = torch.tensor(0.0, device=Config.DEVICE)
                    
                    # Total loss
                    total_loss = proto_loss + Config.ALPHA * exp_loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
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
            val_loss, val_acc = self.validate_model(val_loader)
            
            # Test evaluation on fixed test set
            test_loss, test_acc = self.evaluate_current_performance()
            
            scheduler.step(val_loss)
            
            # Print progress
            print(f"  Epoch {epoch+1} Summary:")
            print(f"    Train - Proto: {avg_proto_loss:.4f}, Exp: {avg_exp_loss:.4f}, "
                  f"Total: {avg_total_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"    Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"    Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
            
            # Log results
            with open(Config.FINETUNE_LOG_FILE, 'a') as f:
                f.write(f"{al_iteration},{epoch+1},summary,{avg_proto_loss:.4f},"
                       f"{avg_exp_loss:.4f},{avg_total_loss:.4f},{train_acc:.4f},"
                       f"{val_loss:.4f},{val_acc:.4f},{test_loss:.4f},{test_acc:.4f}\n")
        
        self.save_model_checkpoint(al_iteration)
        print("Dual-objective retraining completed!")

    def save_model_checkpoint(self, al_iteration):
        """Save model checkpoint after each AL iteration"""
        model_save_path = os.path.join(Config.MODEL_DIR, f'finetuned_model_iter_{al_iteration}.pt')
        
        checkpoint = {
            'al_iteration': al_iteration,
            'model_state_dict': self.model.state_dict(),
            'prototypes': self.prototypes.cpu().clone(),
            'labeled_indices': sorted(list(self.labeled_indices)),
            'unlabeled_indices': sorted(list(self.unlabeled_indices)),
            'fixed_test_indices': self.fixed_test_indices,
            'config': {
                'num_classes': Config.NUM_CLASSES,
                'embedding_size': Config.EMBEDDING_SIZE,
                'classes': Config.CLASSES,
                'initial_labeled_size': Config.INITIAL_LABELED_SIZE,
                'acquisition_batch_size': Config.ACQUISITION_BATCH_SIZE,
                'lambda_uncertainty': Config.LAMBDA_UNCERTAINTY,
                'alpha': Config.ALPHA,
                'finetune_epochs': Config.FINETUNE_EPOCHS,
                'finetune_lr': Config.FINETUNE_LR,
                'test_set_size': Config.TEST_SET_SIZE
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        torch.save(checkpoint, model_save_path)
        self.prototypes = self.prototypes.to(Config.DEVICE)
        
        print(f"Model checkpoint saved to: {model_save_path}")

    def run_active_learning_cycle(self):
        """Main active learning cycle"""
        print("="*80)
        print("BraTS EXPLAINABILITY-GUIDED ACTIVE LEARNING FRAMEWORK")
        print("="*80)
        
        # Initialize labeled set
        self.initialize_labeled_set()
        
        # Initialize log file
        with open(Config.LOG_FILE, 'w') as f:
            f.write("Iteration,Labeled_Size,Test_Accuracy,Test_Loss,Avg_Uncertainty,Avg_Misalignment,Avg_Score\n")
        
        # Initial evaluation using fixed test set
        initial_test_loss, initial_accuracy = self.evaluate_current_performance()
        print(f"Initial test accuracy with {len(self.labeled_indices)} labeled samples: {initial_accuracy:.4f}")
        print(f"Initial test loss: {initial_test_loss:.4f}")
        
        # Active learning iterations
        for iteration in range(Config.MAX_AL_ITERATIONS):
            print(f"\n=== Active Learning Iteration {iteration + 1}/{Config.MAX_AL_ITERATIONS} ===")
            
            if len(self.unlabeled_indices) < Config.ACQUISITION_BATCH_SIZE:
                print("Not enough unlabeled samples remaining. Stopping early.")
                break
            
            # Compute acquisition scores
            scores, uncertainties, misalignments = self.compute_composite_acquisition_scores(
                self.unlabeled_indices
            )
            
            # Select balanced top-K samples
            selected_indices, selected_scores = self.select_balanced_top_k_samples(
                scores, self.unlabeled_indices, Config.ACQUISITION_BATCH_SIZE
            )
            
            print(f"Selected {len(selected_indices)} samples with avg score: {np.mean(selected_scores):.4f}")
            
            # Visualize selected samples
            self.visualize_selected_samples(selected_indices, iteration + 1, num_samples=3)
            
            # Query expert annotations
            new_labels, new_masks = self.query_expert_annotations(selected_indices)
            
            # Update labeled/unlabeled sets
            self.labeled_indices.update(selected_indices)
            self.unlabeled_indices -= set(selected_indices)
            
            # Retrain model
            self.retrain_with_dual_objective(iteration + 1)
            
            # Evaluate on fixed test set
            test_loss, test_accuracy = self.evaluate_current_performance()
            
            # Track metrics
            self.metrics['iterations'].append(iteration + 1)
            self.metrics['labeled_sizes'].append(len(self.labeled_indices))
            self.metrics['accuracies'].append(test_accuracy)
            self.metrics['test_losses'].append(test_loss)
            self.metrics['uncertainties'].append(np.mean(uncertainties))
            self.metrics['misalignments'].append(np.mean(misalignments))
            self.metrics['composite_scores'].append(np.mean(scores))
            
            print(f"Labeled set size: {len(self.labeled_indices)}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Avg Uncertainty: {np.mean(uncertainties):.4f}")
            print(f"Avg Misalignment: {np.mean(misalignments):.4f}")
            
            # Log results
            with open(Config.LOG_FILE, 'a') as f:
                f.write(f"{iteration+1},{len(self.labeled_indices)},{test_accuracy:.4f},"
                       f"{test_loss:.4f},{np.mean(uncertainties):.4f},"
                       f"{np.mean(misalignments):.4f},{np.mean(scores):.4f}\n")
        
        self.save_final_model()
        return self.metrics

    def save_final_model(self):
        """Save final active learning model"""
        final_model_path = os.path.join(Config.MODEL_DIR, 'final_active_learning_model.pt')
        
        final_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'prototypes': self.prototypes.cpu().clone(),
            'final_metrics': self.metrics,
            'labeled_indices': sorted(list(self.labeled_indices)),
            'unlabeled_indices': sorted(list(self.unlabeled_indices)),
            'fixed_test_indices': self.fixed_test_indices,
            'config': {
                'num_classes': Config.NUM_CLASSES,
                'embedding_size': Config.EMBEDDING_SIZE,
                'classes': Config.CLASSES,
                'initial_labeled_size': Config.INITIAL_LABELED_SIZE,
                'acquisition_batch_size': Config.ACQUISITION_BATCH_SIZE,
                'max_iterations': Config.MAX_AL_ITERATIONS,
                'lambda_uncertainty': Config.LAMBDA_UNCERTAINTY,
                'alpha': Config.ALPHA,
                'finetune_epochs': Config.FINETUNE_EPOCHS,
                'finetune_lr': Config.FINETUNE_LR,
                'test_set_size': Config.TEST_SET_SIZE
            },
            'experiment_summary': {
                'total_al_iterations': len(self.metrics['iterations']),
                'final_labeled_size': len(self.labeled_indices),
                'final_accuracy': self.metrics['accuracies'][-1] if self.metrics['accuracies'] else 0.0,
                'initial_accuracy': self.metrics['accuracies'][0] if len(self.metrics['accuracies']) > 1 else 0.0,
                'improvement': (self.metrics['accuracies'][-1] - self.metrics['accuracies'][0]) if len(self.metrics['accuracies']) > 1 else 0.0
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        torch.save(final_checkpoint, final_model_path)
        self.prototypes = self.prototypes.to(Config.DEVICE)
        
        print(f"Final model saved to: {final_model_path}")

# --- Main execution ---
def run_brats_active_learning_experiment():
    """Run BraTS active learning experiment"""
    print("🧠 BraTS 2020 Explainable Active Learning Pipeline")
    print("=" * 70)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create training dataset (with segmentation)  
    print("📚 Loading training dataset...")
    full_dataset = BraTSDataset(Config.TRAIN_DIR, transform=transform, split='train', require_segmentation=True)
    
    if len(full_dataset) == 0:
        print("❌ No training samples found! Check your data directory.")
        return None
    
    # Initialize model - either from baseline or fresh
    print("Initializing prototypical network...")
    
    if Config.BASELINE_MODEL_PATH and os.path.exists(Config.BASELINE_MODEL_PATH):
        print(f"Loading baseline model from: {Config.BASELINE_MODEL_PATH}")
        checkpoint = torch.load(Config.BASELINE_MODEL_PATH, map_location=Config.DEVICE)
        model = PrototypicalNetwork(embedding_size=Config.EMBEDDING_SIZE)
        model.load_state_dict(checkpoint['model_state_dict'])
        prototypes = checkpoint['prototypes'].to(Config.DEVICE)
    else:
        print("Creating new prototypical network...")
        model = PrototypicalNetwork(embedding_size=Config.EMBEDDING_SIZE)
        # Initialize prototypes randomly
        prototypes = torch.randn(Config.NUM_CLASSES, Config.EMBEDDING_SIZE, device=Config.DEVICE)
        prototypes = F.normalize(prototypes, p=2, dim=1)
    
    model = model.to(Config.DEVICE)
    
    print("Initializing Active Learning Framework...")
    al_framework = BraTSActiveLearningFramework(model, prototypes, full_dataset)
    
    # Evaluate baseline model on fixed test set
    baseline_test_loss, baseline_test_acc = al_framework.evaluate_current_performance()
    print(f"Baseline model test accuracy: {baseline_test_acc:.4f}")
    print(f"Baseline model test loss: {baseline_test_loss:.4f}")
    
    print("\nRunning active learning cycle...")
    main_metrics = al_framework.run_active_learning_cycle()
    
    # Save results
    final_results = {
        'baseline_test_accuracy': baseline_test_acc,
        'baseline_test_loss': baseline_test_loss,
        'main_metrics': main_metrics,
        'test_set_indices': al_framework.fixed_test_indices,
        'config': {
            'initial_labeled_size': Config.INITIAL_LABELED_SIZE,
            'acquisition_batch_size': Config.ACQUISITION_BATCH_SIZE,
            'max_iterations': Config.MAX_AL_ITERATIONS,
            'lambda_uncertainty': Config.LAMBDA_UNCERTAINTY,
            'test_set_size': len(al_framework.fixed_test_indices),
            'fixed_test_evaluation': True,
            'test_separate_from_training': True
        }
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, 'final_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)
    
    print("\n" + "="*80)
    print("BraTS ACTIVE LEARNING EXPERIMENT COMPLETED!")
    print("="*80)
    print(f"Fixed test set size: {len(al_framework.fixed_test_indices)} samples")
    print(f"Baseline test accuracy: {baseline_test_acc:.4f}")
    if main_metrics['accuracies']:
        print(f"Final test accuracy: {main_metrics['accuracies'][-1]:.4f}")
        print(f"Test accuracy improvement: {main_metrics['accuracies'][-1] - baseline_test_acc:+.4f}")
    print(f"Final labeled set size: {main_metrics['labeled_sizes'][-1] if main_metrics['labeled_sizes'] else 0}")
    print(f"Results saved to: {Config.OUTPUT_DIR}")
    
    return final_results

if __name__ == "__main__":
    results = run_brats_active_learning_experiment()