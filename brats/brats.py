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

# --- Configuration for BraTS ---
class Config:
    # Dataset paths
    TRAIN_DIR = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    VAL_DIR = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    
    IMG_SIZE = 224
    BATCH_SIZE = 8  # Smaller batch size for medical data
    ALPHA = 0.05  # weight for explanation loss
    NUM_EPOCHS = 5
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
    
    # Minimum tumor voxels to consider a slice valid
    MIN_TUMOR_VOXELS = 100
    
    # Output directories
    OUTPUT_DIR = 'brats_output'
    HEATMAP_DIR = os.path.join(OUTPUT_DIR, 'heatmaps')
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    MODEL_PATH = os.path.join(MODEL_DIR, 'best_brats_explainable_model.pt')
    LOG_FILE = os.path.join(LOG_DIR, 'brats_training_log.txt')
    
    SAVE_EPOCH_HEATMAPS = 2
    SAVE_EPISODE_HEATMAPS = 30
    SEED = 42
    ENABLE_ANOMALY_DETECTION = True

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

if Config.ENABLE_ANOMALY_DETECTION:
    torch.autograd.set_detect_anomaly(True)
    print("PyTorch anomaly detection enabled")

# --- BraTS Dataset Class ---
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
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} slices")
    
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
    
    def sample_few_shot_batch(self, n_shot, n_query):
        """Sample few-shot batch for episodic training"""
        if not self.require_segmentation:
            raise ValueError("Few-shot sampling requires segmentation data")
            
        support_images, support_labels, support_masks = [], [], []
        query_images, query_labels, query_masks = [], [], []
        
        for cls_idx in range(Config.NUM_CLASSES):
            class_slices = self.class_slices[cls_idx]
            
            if len(class_slices) == 0:
                print(f"Warning: No slices for class {Config.CLASSES[cls_idx]}")
                continue
            
            # Sample with replacement if necessary
            total_needed = n_shot + n_query
            if len(class_slices) < total_needed:
                sampled_slices = random.choices(class_slices, k=total_needed)
            else:
                sampled_slices = random.sample(class_slices, total_needed)
            
            # Split into support and query
            support_slice_ids = sampled_slices[:n_shot]
            query_slice_ids = sampled_slices[n_shot:n_shot + n_query]
            
            # Get support examples
            for slice_id in support_slice_ids:
                idx = self.valid_slices.index(slice_id)
                img, label, mask = self[idx]
                support_images.append(img)
                support_labels.append(label)
                support_masks.append(mask)
            
            # Get query examples
            for slice_id in query_slice_ids:
                idx = self.valid_slices.index(slice_id)
                img, label, mask = self[idx]
                query_images.append(img)
                query_labels.append(label)
                query_masks.append(mask)
        
        # Convert to tensors
        if support_images:
            support_images = torch.stack(support_images)
            support_labels = torch.stack(support_labels)
            support_masks = torch.stack(support_masks)
        else:
            support_images = torch.empty(0, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            support_labels = torch.empty(0, dtype=torch.long)
            support_masks = torch.empty(0, Config.IMG_SIZE, Config.IMG_SIZE)
        
        if query_images:
            query_images = torch.stack(query_images)
            query_labels = torch.stack(query_labels)
            query_masks = torch.stack(query_masks)
        else:
            query_images = torch.empty(0, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            query_labels = torch.empty(0, dtype=torch.long)
            query_masks = torch.empty(0, Config.IMG_SIZE, Config.IMG_SIZE)
        
        return (support_images, support_labels, support_masks,
                query_images, query_labels, query_masks)

# --- Few-shot sampler ---
class FewShotSampler:
    def __init__(self, dataset, n_episodes, n_shot, n_query):
        self.dataset = dataset
        self.n_episodes = n_episodes
        self.n_shot = n_shot
        self.n_query = n_query
    
    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self.dataset.sample_few_shot_batch(self.n_shot, self.n_query)
    
    def __len__(self):
        return self.n_episodes

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
            raise ValueError(f"GradCAM computation failed: {str(e)}")
        
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

# --- Training function ---
def train_brats_model(train_episodes, val_loader):
    """Train prototypical network on BraTS with expert-guided explainability"""
    model = PrototypicalNetwork(embedding_size=Config.EMBEDDING_SIZE)
    model = model.to(Config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, verbose=True
    )
    
    best_val_acc = 0.0
    
    # Create log file
    with open(Config.LOG_FILE, 'w') as log_file:
        log_file.write("Epoch,Episode,Loss,CLS_Loss,EXP_Loss,Acc,Val_Loss,Val_Acc\n")
    
    print(f"🚀 Starting BraTS explainable few-shot training...")
    print(f"Device: {Config.DEVICE}")
    print(f"Classes: {Config.CLASSES}")
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_exp_loss = 0.0
        train_acc = 0.0
        train_total = 0
        
        print(f"\n📊 Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Create epoch heatmap directory
        epoch_heatmap_dir = os.path.join(Config.HEATMAP_DIR, 'episodes', f"epoch_{epoch+1}")
        os.makedirs(epoch_heatmap_dir, exist_ok=True)
        
        # Training episodes
        for episode, (support_images, support_labels, support_masks,
                     query_images, query_labels, query_masks) in enumerate(train_episodes):
            
            if support_images.size(0) == 0 or query_images.size(0) == 0:
                continue
            
            # Move to device
            support_images = support_images.to(Config.DEVICE)
            support_labels = support_labels.to(Config.DEVICE)
            support_masks = support_masks.to(Config.DEVICE)
            query_images = query_images.to(Config.DEVICE)
            query_labels = query_labels.to(Config.DEVICE)
            query_masks = query_masks.to(Config.DEVICE)
            
            torch.cuda.empty_cache()
            
            # Get embeddings
            support_embeddings = model(support_images)
            query_embeddings = model(query_images)
            
            # Compute prototypical loss
            cls_loss, prototypes = prototypical_loss(
                query_embeddings, support_embeddings,
                query_labels, support_labels,
                Config.NUM_CLASSES
            )
            
            # Compute explanation loss using expert annotations
            class_exp_losses = []
            save_heatmap = (episode % Config.SAVE_EPISODE_HEATMAPS == 0)
            saved_heatmap = False
            
            for cls_idx in range(Config.NUM_CLASSES):
                cls_mask = (query_labels == cls_idx)
                if not cls_mask.any():
                    continue
                
                cls_images = query_images[cls_mask]
                cls_masks = query_masks[cls_mask]
                cls_prototype = prototypes[cls_idx].detach()
                
                # Process one image at a time to save memory
                for i in range(cls_images.size(0)):
                    torch.cuda.empty_cache()
                    
                    img = cls_images[i:i+1]
                    mask = cls_masks[i:i+1]
                    
                    # Compute GradCAM guided by expert annotation
                    gradcam = model.compute_gradcam(img, cls_prototype)
                    
                    # Compute loss between model attention and expert annotation
                    exp_loss = dice_loss(gradcam, mask.unsqueeze(0))
                    class_exp_losses.append(exp_loss)
                    
                    # Save visualization
                    if save_heatmap and not saved_heatmap:
                        class_name_safe = sanitize_filename(Config.CLASSES[cls_idx])
                        save_path = os.path.join(
                            epoch_heatmap_dir, 
                            f"episode_{episode+1}_class_{cls_idx}_{class_name_safe}.png"
                        )
                        save_brats_heatmap_comparison(
                            img.squeeze(0), mask.squeeze(0), gradcam.squeeze(0),
                            Config.CLASSES[cls_idx], save_path
                        )
                        saved_heatmap = True
            
            if not class_exp_losses:
                continue
            
            exp_loss = torch.stack(class_exp_losses).mean()
            total_loss = cls_loss + Config.ALPHA * exp_loss
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            torch.cuda.empty_cache()
            
            # Compute accuracy
            with torch.no_grad():
                dists = torch.cdist(query_embeddings, prototypes)
                pred_labels = torch.argmin(dists, dim=1)
                accuracy = (pred_labels == query_labels).float().mean().item()
            
            # Track metrics
            train_loss += total_loss.item()
            train_cls_loss += cls_loss.item()
            train_exp_loss += exp_loss.item()
            train_acc += accuracy
            train_total += 1
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1:3d}: Loss {total_loss.item():.4f} "
                      f"(CLS: {cls_loss.item():.4f}, EXP: {exp_loss.item():.4f}) "
                      f"Acc: {accuracy:.4f}")
            
            # Log episode results
            with open(Config.LOG_FILE, 'a') as log_file:
                log_file.write(f"{epoch+1},{episode+1},{total_loss.item():.4f},"
                              f"{cls_loss.item():.4f},{exp_loss.item():.4f},{accuracy:.4f},,\n")
        
        # Calculate averages
        if train_total > 0:
            train_loss /= train_total
            train_cls_loss /= train_total
            train_exp_loss /= train_total
            train_acc /= train_total
        
        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} (CLS: {train_cls_loss:.4f}, EXP: {train_exp_loss:.4f})")
        print(f"Train Accuracy: {train_acc:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                if images.size(0) == 0:
                    continue
                
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
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'prototypes': prototypes,
                'val_acc': val_acc,
                'classes': Config.CLASSES,
                # Remove 'config': Config to avoid pickling issues
            }, Config.MODEL_PATH)
            print(f"💾 Saved new best model! Val Acc: {val_acc:.4f}")
        
        # Log epoch results
        with open(Config.LOG_FILE, 'a') as log_file:
            log_file.write(f"{epoch+1},summary,{train_loss:.4f},{train_cls_loss:.4f},"
                          f"{train_exp_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")
        
        scheduler.step(val_loss)
    
    return model, prototypes

# --- Evaluation function ---
def evaluate_model(model, prototypes, data_loader, dataset_name="Test"):
    """Evaluate model on dataset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"\n📊 Evaluating on {dataset_name} set...")
    
    with torch.no_grad():
        for images, labels, _ in data_loader:
            if images.size(0) == 0:
                continue
                
            images = images.to(Config.DEVICE)
            
            embeddings = model(images)
            dists = torch.cdist(embeddings, prototypes)
            preds = torch.argmin(dists, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    if len(all_preds) == 0:
        print("No predictions to evaluate!")
        return
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Print classification report
    print(f"\n{dataset_name} Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=Config.CLASSES)
    print(report)
    
    # Save report
    report_path = os.path.join(Config.MODEL_DIR, f'{dataset_name.lower()}_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(Config.CLASSES))
    plt.xticks(tick_marks, Config.CLASSES, rotation=45)
    plt.yticks(tick_marks, Config.CLASSES)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_path = os.path.join(Config.MODEL_DIR, f'{dataset_name.lower()}_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.show()
    
    return all_preds, all_labels

# --- Test on blind validation set ---
def test_on_blind_validation(model, prototypes, test_dir):
    """Test trained model on blind validation set (no segmentation masks)"""
    print(f"\n🧪 Testing on blind validation set...")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset (no segmentation required)
    test_dataset = BraTSDataset(test_dir, transform=transform, split='test', require_segmentation=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model.eval()
    predictions = []
    case_predictions = {}
    
    print(f"Found {len(test_dataset)} test slices")
    
    with torch.no_grad():
        for images, _, _ in test_loader:
            if images.size(0) == 0:
                continue
                
            images = images.to(Config.DEVICE)
            
            embeddings = model(images)
            dists = torch.cdist(embeddings, prototypes)
            preds = torch.argmin(dists, dim=1)
            
            predictions.extend(preds.cpu().numpy())
    
    # Group predictions by case
    for i, slice_id in enumerate(test_dataset.valid_slices):
        if i < len(predictions):
            case_name = test_dataset.slice_info[slice_id]['case_name']
            if case_name not in case_predictions:
                case_predictions[case_name] = []
            case_predictions[case_name].append(predictions[i])
    
    # Analyze case-level predictions
    print(f"\n📊 Test Results Summary:")
    print(f"Total test cases: {len(case_predictions)}")
    
    case_dominant_predictions = {}
    for case_name, slice_preds in case_predictions.items():
        # Determine dominant prediction for this case
        pred_counts = np.bincount(slice_preds, minlength=Config.NUM_CLASSES)
        dominant_pred = np.argmax(pred_counts)
        case_dominant_predictions[case_name] = {
            'dominant_class': Config.CLASSES[dominant_pred],
            'class_distribution': pred_counts,
            'total_slices': len(slice_preds)
        }
    
    # Print case-level results
    class_case_counts = {cls: 0 for cls in Config.CLASSES}
    for case_name, pred_info in case_dominant_predictions.items():
        dominant_class = pred_info['dominant_class']
        class_case_counts[dominant_class] += 1
        print(f"{case_name}: {dominant_class} ({pred_info['total_slices']} slices)")
    
    print(f"\n📈 Case-level Predictions:")
    for class_name, count in class_case_counts.items():
        percentage = (count / len(case_predictions)) * 100
        print(f"  {class_name}: {count} cases ({percentage:.1f}%)")
    
    # Save test results
    test_results_path = os.path.join(Config.MODEL_DIR, 'blind_test_results.txt')
    with open(test_results_path, 'w') as f:
        f.write("BraTS Blind Validation Test Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total test cases: {len(case_predictions)}\n\n")
        
        f.write("Case-level Predictions:\n")
        for class_name, count in class_case_counts.items():
            percentage = (count / len(case_predictions)) * 100
            f.write(f"  {class_name}: {count} cases ({percentage:.1f}%)\n")
        
        f.write("\nDetailed Case Results:\n")
        for case_name, pred_info in case_dominant_predictions.items():
            f.write(f"{case_name}: {pred_info['dominant_class']} "
                    f"({pred_info['total_slices']} slices)\n")
    
    print(f"\n✅ Test results saved to: {test_results_path}")
    
    return case_dominant_predictions

# --- Main execution ---
def run_brats_pipeline():
    print("🧠 BraTS 2020 Explainable Few-Shot Learning Pipeline")
    print("=" * 70)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create training dataset (with segmentation)  
    print("📚 Loading training dataset...")
    full_train_dataset = BraTSDataset(Config.TRAIN_DIR, transform=transform, split='train', require_segmentation=True)
    
    if len(full_train_dataset) == 0:
        print("❌ No training samples found! Check your data directory.")
        return None, None
    
    # Split training data into train/validation (80/20)
    print("📊 Creating 80/20 train/validation split...")
    train_size = int(0.8 * len(full_train_dataset.valid_slices))
    val_size = len(full_train_dataset.valid_slices) - train_size
    
    # Shuffle and split slice indices
    import random
    random.seed(Config.SEED)
    shuffled_slices = full_train_dataset.valid_slices.copy()
    random.shuffle(shuffled_slices)
    
    # Create train and validation slice lists
    train_slices = shuffled_slices[:train_size]
    val_slices = shuffled_slices[train_size:]
    
    # Create separate dataset objects for train and validation
    train_dataset = BraTSDataset.__new__(BraTSDataset)
    train_dataset.__dict__ = full_train_dataset.__dict__.copy()
    train_dataset.valid_slices = train_slices
    train_dataset.split = 'train_split'
    
    # Rebuild class_slices for train split
    train_dataset.class_slices = {cls_idx: [] for cls_idx in range(Config.NUM_CLASSES)}
    for slice_path in train_slices:
        cls_idx = train_dataset.slice_info[slice_path]['class_idx']
        train_dataset.class_slices[cls_idx].append(slice_path)
    
    val_dataset = BraTSDataset.__new__(BraTSDataset)
    val_dataset.__dict__ = full_train_dataset.__dict__.copy()
    val_dataset.valid_slices = val_slices
    val_dataset.split = 'val_split'
    
    # Create validation loader
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Create episodic training sampler
    n_episodes = 60  # episodes per epoch
    train_episodes = FewShotSampler(
        train_dataset, n_episodes=n_episodes, 
        n_shot=Config.N_SHOT, n_query=Config.N_QUERY
    )
    
    print("✅ Dataset preparation complete!")
    print(f"Training slices: {len(train_slices)}")
    print(f"Validation slices: {len(val_slices)}")
    print(f"Episodes per epoch: {n_episodes}")
    
    # Train model
    print("\n🚀 Starting training with expert-guided explainability...")
    model, prototypes = train_brats_model(train_episodes, val_loader)
    
    # Load best model for evaluation
    checkpoint = torch.load(Config.MODEL_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    prototypes = checkpoint['prototypes']
    
    # Final evaluation on validation set
    print("\n🎯 Final Evaluation on Validation Set...")
    evaluate_model(model, prototypes, val_loader, "Validation")
    
    # Test on blind validation set (no segmentation masks)
    print("\n🧪 Testing on Blind Validation Set...")
    test_results = test_on_blind_validation(model, prototypes, Config.VAL_DIR)
    
    print(f"\n✅ Training and Testing Complete!")
    print(f"Best model saved to: {Config.MODEL_PATH}")
    print(f"Training logs: {Config.LOG_FILE}")
    print(f"Heatmaps saved to: {Config.HEATMAP_DIR}")
    
    return model, prototypes

# Run the pipeline
if __name__ == "__main__":
    model, prototypes = run_brats_pipeline()