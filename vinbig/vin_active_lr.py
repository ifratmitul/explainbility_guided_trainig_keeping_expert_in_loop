import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import time
import random
import copy
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import pydicom
import torchvision.transforms.functional as TF
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import entropy

import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np

# --- Configuration ---
class Config:
    IMG_DIR = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train'
    CSV_LABELS = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv'
    IMG_SIZE = 224
    BATCH_SIZE = 16
    ALPHA = 0.10  # weight for explanation loss in retraining
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CLASSES = ['Nodule/Mass', 'Pulmonary fibrosis', 'Lung Opacity']
    NUM_CLASSES = len(CLASSES)
    N_SHOT = 5
    N_QUERY = 5
    EMBEDDING_SIZE = 512
    
    # Active Learning Configuration
    INITIAL_LABELED_SIZE = 200  # Start with small labeled set
    ACQUISITION_BATCH_SIZE = 100  # K samples to query per iteration
    MAX_AL_ITERATIONS = 2  # Number of active learning rounds
    LAMBDA_UNCERTAINTY = .3  # λ parameter balancing uncertainty vs misalignment
    
    # Fine-tuning Configuration
    FINETUNE_EPOCHS = 1  # REDUCED to prevent overfitting
    FINETUNE_EPISODES = 20
    FINETUNE_LR = 1e-4  # Learning rate for fine-tuning
    WEIGHT_DECAY = 1e-5  
    
    # Test Set Configuration
    TEST_SET_SIZE = 50  # Fixed test set size (50 per class)
    
    # Model paths
    BASELINE_MODEL_PATH = '/kaggle/input/random_sampling_baseline/pytorch/default/1/best_prototypical_model.pt'
    OUTPUT_DIR = 'output_active_learning'
    LOG_FILE = os.path.join(OUTPUT_DIR, 'active_learning_log.csv')
    FINETUNE_LOG_FILE = os.path.join(OUTPUT_DIR, 'fine_tuning_log.csv')
    
    SEED = 42

# Create directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# Set random seed
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)

# --- Dataset with BALANCED few-shot sampling ---
class ChestXrayMultiClassDataset(Dataset):
    def __init__(self, img_dir, df, classes=Config.CLASSES, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.classes = classes
        self.transform = transform
        
        # Map class names to indices
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        
        # Get unique image IDs
        all_image_ids = self.df['image_id'].unique()
        
        # Filter to only include images that have at least one of our target classes
        self.image_ids = []
        self.labels = {}  # Store class index for each image
        
        for img_id in all_image_ids:
            img_classes = self.df[self.df['image_id'] == img_id]['class_name'].tolist()
            found_classes = [c for c in img_classes if c in self.classes]
            
            if found_classes:
                target_class = found_classes[0]
                self.image_ids.append(img_id)
                self.labels[img_id] = self.class_to_idx[target_class]
        
        print(f"Dataset contains {len(self.image_ids)} images")
        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            count = sum(1 for img_id in self.image_ids if self.labels[img_id] == cls_idx)
            print(f"{cls_name}: {count} images")
            
        # Group images by class for easy sampling
        self.class_images = {cls_idx: [] for cls_idx in range(len(classes))}
        for img_id in self.image_ids:
            cls_idx = self.labels[img_id]
            self.class_images[cls_idx].append(img_id)

    def normalize_dicom(self, dicom):
        img = dicom.pixel_array.astype(np.float32)
        img -= np.min(img)
        img /= np.max(img)
        img *= 255.0
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = str(self.image_ids[idx]).strip()
        img_path = os.path.join(self.img_dir, image_id + '.dicom')
        
        # Load and normalize DICOM
        dicom = pydicom.dcmread(img_path)
        image = self.normalize_dicom(dicom)
        
        # Get label for this image
        label = self.labels[image_id]
        
        # Create binary mask for bounding boxes (this is the "prior mask" PM(x))
        mask = Image.new('L', (dicom.Columns, dicom.Rows), 0)
        target_class = self.classes[label]
        
        # Draw bounding boxes for the target class - THIS IS THE EXPERT ANNOTATION
        draw = ImageDraw.Draw(mask)
        boxes = self.df[(self.df['image_id'] == image_id) & 
                      (self.df['class_name'] == target_class)]
        
        for _, row in boxes.iterrows():
            x0, y0, x1, y1 = int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])
            draw.rectangle([x0, y0, x1, y1], fill=255)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Resize mask
        mask = mask.resize((Config.IMG_SIZE, Config.IMG_SIZE), resample=Image.NEAREST)
        mask = TF.to_tensor(mask).squeeze(0)  # (H, W)

        return image, torch.tensor(label, dtype=torch.long), mask

    def sample_balanced_few_shot_batch(self, n_shot, n_query, labeled_indices):
        
        support_images, support_labels, support_masks = [], [], []
        query_images, query_labels, query_masks = [], [], []
        
        # Group labeled indices by class
        class_labeled_indices = {cls_idx: [] for cls_idx in range(len(self.classes))}
        for idx in labeled_indices:
            img_id = self.image_ids[idx]
            cls_idx = self.labels[img_id]
            class_labeled_indices[cls_idx].append(idx)
        
        # print(f"  Balanced episode sampling from:")
        
        # Sample EXACTLY n_shot + n_query from each class
        for cls_idx in range(len(self.classes)):
            available_indices = class_labeled_indices[cls_idx]
            
            # print(f"    Class {self.classes[cls_idx]}: {len(available_indices)} available samples")
            
            if len(available_indices) == 0:
                print(f"    Warning: No samples for class {self.classes[cls_idx]} - skipping")
                continue
                
            total_needed = n_shot + n_query
            
            if len(available_indices) < total_needed:
                # Sample with replacement if needed
                sampled_indices = random.choices(available_indices, k=total_needed)
                print(f"    Sampling with replacement: {total_needed} samples from {len(available_indices)} available")
            else:
                # Sample without replacement
                sampled_indices = random.sample(available_indices, total_needed)
                print(f"    Sampling without replacement: {total_needed} samples")
            
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
        
        print(f"  Created balanced episode: {len(support_images)} support, {len(query_images)} query")
        print(f"  Support classes: {support_labels.tolist()}")
        print(f"  Query classes: {query_labels.tolist()}")
        
        return (support_images, support_labels, support_masks, 
                query_images, query_labels, query_masks)

# --- Prototypical Network (same as yours) ---
class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_size=Config.EMBEDDING_SIZE, pretrained=True):
        super(PrototypicalNetwork, self).__init__()
        densenet = models.densenet121(pretrained=pretrained)
        self.features = densenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(1024, embedding_size)
        self.target_layer = self.features.denseblock4.denselayer16.conv2
    
    def forward(self, x):
        features = self.features(x)
        pooled = self.avgpool(features)
        flat = torch.flatten(pooled, 1)
        embeddings = self.embedding(flat)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def compute_gradcam(self, image, prototype):
        """Compute GradCAM for explanation misalignment"""
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
            
            if activations is not None and gradients is not None:
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
            print(f"GradCAM computation failed: {e}")
            
        return gradcam

# --- Prototypical Loss ---
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

# --- Dice Loss ---
def dice_loss(pred, target, epsilon=1e-6):
    pred_flat = pred.clone().view(-1)
    target_flat = target.clone().view(-1)
    intersection = torch.sum(pred_flat * target_flat)
    pred_sum = torch.sum(pred_flat)
    target_sum = torch.sum(target_flat)
    return 1 - (2 * intersection + epsilon) / (pred_sum + target_sum + epsilon)

# --- UPDATED: Fixed Test Set Creation ---
def create_fixed_test_set(full_dataset, test_size=Config.TEST_SET_SIZE):
    """Create a stratified, fixed test set for consistent evaluation"""
    samples_per_class = test_size // Config.NUM_CLASSES
    test_indices = []
    
    print(f"Creating fixed test set with {samples_per_class} samples per class...")
    
    for class_idx in range(Config.NUM_CLASSES):
        class_image_ids = full_dataset.class_images[class_idx]
        
        if len(class_image_ids) >= samples_per_class:
            selected_ids = random.sample(class_image_ids, samples_per_class)
        else:
            selected_ids = class_image_ids
        
        for img_id in selected_ids:
            idx = full_dataset.image_ids.index(img_id)
            test_indices.append(idx)
        
        print(f"Test set - Class {Config.CLASSES[class_idx]}: {len(selected_ids)} samples")
    
    print(f"Total fixed test set size: {len(test_indices)} samples")
    return test_indices

# --- UPDATED: Proper Test Evaluation Function ---
def evaluate_model_properly(model, prototypes, test_loader):
    """
    PROPER test evaluation using fixed test set and stable prototypes
    (Same approach as basic prototypical network)
    """
    model.eval()
    correct = 0
    total = 0
    losses = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:  # Note: we ignore masks for classification
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

# --- UPDATED: Active Learning Framework with Fixed Test Set ---
class ActiveLearningFramework:
    def __init__(self, model, prototypes, full_dataset):
        self.model = model
        self.prototypes = prototypes
        self.full_dataset = full_dataset
        self.lambda_param = Config.LAMBDA_UNCERTAINTY
        
        # ADDED: Create fixed test set at initialization
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
            'test_losses': [],  # ADDED: Track test losses
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
            class_image_ids = self.full_dataset.class_images[class_idx]
            available_class_indices = []
            
            for img_id in class_image_ids:
                idx = self.full_dataset.image_ids.index(img_id)
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
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=16, shuffle=False)
        
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
        batch_size = 8
        
        for i in range(0, len(unlabeled_list), batch_size):
            batch_indices = unlabeled_list[i:i+batch_size]
            
            for idx in batch_indices:
                try:
                    image, label, expert_mask = self.full_dataset[idx]  # expert_mask is PM(x)
                    image = image.unsqueeze(0).to(Config.DEVICE)
                    
                    # Get prediction
                    with torch.no_grad():
                        embedding = self.model(image)
                        dists = torch.cdist(embedding, self.prototypes)
                        pred_class = torch.argmin(dists, dim=1).item()
                    
                    # Generate GradCAM for predicted class ŷ
                    pred_prototype = self.prototypes[pred_class]
                    gradcam = self.model.compute_gradcam(image, pred_prototype)  # CAM_ŷ(x)
                    gradcam_2d = gradcam.squeeze().cpu()
                    
                    # Compute Dice misalignment with expert annotation (prior mask)
                    intersection = torch.sum(gradcam_2d * expert_mask)
                    union = torch.sum(gradcam_2d) + torch.sum(expert_mask)
                    
                    if union > 0:
                        dice_similarity = (2.0 * intersection) / union
                        misalignment = 1.0 - dice_similarity  # D_exp(x)
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
        
        # if len(uncertainties) > 1:
        #     if uncertainties.max() > uncertainties.min():
        #         uncertainties = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min())
        #     if misalignments.max() > misalignments.min():
        #         misalignments = (misalignments - misalignments.min()) / (misalignments.max() - misalignments.min())
        
        # Compute composite scores
        composite_scores = self.lambda_param * uncertainties + (1 - self.lambda_param) * misalignments
        
        return composite_scores, uncertainties, misalignments
    
    def select_balanced_top_k_samples(self, composite_scores, unlabeled_indices, k):
        """
        Select top-K samples ensuring BALANCED representation
        """
        k_per_class = k // Config.NUM_CLASSES  # e.g., 15 // 3 = 5 per class
        
        # print(f"Selecting {k} samples with balanced representation ({k_per_class} per class)...")
        
        # Group unlabeled indices by class
        class_unlabeled = {cls_idx: [] for cls_idx in range(Config.NUM_CLASSES)}
        unlabeled_list = list(unlabeled_indices)
        
        for i, idx in enumerate(unlabeled_list):
            img_id = self.full_dataset.image_ids[idx]
            cls_idx = self.full_dataset.labels[img_id]
            class_unlabeled[cls_idx].append((i, idx))  # (score_index, dataset_index)
        
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
                
            # print(f"Selected {num_to_select} samples from class {Config.CLASSES[cls_idx]} (avg score: {np.mean([class_scores[i][0] for i in range(num_to_select)]):.4f})")
        
        # print(f"Total selected: {len(selected_indices)} samples")
        return selected_indices, selected_scores
    
    def visualize_selected_samples(self, selected_indices, al_iteration, num_samples=5):
        """
        Visualize selected samples with their expert annotations
        """
        print(f"Visualizing {num_samples} selected samples from AL iteration {al_iteration}...")
        
        # Create output directory for visualizations
        viz_dir = os.path.join(Config.OUTPUT_DIR, f'visualizations_iter_{al_iteration}')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Select first 5 samples (or fewer if less available)
        samples_to_viz = selected_indices[:min(num_samples, len(selected_indices))]
        
        for i, idx in enumerate(samples_to_viz):
            try:
                # Get sample data
                image, label, expert_mask = self.full_dataset[idx]
                img_id = self.full_dataset.image_ids[idx]
                class_name = Config.CLASSES[label]
                
                # Convert tensor to numpy for visualization
                if isinstance(image, torch.Tensor):
                    # Denormalize the image
                    image_np = image.clone()
                    # Reverse normalization: img = (img * std) + mean
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image_np = (image_np * std) + mean
                    image_np = torch.clamp(image_np, 0, 1)
                    image_np = image_np.permute(1, 2, 0).numpy()
                
                # Convert mask to numpy
                if isinstance(expert_mask, torch.Tensor):
                    mask_np = expert_mask.numpy()
                else:
                    mask_np = expert_mask
                
                # Create figure with subplots
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Plot 1: Original image
                axes[0].imshow(image_np)
                axes[0].set_title(f'Original Image\nID: {img_id}\nClass: {class_name}', fontsize=10)
                axes[0].axis('off')
                
                # Plot 2: Expert annotation mask
                axes[1].imshow(image_np)
                axes[1].imshow(mask_np, alpha=0.3, cmap='Reds')
                axes[1].set_title(f'Expert Annotation\n(Red overlay)', fontsize=10)
                axes[1].axis('off')
                
                # Plot 3: Bounding boxes on original image
                axes[2].imshow(image_np)
                
                # Get bounding box coordinates from CSV
                boxes = self.full_dataset.df[
                    (self.full_dataset.df['image_id'] == img_id) & 
                    (self.full_dataset.df['class_name'] == class_name)
                ]
                
                # Draw bounding boxes
                for _, row in boxes.iterrows():
                    x_min, y_min = int(row['x_min']), int(row['y_min'])
                    x_max, y_max = int(row['x_max']), int(row['y_max'])
                    width, height = x_max - x_min, y_max - y_min
                    
                    # Scale coordinates to match resized image
                    scale_x = Config.IMG_SIZE / image_np.shape[1]
                    scale_y = Config.IMG_SIZE / image_np.shape[0]
                    
                    x_min_scaled = int(x_min * scale_x)
                    y_min_scaled = int(y_min * scale_y)
                    width_scaled = int(width * scale_x)
                    height_scaled = int(height * scale_y)
                    
                    # Create rectangle
                    rect = Rectangle(
                        (x_min_scaled, y_min_scaled), 
                        width_scaled, height_scaled,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    axes[2].add_patch(rect)
                
                axes[2].set_title(f'Bounding Boxes\n(Red rectangles)', fontsize=10)
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # Save the visualization
                save_path = os.path.join(viz_dir, f'selected_sample_{i+1}_id_{img_id}_class_{label}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved: {save_path}")
                
            except Exception as e:
                print(f"Error visualizing sample {i+1} (idx {idx}): {e}")
                continue
        
        print(f"Visualizations saved to: {viz_dir}")

    def analyze_selected_sample_quality(self, selected_indices, uncertainties, misalignments, composite_scores):
        """
        Analyze the quality and characteristics of selected samples
        """
        print("=== SELECTED SAMPLE ANALYSIS ===")
        
        # Get class distribution of selected samples
        selected_classes = []
        selected_class_names = []
        
        for idx in selected_indices:
            img_id = self.full_dataset.image_ids[idx]
            label = self.full_dataset.labels[img_id]
            class_name = Config.CLASSES[label]
            selected_classes.append(label)
            selected_class_names.append(class_name)
        
        from collections import Counter
        class_dist = Counter(selected_classes)
        
        print(f"Selected samples class distribution:")
        for class_idx, count in class_dist.items():
            print(f"  {Config.CLASSES[class_idx]}: {count} samples")
        
        # Calculate statistics
        print(f"\nSample selection statistics:")
        print(f"  Average uncertainty: {np.mean(uncertainties):.4f}")
        print(f"  Average misalignment: {np.mean(misalignments):.4f}")
        print(f"  Average composite score: {np.mean(composite_scores):.4f}")
        
        # Compare with random baseline
        all_unlabeled = list(self.unlabeled_indices)
        if len(all_unlabeled) >= len(selected_indices):
            random_sample = random.sample(all_unlabeled, len(selected_indices))
            
            # Get random uncertainties (approximate)
            try:
                random_uncertainties = self.compute_classification_uncertainty(random_sample)
                
                print(f"\nComparison with random sampling:")
                print(f"  AL selected avg uncertainty: {np.mean(uncertainties):.4f}")
                print(f"  Random sample avg uncertainty: {np.mean(random_uncertainties):.4f}")
                print(f"  AL improvement: {np.mean(uncertainties) - np.mean(random_uncertainties):+.4f}")
                
                if np.mean(uncertainties) > np.mean(random_uncertainties):
                    print("  ✅ AL is selecting more uncertain samples than random")
                else:
                    print("  ⚠️ AL is NOT selecting more uncertain samples than random")
            except Exception as e:
                print(f"  Could not compute random baseline: {e}")
        else:
            print(f"\nNot enough unlabeled samples for random comparison")
    
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
        """
        UPDATED: Use fixed test set for consistent, reliable evaluation
        """
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
        """Fine-tune model with balanced episodes and adaptive learning rate"""
        print(f"Retraining model with dual-objective loss (AL Iteration {al_iteration})...")
        print(f"Using balanced few-shot episodes from {len(self.labeled_indices)} labeled samples")
        
        # Set up optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=Config.FINETUNE_LR,
            weight_decay=Config.WEIGHT_DECAY
        )
    
        # UPDATED: Use ReduceLROnPlateau instead of StepLR
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min',        # Monitor loss decrease
        #     factor=0.7,        # Reduce LR by 30% when plateau detected
        #     patience=2,        # Wait 2 epochs before reducing
        #     min_lr=1e-6,       # Don't go below this learning rate
        #     verbose=True       # Print when LR changes
        # )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=3, verbose=True
                    )
        
        # Create validation set from unlabeled data
        val_indices = random.sample(list(self.unlabeled_indices), 
                                   min(30, len(self.unlabeled_indices)))
        val_subset = Subset(self.full_dataset, val_indices)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)
        
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
            prev_loss = float('inf')  # Initialize prev_loss for early stopping
            
            for episode in range(max_episodes):
                try:
                    print(f"    Episode {episode+1}/{max_episodes}")
                    
                    # Sample balanced few-shot episode
                    episode_data = self.full_dataset.sample_balanced_few_shot_batch(
                        Config.N_SHOT, Config.N_QUERY, list(self.labeled_indices)
                    )
                    
                    if episode_data is None:
                        print(f"    Skipping episode {episode+1} - insufficient balanced data")
                        continue
                    
                    support_images, support_labels, support_masks, query_images, query_labels, query_masks = episode_data
                    
                    # Move to device
                    support_images = support_images.to(Config.DEVICE)
                    support_labels = support_labels.to(Config.DEVICE)
                    support_masks = support_masks.to(Config.DEVICE)
                    query_images = query_images.to(Config.DEVICE)
                    query_labels = query_labels.to(Config.DEVICE)
                    query_masks = query_masks.to(Config.DEVICE)
                    
                    # optimizer.zero_grad()
                    
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
                            print(f"      Warning: GradCAM failed for sample {i}: {e}")
                            exp_losses.append(torch.tensor(0.0, device=Config.DEVICE))
                    
                    if exp_losses:
                        exp_loss = torch.stack(exp_losses).mean()
                    else:
                        exp_loss = torch.tensor(0.0, device=Config.DEVICE)
                    
                    # Total loss
                    total_loss = proto_loss + Config.ALPHA * exp_loss
    
                    # Early stopping check
                    if episode > 0 and total_loss.item() > prev_loss * 1.1:
                        print(f"    Loss increasing from {prev_loss:.4f} to {total_loss.item():.4f}, stopping early")
                        break
                    
                    # Store current loss for next comparison
                    prev_loss = total_loss.item()
                    
                    # Backward pass
                    # total_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    # optimizer.step()

                    optimizer.zero_grad()
                    total_loss.backward()
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    optimizer.step()
                    
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
                    
                    batch_acc = correct / total if total > 0 else 0.0
                    print(f"      Proto={proto_loss.item():.4f}, Exp={exp_loss.item():.4f}, "
                          f"Total={total_loss.item():.4f}, Acc={batch_acc:.4f}")
                    
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
            
            # UPDATED: Step scheduler with validation loss (better signal than total loss)
            scheduler.step(val_loss)
            
            # Print current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1} Summary:")
            print(f"    Train - Proto: {avg_proto_loss:.4f}, Exp: {avg_exp_loss:.4f}, "
                  f"Total: {avg_total_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"    Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"    Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
            print(f"    Learning Rate: {current_lr:.2e}")  # Show current LR
            
            # Log with test metrics
            with open(Config.FINETUNE_LOG_FILE, 'a') as f:
                f.write(f"{al_iteration},{epoch+1},summary,{avg_proto_loss:.4f},"
                       f"{avg_exp_loss:.4f},{avg_total_loss:.4f},{train_acc:.4f},"
                       f"{val_loss:.4f},{val_acc:.4f},{test_loss:.4f},{test_acc:.4f},{current_lr:.2e}\n")
        
        self.save_model_checkpoint(al_iteration)
        print("Dual-objective retraining completed!")

    def save_model_checkpoint(self, al_iteration):
        """Save model checkpoint after each AL iteration"""
        print(f"Saving model checkpoint for AL iteration {al_iteration}...")
        
        model_save_path = os.path.join(Config.OUTPUT_DIR, f'finetuned_model_iter_{al_iteration}.pt')
        
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
        """
        UPDATED: Main active learning cycle with proper test evaluation
        """
        print("="*80)
        print("EXPLAINABILITY-GUIDED ACTIVE LEARNING FRAMEWORK")
        print("="*80)
        
        # Initialize labeled set
        self.initialize_labeled_set()
        
        # Initialize log file
        with open(Config.LOG_FILE, 'w') as f:
            f.write("Iteration,Labeled_Size,Test_Accuracy,Test_Loss,Avg_Uncertainty,Avg_Misalignment,Avg_Score\n")
        
        # UPDATED: Initial evaluation using fixed test set
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
            

            # ===== ADD VISUALIZATION AND ANALYSIS =====
            # Visualize selected samples
            self.visualize_selected_samples(selected_indices, iteration + 1, num_samples=5)
        
            # Analyze sample quality
            self.analyze_selected_sample_quality(selected_indices, uncertainties, misalignments, scores)
            # ==========================================

            # Query expert annotations
            new_labels, new_masks = self.query_expert_annotations(selected_indices)
            
            # Update labeled/unlabeled sets
            self.labeled_indices.update(selected_indices)
            self.unlabeled_indices -= set(selected_indices)
            
            # Retrain model
            self.retrain_with_dual_objective(iteration + 1)
            
            # UPDATED: Evaluate on fixed test set
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
            
            # UPDATED: Log with test metrics
            with open(Config.LOG_FILE, 'a') as f:
                f.write(f"{iteration+1},{len(self.labeled_indices)},{test_accuracy:.4f},"
                       f"{test_loss:.4f},{np.mean(uncertainties):.4f},"
                       f"{np.mean(misalignments):.4f},{np.mean(scores):.4f}\n")
        
        self.save_final_model()
        return self.metrics

    def save_final_model(self):
        """Save final active learning model"""
        print("Saving final active learning model...")
        
        final_model_path = os.path.join(Config.OUTPUT_DIR, 'final_active_learning_model.pt')
        
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

# --- UPDATED: Main execution with proper test evaluation ---
def run_complete_active_learning_experiment():
    """
    UPDATED: Run experiment with proper fixed test set evaluation
    """
    print("Loading dataset...")
    df_labels = pd.read_csv(Config.CSV_LABELS)
    df_labels['image_id'] = df_labels['image_id'].astype(str).str.strip()
    df_labels['class_name'] = df_labels['class_name'].astype(str).str.strip()

    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = ChestXrayMultiClassDataset(
        Config.IMG_DIR, df_labels, classes=Config.CLASSES, transform=transform
    )

    print("Loading pre-trained baseline model...")
    checkpoint = torch.load(Config.BASELINE_MODEL_PATH, map_location=Config.DEVICE)
    
    model = PrototypicalNetwork(embedding_size=Config.EMBEDDING_SIZE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    
    prototypes = checkpoint['prototypes'].to(Config.DEVICE)
    
    print("Initializing Active Learning Framework with fixed test set...")
    al_framework = ActiveLearningFramework(model, prototypes, full_dataset)
    
    # ADDED: Evaluate baseline model on fixed test set
    baseline_test_loss, baseline_test_acc = al_framework.evaluate_current_performance()
    print(f"Baseline model test accuracy: {baseline_test_acc:.4f}")
    print(f"Baseline model test loss: {baseline_test_loss:.4f}")
    
    print("\nRunning active learning cycle with proper test evaluation...")
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
    
    import pickle
    with open(os.path.join(Config.OUTPUT_DIR, 'final_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED WITH PROPER TEST EVALUATION!")
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
    results = run_complete_active_learning_experiment()