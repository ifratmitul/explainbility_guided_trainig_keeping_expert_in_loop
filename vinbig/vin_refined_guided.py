import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import time
import random
import copy
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures

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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report

# --- Utility functions ---
def sanitize_filename(name):
    """Sanitize filename by replacing problematic characters with underscores"""
    return name.replace('/', '_').replace(' ', '_').replace('\\', '_')

# --- Configuration ---
class Config:
    IMG_DIR = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train'
    CSV_LABELS = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv'
    CSV_BBOXES = CSV_LABELS  # same CSV file
    IMG_SIZE = 224
    BATCH_SIZE = 8
    ALPHA = 0.10  # weight for explanation loss
    NUM_EPOCHS = 7
    NUM_EPISDOES = 60
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CLASSES = ['Nodule/Mass', 'Pulmonary fibrosis', 'Lung Opacity']  # Classes with bounding boxes
    NUM_CLASSES = len(CLASSES)  # 3 classes
    N_SHOT = 5  # Number of examples per class for few-shot learning
    N_QUERY = 5  # Number of query examples per class for testing
    EMBEDDING_SIZE = 512  # Size of feature embeddings from the backbone
    
    # Output directories and files
    OUTPUT_DIR = 'output'
    HEATMAP_DIR = os.path.join(OUTPUT_DIR, 'heatmaps')
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    # File paths
    MODEL_PATH = os.path.join(MODEL_DIR, 'best_few_shot_model.pt')
    LOG_FILE = os.path.join(LOG_DIR, 'few_shot_training_log.txt')
    
    # Other settings
    SAVE_EPOCH_HEATMAPS = 1  # Save heatmaps every N epochs
    SAVE_EPISODE_HEATMAPS = 50  # Save heatmaps every N episodes
    SEED = 42  # Random seed for reproducibility
    ENABLE_ANOMALY_DETECTION = True  # Enable PyTorch anomaly detection for debugging

# --- Create output directories ---
os.makedirs(Config.HEATMAP_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(Config.HEATMAP_DIR, 'samples'), exist_ok=True)
os.makedirs(os.path.join(Config.HEATMAP_DIR, 'episodes'), exist_ok=True)
os.makedirs(os.path.join(Config.HEATMAP_DIR, 'validation'), exist_ok=True)

# --- Set random seeds for reproducibility ---
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

# If anomaly detection is enabled
if Config.ENABLE_ANOMALY_DETECTION:
    torch.autograd.set_detect_anomaly(True)
    print("PyTorch anomaly detection enabled")

# --- Multi-Class Dataset ---
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
            # Get all class names for this image
            img_classes = self.df[self.df['image_id'] == img_id]['class_name'].tolist()
            
            # Check if any of our target classes are in this image
            found_classes = [c for c in img_classes if c in self.classes]
            
            if found_classes:
                # If multiple of our target classes are present, use the first one
                # (We're simplifying to one class per image for this implementation)
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
        
        # Create binary mask for bounding boxes
        mask = Image.new('L', (dicom.Columns, dicom.Rows), 0)
        
        # Get the target class name for this image
        target_class = self.classes[label]
        
        # Draw bounding boxes for the target class
        draw = ImageDraw.Draw(mask)
        boxes = self.df[(self.df['image_id'] == image_id) & 
                      (self.df['class_name'] == target_class)]
        
        for _, row in boxes.iterrows():
            x0, y0, x1, y1 = int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])
            draw.rectangle([x0, y0, x1, y1], fill=255)
        
        # Resize image
        if self.transform:
            image = self.transform(image)
        
        # Resize mask with NEAREST to preserve binary values
        mask = mask.resize((Config.IMG_SIZE, Config.IMG_SIZE), resample=Image.NEAREST)
        mask = TF.to_tensor(mask).squeeze(0)  # (H, W)

        return image, torch.tensor(label, dtype=torch.long), mask

    def sample_few_shot_batch(self, n_shot, n_query):
        """
        Sample a few-shot batch consisting of:
        - n_shot support examples per class
        - n_query query examples per class
        
        Returns:
            support_images, support_labels, support_masks
            query_images, query_labels, query_masks
        """
        support_images, support_labels, support_masks = [], [], []
        query_images, query_labels, query_masks = [], [], []
        
        # For each class
        for cls_idx in range(len(self.classes)):
            # Get all images of this class
            class_image_ids = self.class_images[cls_idx]
            
            # If we don't have enough examples, use all available with replacement
            if len(class_image_ids) < n_shot + n_query:
                sampled_ids = random.choices(class_image_ids, k=n_shot + n_query)
            else:
                # Sample without replacement
                sampled_ids = random.sample(class_image_ids, n_shot + n_query)
            
            # Split into support and query sets
            support_ids = sampled_ids[:n_shot]
            query_ids = sampled_ids[n_shot:n_shot + n_query]
            
            # Get examples for the support set
            for img_id in support_ids:
                idx = self.image_ids.index(img_id)
                img, label, mask = self[idx]
                support_images.append(img)
                support_labels.append(label)
                support_masks.append(mask)
            
            # Get examples for the query set
            for img_id in query_ids:
                idx = self.image_ids.index(img_id)
                img, label, mask = self[idx]
                query_images.append(img)
                query_labels.append(label)
                query_masks.append(mask)
        
        # Convert to tensors
        support_images = torch.stack(support_images)
        support_labels = torch.stack(support_labels)
        support_masks = torch.stack(support_masks)
        
        query_images = torch.stack(query_images)
        query_labels = torch.stack(query_labels)
        query_masks = torch.stack(query_masks)
        
        return (support_images, support_labels, support_masks, 
                query_images, query_labels, query_masks)

# --- Create few-shot episodic dataloaders ---
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
        # Remove the classifier
        self.features = densenet.features
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Embedding layer to get the desired embedding size
        self.embedding = nn.Linear(1024, embedding_size)
        
        # Set the target layer for GradCAM (last conv layer)
        self.target_layer = self.features.denseblock4.denselayer16.conv2
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        # Global average pooling
        pooled = self.avgpool(features)
        # Flatten
        flat = torch.flatten(pooled, 1)
        # Embedding
        embeddings = self.embedding(flat)
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def compute_gradcam(self, image, prototype):
        """
        Compute GradCAM for a single image based on the given prototype.
        Uses the true label prototype for guidance rather than the predicted class.
        
        Args:
            image: Single image tensor with shape [1, C, H, W]
            prototype: Class prototype tensor with shape [D]
            
        Returns:
            GradCAM heatmap with shape [1, 1, H, W]
        """
        # Ensure we're in eval mode temporarily
        original_mode = self.training
        self.eval()
        
        # Initialize output tensor
        gradcam = torch.zeros(1, 1, Config.IMG_SIZE, Config.IMG_SIZE, device=image.device)
        
        try:
            # Make a copy of the image that requires gradients
            image_for_grad = image.clone().detach().requires_grad_(True)
            
            # Store activations and gradients
            activations = None
            gradients = None
            
            # Define hooks
            def forward_hook(module, input, output):
                nonlocal activations
                activations = output.clone()
                
            def backward_hook(module, grad_input, grad_output):
                nonlocal gradients
                gradients = grad_output[0].clone()
            
            # Register hooks
            forward_handle = self.target_layer.register_forward_hook(forward_hook)
            backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
            
            # Forward pass to get embedding
            embedding = self(image_for_grad)
            
            # Calculate prototype distance
            prototype = prototype.to(embedding.device)
            distance = torch.norm(embedding - prototype, p=2, dim=1)
            
            # Use negative distance as the score to maximize similarity
            score = -distance
            
            # Backward pass to get gradients
            self.zero_grad()
            if image_for_grad.grad is not None:
                image_for_grad.grad.zero_()
            score.backward(retain_graph=True)
            
            # Check if hooks captured the activations and gradients
            if activations is None or gradients is None:
                raise ValueError("Failed to capture activations or gradients for GradCAM")
            
            # Calculate GradCAM
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            
            # Apply ReLU to focus on positive contributions
            cam = F.relu(cam)
            
            # Resize to match input size
            cam = F.interpolate(cam, size=(Config.IMG_SIZE, Config.IMG_SIZE), 
                              mode='bilinear', align_corners=False)
            
            # Normalize between 0 and 1
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max > cam_min:
                gradcam[0] = (cam - cam_min) / (cam_max - cam_min)
            else:
                gradcam[0] = cam
            
            # Clean up
            forward_handle.remove()
            backward_handle.remove()
            
            # Restore original training mode
            self.train(original_mode)
            
        except Exception as e:
            # Clean up in case of error
            self.train(original_mode)
            torch.cuda.empty_cache()
            
            # Fail explicitly for guided training
            raise ValueError(f"GradCAM computation failed: {str(e)}")
        
        return gradcam

# --- Dice Loss ---
def dice_loss(pred, target, epsilon=1e-6):
    # Create copies to avoid in-place operations
    pred_flat = pred.clone().view(-1)
    target_flat = target.clone().view(-1)
    
    # Calculate intersection and union
    intersection = torch.sum(pred_flat * target_flat)
    pred_sum = torch.sum(pred_flat)
    target_sum = torch.sum(target_flat)
    
    # Return Dice loss
    return 1 - (2 * intersection + epsilon) / (pred_sum + target_sum + epsilon)

# --- Prototypical Loss ---
def prototypical_loss(query_embeddings, support_embeddings, query_labels, support_labels, n_classes):
    """
    Compute the prototypical loss.
    
    Args:
        query_embeddings: Embeddings of query samples [n_query*n_classes, D]
        support_embeddings: Embeddings of support samples [n_support*n_classes, D]
        query_labels: Labels of query samples [n_query*n_classes]
        support_labels: Labels of support samples [n_support*n_classes]
        n_classes: Number of classes
    
    Returns:
        Loss value and computed prototypes
    """
    # Compute prototypes
    prototypes = torch.zeros(n_classes, support_embeddings.shape[1], device=support_embeddings.device)
    for c in range(n_classes):
        mask = support_labels == c
        if mask.sum() > 0:
            prototypes[c] = support_embeddings[mask].mean(0)
    
    # Compute distances between query samples and prototypes
    dists = torch.cdist(query_embeddings, prototypes)
    
    # Log softmax of negative distances (smaller distance = higher probability)
    log_p_y = F.log_softmax(-dists, dim=1)
    
    # Target log likelihood
    target_inds = query_labels
    loss = -log_p_y.gather(1, target_inds.unsqueeze(1)).squeeze().mean()
    
    return loss, prototypes

# --- Save heatmap visualization ---
def save_heatmap_comparison(image, mask, gradcam, class_name, save_path):
    """
    Save a comparison of original image, ground truth mask, and GradCAM heatmap
    
    Args:
        image: Image tensor [C, H, W]
        mask: Ground truth mask tensor [H, W]
        gradcam: GradCAM heatmap tensor of any shape
        class_name: Name of the class
        save_path: Path to save the figure
    """
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Process image for display - ensure it's detached and on CPU
    img = image.detach().cpu().permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Class: {class_name}")
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    # Ensure mask is properly prepared for visualization
    mask_np = mask.detach().cpu().numpy()
    plt.imshow(mask_np, cmap='Reds', alpha=0.5)
    plt.title("Ground Truth Mask")
    plt.axis('off')
    
    # GradCAM
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    
    # Convert gradcam to numpy and ensure it's 2D
    # First detach from computation graph and move to CPU
    gradcam_cpu = gradcam.detach().cpu()
    
    # Convert to numpy array
    gradcam_np = gradcam_cpu.numpy()
    
    # Check dimensions and squeeze if needed
    while gradcam_np.ndim > 2:
        gradcam_np = np.squeeze(gradcam_np, axis=0)
    
    # Now it should be a 2D array suitable for imshow
    plt.imshow(gradcam_np, cmap='jet', alpha=0.7)
    plt.title(f"GradCAM (True Label)")
    plt.axis('off')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# --- Training function ---
def train_prototypical_model(train_episodes, val_loader):
    # Initialize model
    model = PrototypicalNetwork(embedding_size=Config.EMBEDDING_SIZE)
    model = model.to(Config.DEVICE)
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Initialize best metrics
    best_val_acc = 0.0
    
    # Create log file
    os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
    with open(Config.LOG_FILE, 'w') as log_file:
        log_file.write("Epoch,Episode,Loss,CLS_Loss,EXP_Loss,Acc,Val_Loss,Val_Acc\n")
    
    # Gradually increase the weight of the explanation loss
    alpha_schedule = np.linspace(0.1, Config.ALPHA, Config.NUM_EPOCHS)
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()
        # current_alpha = alpha_schedule[epoch]
        current_alpha = Config.ALPHA
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_exp_loss = 0.0
        train_acc = 0.0
        train_total = 0
        
        # Create epoch directory for heatmaps
        epoch_heatmap_dir = os.path.join(Config.HEATMAP_DIR, 'episodes', f"epoch_{epoch+1}")
        os.makedirs(epoch_heatmap_dir, exist_ok=True)
        
        # Training episodes
        for episode, (support_images, support_labels, support_masks,
                     query_images, query_labels, query_masks) in enumerate(train_episodes):
            # Move data to device
            support_images = support_images.to(Config.DEVICE)
            support_labels = support_labels.to(Config.DEVICE)
            support_masks = support_masks.to(Config.DEVICE)
            query_images = query_images.to(Config.DEVICE)
            query_labels = query_labels.to(Config.DEVICE)
            query_masks = query_masks.to(Config.DEVICE)
            
            # Clear cached memory
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
            
            # Process one class at a time to save memory
            class_exp_losses = []
            
            # Save one heatmap per episode at specified intervals
            save_heatmap = (episode % Config.SAVE_EPISODE_HEATMAPS == 0) or (epoch % Config.SAVE_EPOCH_HEATMAPS == 0 and episode == 0)
            saved_heatmap = False
            
            for cls_idx in range(Config.NUM_CLASSES):
                # Get query images of this class
                cls_mask = (query_labels == cls_idx)
                if not cls_mask.any():
                    continue
                    
                cls_images = query_images[cls_mask]
                cls_masks = query_masks[cls_mask]
                
                # Get prototype for TRUE CLASS (not predicted class)
                cls_prototype = prototypes[cls_idx].detach()  # Detach to save memory
                
                # Process in mini-batches to avoid OOM
                mini_batch_size = 2  # Process only 2 images at a time
                
                for i in range(0, cls_images.size(0), mini_batch_size):
                    # Clear cache between mini-batches
                    torch.cuda.empty_cache()
                    
                    # Get mini-batch
                    end_idx = min(i + mini_batch_size, cls_images.size(0))
                    mini_batch_images = cls_images[i:end_idx]
                    mini_batch_masks = cls_masks[i:end_idx]
                    
                    # Process one image at a time
                    for j in range(mini_batch_images.size(0)):
                        # Clear cache before each computation
                        torch.cuda.empty_cache()
                        
                        # Get single image and mask
                        img = mini_batch_images[j:j+1]
                        mask = mini_batch_masks[j:j+1]
                        
                        # Compute GradCAM using TRUE CLASS prototype (key change)
                        # This ensures we're guiding the model to focus on regions for the correct class
                        gradcam = model.compute_gradcam(img, cls_prototype)
                        
                        # If GradCAM fails, training should fail as well
                        if gradcam is None:
                            raise ValueError("GradCAM computation failed - cannot continue guided training")
                            
                        # Compute Dice loss for this single image
                        img_exp_loss = dice_loss(gradcam, mask.unsqueeze(0))
                        class_exp_losses.append(img_exp_loss)
                        
                        # Save one heatmap visualization if requested
                        if save_heatmap and not saved_heatmap:
                            class_name_safe = sanitize_filename(Config.CLASSES[cls_idx])
                            save_path = os.path.join(
                                epoch_heatmap_dir, 
                                f"episode_{episode+1}_class_{cls_idx}_{class_name_safe}.png"
                            )
                            save_heatmap_comparison(
                                img.squeeze(0),
                                mask.squeeze(0),
                                gradcam.squeeze(0),
                                Config.CLASSES[cls_idx],
                                save_path
                            )
                            saved_heatmap = True
                
                # Clear memory after processing each class
                torch.cuda.empty_cache()
            
            # Calculate average explanation loss
            # This must have values for guided training to work
            if not class_exp_losses:
                raise ValueError("No explanation losses computed - guided training cannot continue")
                
            exp_loss = torch.stack(class_exp_losses).mean()
            
            # Total loss with scaled explanation loss
            loss = cls_loss + current_alpha * exp_loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            # Free up memory
            torch.cuda.empty_cache()
            
            # Compute accuracy
            with torch.no_grad():
                # Distances to prototypes
                dists = torch.cdist(query_embeddings, prototypes)
                # Predicted classes
                pred_labels = torch.argmin(dists, dim=1)
                # Accuracy
                accuracy = (pred_labels == query_labels).float().mean().item()
            
            # Track metrics
            train_loss += loss.item()
            train_cls_loss += cls_loss.item()
            train_exp_loss += exp_loss.item()
            train_acc += accuracy
            train_total += 1
            
            # Print progress
            if (episode + 1) % 5 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Episode {episode+1}/{len(train_episodes)}, "
                      f"Loss: {loss.item():.4f} (CLS: {cls_loss.item():.4f}, EXP: {exp_loss.item():.4f}), "
                      f"Acc: {accuracy:.4f}, Alpha: {current_alpha:.2f}, "
                      f"Time: {elapsed:.2f}s")
            
            # Log episode results
            with open(Config.LOG_FILE, 'a') as log_file:
                log_file.write(f"{epoch+1},{episode+1},{loss.item():.4f},{cls_loss.item():.4f},"
                              f"{exp_loss.item():.4f},{accuracy:.4f},,\n")
        
        # Calculate average training metrics
        train_loss /= train_total
        train_cls_loss /= train_total
        train_exp_loss /= train_total
        train_acc /= train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_total = 0
        
        with torch.no_grad():
            # Process each batch in the validation loader
            for images, labels, _ in val_loader:
                # Move to device
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                # Get embeddings
                embeddings = model(images)
                
                # Compute distances to prototypes
                dists = torch.cdist(embeddings, prototypes)
                
                # Predicted classes
                pred_labels = torch.argmin(dists, dim=1)
                
                # Compute loss (cross entropy from log softmax of negative distances)
                log_p_y = F.log_softmax(-dists, dim=1)
                batch_loss = F.nll_loss(log_p_y, labels)
                
                # Compute accuracy
                accuracy = (pred_labels == labels).float().mean().item()
                
                # Track metrics
                val_loss += batch_loss.item()
                val_acc += accuracy
                val_total += 1
        
        # Calculate average validation metrics
        val_loss /= val_total
        val_acc /= val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} (CLS: {train_cls_loss:.4f}, EXP: {train_exp_loss:.4f}), "
              f"Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Log epoch results
        with open(Config.LOG_FILE, 'a') as log_file:
            log_file.write(f"{epoch+1},summary,{train_loss:.4f},{train_cls_loss:.4f},{train_exp_loss:.4f},"
                          f"{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'prototypes': prototypes,
                'val_acc': val_acc,
            }, Config.MODEL_PATH)
            print(f"Saved new best model with val_acc: {val_acc:.4f}")
            
            # Also save a copy with epoch number
            epoch_model_path = os.path.join(
                Config.MODEL_DIR, 
                f"best_model_epoch_{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'prototypes': prototypes,
                'val_acc': val_acc,
            }, epoch_model_path)
        
        # Generate and save validation heatmaps at end of each epoch
        # if epoch % Config.SAVE_EPOCH_HEATMAPS == 0:
        #     generate_validation_heatmaps(model, prototypes, val_loader, epoch)
    
    # Load the best model for evaluation
    checkpoint = torch.load(Config.MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    prototypes = checkpoint['prototypes']
    
    # Generate final validation heatmaps
    # generate_validation_heatmaps(model, prototypes, val_loader, Config.NUM_EPOCHS, prefix="final")
    
    return model, prototypes

# --- Generate validation heatmaps ---
def generate_validation_heatmaps(model, prototypes, data_loader, epoch, num_samples=3, prefix=""):
    """Generate and save heatmaps for validation samples"""
    model.eval()
    
    # Create directory
    heatmap_dir = os.path.join(Config.HEATMAP_DIR, 'validation', f"{prefix}_epoch_{epoch+1}")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Find samples from each class
    class_samples = {i: [] for i in range(Config.NUM_CLASSES)}
    samples_needed = num_samples * Config.NUM_CLASSES
    
    for images, labels, masks in data_loader:
        for i, label in enumerate(labels):
            class_idx = label.item()
            if len(class_samples[class_idx]) < num_samples:
                class_samples[class_idx].append((
                    images[i:i+1].to(Config.DEVICE),
                    label.item(),
                    masks[i]
                ))
        
        # Check if we have enough samples
        total_samples = sum(len(samples) for samples in class_samples.values())
        if total_samples >= samples_needed:
            break
    
    # Generate heatmaps
    for cls_idx, samples in class_samples.items():
        for i, (image, label, mask) in enumerate(samples):
            try:
                # Get prediction
                with torch.no_grad():
                    embedding = model(image)
                    dists = torch.cdist(embedding, prototypes)
                    pred_class = torch.argmin(dists, dim=1).item()
                
                # Generate GradCAM using TRUE CLASS prototype
                true_prototype = prototypes[cls_idx]
                true_gradcam = model.compute_gradcam(image, true_prototype)
                
                # Generate GradCAM using PREDICTED CLASS prototype
                pred_prototype = prototypes[pred_class]
                pred_gradcam = model.compute_gradcam(image, pred_prototype)
                
                # Save comparison - True class GradCAM
                class_name_safe = sanitize_filename(Config.CLASSES[cls_idx])
                true_save_path = os.path.join(
                    heatmap_dir, 
                    f"class_{cls_idx}_{class_name_safe}_sample_{i+1}_true.png"
                )
                save_heatmap_comparison(
                    image.squeeze(0),
                    mask,
                    true_gradcam.squeeze(0),
                    f"{Config.CLASSES[cls_idx]} (True)",
                    true_save_path
                )
                
                # Save comparison - Predicted class GradCAM
                pred_class_name_safe = sanitize_filename(Config.CLASSES[pred_class])
                pred_save_path = os.path.join(
                    heatmap_dir, 
                    f"class_{cls_idx}_{class_name_safe}_sample_{i+1}_pred_{pred_class}_{pred_class_name_safe}.png"
                )
                
                # Create figure for predicted class
                plt.figure(figsize=(15, 5))
                
                # Process image for display
                img = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                # Original image
                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title(f"True: {Config.CLASSES[cls_idx]}\nPred: {Config.CLASSES[pred_class]}")
                plt.axis('off')
                
                # Ground truth mask
                plt.subplot(1, 3, 2)
                plt.imshow(img)
                plt.imshow(mask.cpu().numpy(), cmap='Reds', alpha=0.5)
                plt.title("Ground Truth Mask")
                plt.axis('off')
                
                # GradCAM for predicted class
                plt.subplot(1, 3, 3)
                plt.imshow(img)
                plt.imshow(pred_gradcam.squeeze(0).cpu().numpy(), cmap='jet', alpha=0.7)
                plt.title(f"GradCAM (Pred: {Config.CLASSES[pred_class]})")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(pred_save_path, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error generating heatmap for class {cls_idx}, sample {i}: {str(e)}")

# --- Visualization functions ---
def visualize_samples(dataset, num_samples=6):
    """Visualize samples from the dataset, trying to show all classes"""
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Try to collect samples from each class
    class_samples = {i: [] for i in range(Config.NUM_CLASSES)}
    samples_needed = min(num_samples, len(dataset))
    
    # Collect samples
    for image, label, mask in loader:
        class_idx = label.item()
        if len(class_samples[class_idx]) < samples_needed // Config.NUM_CLASSES + 1:
            class_samples[class_idx].append((image.squeeze(0), mask.squeeze(0)))
        
        # Check if we have enough samples
        total_samples = sum(len(samples) for samples in class_samples.values())
        if total_samples >= samples_needed:
            break
    
    # Visualize samples
    fig_idx = 1
    for class_idx, samples in class_samples.items():
        for image, mask in samples:
            if fig_idx > samples_needed:
                break
                
            # Denormalize the image
            img = image.permute(1, 2, 0).cpu().numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            mask = mask.cpu().numpy()
            class_name = Config.CLASSES[class_idx]
            class_name_safe = sanitize_filename(class_name)

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(img)
            axs[0].set_title(f"Class: {class_name}")
            axs[0].axis('off')

            axs[1].imshow(img)
            axs[1].imshow(mask, cmap='Reds', alpha=0.5)
            axs[1].set_title("Bounding Box Mask")
            axs[1].axis('off')

            plt.tight_layout()
            plt.show()
            
            # Also save the visualization
            samples_dir = os.path.join(Config.HEATMAP_DIR, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            plt.savefig(
                os.path.join(samples_dir, f"sample_{fig_idx}_{class_name_safe}.png"),
                bbox_inches='tight'
            )
            plt.close()
            
            fig_idx += 1

def visualize_gradcam(model, prototypes, data_loader, num_samples=6):
    """Visualize GradCAM for samples from each class"""
    model.eval()
    
    # Try to collect samples from each class
    class_samples = {i: [] for i in range(Config.NUM_CLASSES)}
    samples_needed = min(num_samples, len(data_loader.dataset))
    
    # Find samples
    for images, labels, masks in data_loader:
        for i, label in enumerate(labels):
            class_idx = label.item()
            if len(class_samples[class_idx]) < samples_needed // Config.NUM_CLASSES + 1:
                class_samples[class_idx].append((
                    images[i:i+1].to(Config.DEVICE), 
                    masks[i].cpu().numpy(),
                    class_idx
                ))
        
        # Check if we have enough samples
        total_samples = sum(len(samples) for samples in class_samples.values())
        if total_samples >= samples_needed:
            break
    
    # Create directory for visualization
    vis_dir = os.path.join(Config.HEATMAP_DIR, "final_visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize GradCAM for each sample
    fig_idx = 1
    for class_idx, samples in class_samples.items():
        for image, mask, _ in samples:
            if fig_idx > samples_needed:
                break
            
            # Get prediction
            with torch.no_grad():
                # Get embeddings
                embedding = model(image)
                # Distances to prototypes
                dists = torch.cdist(embedding, prototypes)
                # Predicted class and probability
                pred_class = torch.argmin(dists, dim=1).item()
                probs = F.softmax(-dists, dim=1)
                prob = probs[0, pred_class].item()
            
            # Compute GradCAM for TRUE class (not predicted)
            try:
                true_gradcam = model.compute_gradcam(image, prototypes[class_idx])
                true_gradcam_np = true_gradcam.squeeze().cpu().numpy()
                
                # Also compute GradCAM for predicted class (for comparison)
                pred_gradcam = model.compute_gradcam(image, prototypes[pred_class])
                pred_gradcam_np = pred_gradcam.squeeze().cpu().numpy()
            except Exception as e:
                print(f"GradCAM computation failed: {str(e)}")
                true_gradcam_np = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE))
                pred_gradcam_np = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE))
            
            # Process image for display
            img = image.squeeze().cpu().permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            # Create visualization with both true and predicted GradCAM
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            
            # Original image
            axs[0].imshow(img)
            true_class_name = Config.CLASSES[class_idx]
            pred_class_name = Config.CLASSES[pred_class]
            axs[0].set_title(f"True: {true_class_name}\nPred: {pred_class_name} ({prob:.2f})")
            axs[0].axis('off')
            
            # Ground truth mask
            axs[1].imshow(img)
            axs[1].imshow(mask, cmap='Reds', alpha=0.5)
            axs[1].set_title("Ground Truth Mask")
            axs[1].axis('off')
            
            # GradCAM for true class
            axs[2].imshow(img)
            axs[2].imshow(true_gradcam_np, cmap='jet', alpha=0.7)
            axs[2].set_title(f"GradCAM (True: {true_class_name})")
            axs[2].axis('off')
            
            # GradCAM for predicted class
            axs[3].imshow(img)
            axs[3].imshow(pred_gradcam_np, cmap='jet', alpha=0.7)
            axs[3].set_title(f"GradCAM (Pred: {pred_class_name})")
            axs[3].axis('off')
            
            # Ensure path exists
            os.makedirs(vis_dir, exist_ok=True)
            
            # Save figure
            true_class_safe = sanitize_filename(true_class_name)
            pred_class_safe = sanitize_filename(pred_class_name)
            plt.tight_layout()
            plt.savefig(
                os.path.join(vis_dir, f"gradcam_{fig_idx}_{true_class_safe}_pred_{pred_class_safe}.png"),
                bbox_inches='tight'
            )
            plt.show()
            
            fig_idx += 1

# --- Evaluation function ---
def evaluate_model(model, prototypes, data_loader):
    """Evaluate model on the dataset"""
    model.eval()
    
    # Collect predictions and ground truth
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(Config.DEVICE)
            
            # Get embeddings
            embeddings = model(images)
            
            # Calculate distances to prototypes
            dists = torch.cdist(embeddings, prototypes)
            
            # Get predictions
            preds = torch.argmin(dists, dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Print classification report
    class_names = Config.CLASSES
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    
    # Save report to file
    with open(os.path.join(Config.MODEL_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
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
    plt.savefig(os.path.join(Config.MODEL_DIR, 'confusion_matrix.png'))
    plt.show()
    
    return all_preds, all_labels

# --- Main execution ---
def run_pipeline():
    # --- Read and clean CSV ---
    print("Reading CSV data...")
    df_labels = pd.read_csv(Config.CSV_LABELS)
    df_labels['image_id'] = df_labels['image_id'].astype(str).str.strip()
    df_labels['class_name'] = df_labels['class_name'].astype(str).str.strip()

    # --- Define transforms ---
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Build dataset with only the specified classes ---
    print(f"Building dataset with classes: {Config.CLASSES}...")
    full_dataset = ChestXrayMultiClassDataset(
        Config.IMG_DIR,
        df_labels,
        classes=Config.CLASSES,
        transform=transform
    )

    # --- Split into train, validation, and test sets ---
    print("Splitting dataset...")
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    # Random split into three sets
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset split - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # --- Create dataloaders ---
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # --- Create episodic training sampler ---
    n_episodes = Config.NUM_EPISDOES  # Number of episodes per epoch
    train_episodes = FewShotSampler(
        full_dataset,  # Use the full dataset for sampling episodes
        n_episodes=n_episodes,
        n_shot=Config.N_SHOT,
        n_query=Config.N_QUERY
    )

    print("✅ Dataset preparation complete.")

    # --- Visualize some samples ---
    print("Visualizing sample images...")
    visualize_samples(full_dataset, num_samples=6)

    # --- Train the model ---
    print("Starting model training...")
    model, prototypes = train_prototypical_model(train_episodes, val_loader)

    # --- Final Evaluation on Test Set ---
    print("Final evaluation on test set...")
    evaluate_model(model, prototypes, test_loader)

    # --- Visualize GradCAM for better understanding ---
    print("Visualizing GradCAM for final model...")
    # visualize_gradcam(model, prototypes, test_loader, num_samples=9)
    
    # Save final model and prototypes
    final_model_path = os.path.join(Config.MODEL_DIR, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'prototypes': prototypes,
        'classes': Config.CLASSES
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

    return model, prototypes

if __name__ == "__main__":
    model, prototypes = run_pipeline()