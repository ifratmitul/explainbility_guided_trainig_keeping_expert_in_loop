import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import glob

def load_brats_case(case_dir):
    """Load all modalities and segmentation for a BraTS case"""
    case_name = os.path.basename(case_dir)
    
    # Define file paths
    files = {
        't1': os.path.join(case_dir, f"{case_name}_t1.nii"),
        't1ce': os.path.join(case_dir, f"{case_name}_t1ce.nii"), 
        't2': os.path.join(case_dir, f"{case_name}_t2.nii"),
        'flair': os.path.join(case_dir, f"{case_name}_flair.nii"),
        'seg': os.path.join(case_dir, f"{case_name}_seg.nii")
    }
    
    # Check if all files exist
    missing_files = [k for k, v in files.items() if not os.path.exists(v)]
    if missing_files:
        print(f"Missing files for {case_name}: {missing_files}")
        return None
    
    # Load all files
    data = {}
    try:
        for modality, file_path in files.items():
            nii_img = nib.load(file_path)
            data[modality] = nii_img.get_fdata()
            
        # Add metadata
        data['case_name'] = case_name
        data['shape'] = data['t1'].shape
        
        return data
        
    except Exception as e:
        print(f"Error loading {case_name}: {e}")
        return None

def analyze_segmentation(seg_data):
    """Analyze segmentation to understand tumor composition"""
    unique_labels = np.unique(seg_data)
    label_counts = {}
    
    # BraTS label meanings
    label_meanings = {
        0: "Background/Healthy",
        1: "Necrotic and non-enhancing tumor core", 
        2: "Peritumoral edema",
        4: "GD-enhancing tumor"
    }
    
    total_voxels = seg_data.size
    
    for label in unique_labels:
        count = np.sum(seg_data == label)
        percentage = (count / total_voxels) * 100
        label_counts[label] = {
            'count': count,
            'percentage': percentage,
            'meaning': label_meanings.get(label, 'Unknown')
        }
    
    return label_counts

def classify_case_by_dominant_tumor(seg_data, min_voxels=100):
    """Classify case by dominant tumor type"""
    # Count tumor voxels (excluding background)
    tumor_counts = {}
    
    for label in [1, 2, 4]:  # Only tumor labels
        count = np.sum(seg_data == label)
        if count >= min_voxels:  # Only significant regions
            tumor_counts[label] = count
    
    if not tumor_counts:
        return None, "No significant tumor"
    
    # Find dominant tumor type
    dominant_label = max(tumor_counts.keys(), key=lambda x: tumor_counts[x])
    
    # Map to class names
    class_names = {
        1: "Necrotic Core Dominant",
        2: "Edema Dominant", 
        4: "Enhancing Tumor Dominant"
    }
    
    return dominant_label, class_names[dominant_label]

def find_best_slice_with_tumor(seg_data):
    """Find slice with most tumor content for visualization"""
    slice_tumor_counts = []
    
    # Check each axial slice (axis=2)
    for z in range(seg_data.shape[2]):
        slice_seg = seg_data[:, :, z]
        tumor_voxels = np.sum(slice_seg > 0)  # Any non-background
        slice_tumor_counts.append(tumor_voxels)
    
    # Find slice with most tumor
    best_slice_idx = np.argmax(slice_tumor_counts)
    return best_slice_idx, slice_tumor_counts[best_slice_idx]

def visualize_brats_case(case_data, slice_idx=None):
    """Visualize all modalities and segmentation for a case"""
    if slice_idx is None:
        # Find best slice automatically
        slice_idx, tumor_count = find_best_slice_with_tumor(case_data['seg'])
        print(f"Auto-selected slice {slice_idx} (tumor voxels: {tumor_count})")
    
    # Extract the slice from all modalities
    t1_slice = case_data['t1'][:, :, slice_idx]
    t1ce_slice = case_data['t1ce'][:, :, slice_idx]
    t2_slice = case_data['t2'][:, :, slice_idx]
    flair_slice = case_data['flair'][:, :, slice_idx]
    seg_slice = case_data['seg'][:, :, slice_idx]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"BraTS Case: {case_data['case_name']} - Slice {slice_idx}", fontsize=16)
    
    # Plot each modality
    axes[0, 0].imshow(t1_slice.T, cmap='gray', origin='lower')
    axes[0, 0].set_title('T1-weighted')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(t1ce_slice.T, cmap='gray', origin='lower')
    axes[0, 1].set_title('T1CE (Contrast Enhanced)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(t2_slice.T, cmap='gray', origin='lower')
    axes[0, 2].set_title('T2-weighted')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(flair_slice.T, cmap='gray', origin='lower')
    axes[1, 0].set_title('FLAIR')
    axes[1, 0].axis('off')
    
    # Segmentation overlay
    axes[1, 1].imshow(t1ce_slice.T, cmap='gray', origin='lower')
    # Create colored overlay for segmentation
    seg_colored = np.zeros((*seg_slice.shape, 4))  # RGBA
    seg_colored[seg_slice == 1] = [1, 0, 0, 0.7]    # Red for necrotic
    seg_colored[seg_slice == 2] = [0, 1, 0, 0.7]    # Green for edema
    seg_colored[seg_slice == 4] = [0, 0, 1, 0.7]    # Blue for enhancing
    
    axes[1, 1].imshow(seg_colored.transpose(1, 0, 2), origin='lower')
    axes[1, 1].set_title('Segmentation Overlay\n(Red=Necrotic, Green=Edema, Blue=Enhancing)')
    axes[1, 1].axis('off')
    
    # Pure segmentation
    axes[1, 2].imshow(seg_slice.T, cmap='viridis', origin='lower')
    axes[1, 2].set_title('Segmentation Labels')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def explore_brats_dataset(dataset_path, num_samples=5):
    """Explore BraTS dataset and find cases from different classes"""
    
    print("🧠 BraTS 2020 Dataset Explorer")
    print("=" * 50)
    
    # Find all case directories
    if 'MICCAI_BraTS2020_TrainingData' in dataset_path:
        case_pattern = os.path.join(dataset_path, "BraTS20_Training_*")
    else:
        case_pattern = os.path.join(dataset_path, "*", "BraTS20_Training_*")
    
    case_dirs = glob.glob(case_pattern)
    print(f"Found {len(case_dirs)} cases")
    
    if len(case_dirs) == 0:
        print("❌ No cases found! Check your dataset path.")
        print("Expected structure: .../MICCAI_BraTS2020_TrainingData/BraTS20_Training_XXX/")
        return
    
    # Analyze cases and group by class
    class_cases = defaultdict(list)
    analyzed_cases = []
    
    print("\n📊 Analyzing cases...")
    
    # Sample a subset for analysis (to speed up)
    sample_dirs = random.sample(case_dirs, min(50, len(case_dirs)))
    
    for i, case_dir in enumerate(sample_dirs):
        print(f"Analyzing case {i+1}/{len(sample_dirs)}: {os.path.basename(case_dir)}")
        
        # Load case data
        case_data = load_brats_case(case_dir)
        if case_data is None:
            continue
        
        # Analyze segmentation
        seg_analysis = analyze_segmentation(case_data['seg'])
        dominant_label, class_name = classify_case_by_dominant_tumor(case_data['seg'])
        
        if dominant_label is not None:
            case_info = {
                'case_dir': case_dir,
                'case_data': case_data,
                'dominant_label': dominant_label,
                'class_name': class_name,
                'seg_analysis': seg_analysis
            }
            
            class_cases[class_name].append(case_info)
            analyzed_cases.append(case_info)
    
    # Print summary
    print(f"\n📈 Dataset Summary:")
    print(f"Total analyzed cases: {len(analyzed_cases)}")
    print("\nClass distribution:")
    for class_name, cases in class_cases.items():
        print(f"  {class_name}: {len(cases)} cases")
    
    # Select diverse samples
    print(f"\n🎯 Selecting {num_samples} diverse samples...")
    selected_cases = []
    
    # Try to get samples from each class
    for class_name, cases in class_cases.items():
        if cases and len(selected_cases) < num_samples:
            # Select one case from this class
            case = random.choice(cases)
            selected_cases.append(case)
            print(f"Selected: {case['case_data']['case_name']} - {class_name}")
    
    # Fill remaining slots randomly if needed
    while len(selected_cases) < num_samples and len(analyzed_cases) > len(selected_cases):
        remaining_cases = [c for c in analyzed_cases if c not in selected_cases]
        if remaining_cases:
            case = random.choice(remaining_cases)
            selected_cases.append(case)
            print(f"Selected: {case['case_data']['case_name']} - {case['class_name']}")
    
    # Visualize selected cases
    print(f"\n🖼️  Visualizing {len(selected_cases)} cases...")
    
    for i, case_info in enumerate(selected_cases):
        print(f"\n--- Case {i+1}: {case_info['case_data']['case_name']} ---")
        print(f"Class: {case_info['class_name']}")
        print(f"Volume shape: {case_info['case_data']['shape']}")
        
        # Print detailed segmentation analysis
        print("Segmentation breakdown:")
        for label, info in case_info['seg_analysis'].items():
            if label > 0:  # Skip background
                print(f"  Label {label} ({info['meaning']}): {info['count']:,} voxels ({info['percentage']:.1f}%)")
        
        # Visualize
        fig = visualize_brats_case(case_info['case_data'])
        
        print("-" * 60)
    
    return selected_cases, class_cases

def quick_dataset_check(dataset_path):
    """Quick check of dataset structure"""
    print("🔍 Quick Dataset Structure Check")
    print("-" * 40)
    
    # Look for case directories
    possible_patterns = [
        os.path.join(dataset_path, "BraTS20_Training_*"),
        os.path.join(dataset_path, "*", "BraTS20_Training_*"),
        os.path.join(dataset_path, "MICCAI_BraTS2020_TrainingData", "BraTS20_Training_*")
    ]
    
    for pattern in possible_patterns:
        dirs = glob.glob(pattern)
        if dirs:
            print(f"✅ Found {len(dirs)} cases with pattern: {pattern}")
            
            # Check first case
            first_case = dirs[0]
            case_name = os.path.basename(first_case)
            print(f"Sample case: {case_name}")
            
            # Check files in first case
            expected_files = [f"{case_name}_t1.nii", f"{case_name}_t1ce.nii", 
                            f"{case_name}_t2.nii", f"{case_name}_flair.nii", 
                            f"{case_name}_seg.nii"]
            
            for file_name in expected_files:
                file_path = os.path.join(first_case, file_name)
                if os.path.exists(file_path):
                    print(f"  ✅ {file_name}")
                else:
                    print(f"  ❌ {file_name}")
            
            return dirs
    
    print("❌ No BraTS cases found!")
    print("Expected structure:")
    print("  dataset_path/BraTS20_Training_XXX/BraTS20_Training_XXX_*.nii")
    return []

# Usage example
if __name__ == "__main__":
    # Update this path to your dataset location
    dataset_path = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    
    # Quick check first
    print("Checking dataset structure...")
    case_dirs = quick_dataset_check(dataset_path)
    
    if case_dirs:
        print(f"\n✅ Dataset looks good! Found {len(case_dirs)} cases")
        
        # Full exploration
        selected_cases, all_classes = explore_brats_dataset(dataset_path, num_samples=5)
        
        print(f"\n🎉 Exploration complete!")
        print(f"Visualized {len(selected_cases)} diverse cases")
        
    else:
        print("\n❌ Please check your dataset path and structure")
        print("Expected: /path/to/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/")