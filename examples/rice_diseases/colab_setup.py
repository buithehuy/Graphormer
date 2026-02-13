"""
Colab Setup Utilities for Rice Diseases Dataset

Handles data copying from Google Drive and extraction to local Colab storage
to avoid RAM overflow issues.
"""

import os
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm


def copy_and_extract_dataset(
    drive_zip_path="MyDrive/Rice_Diseases_Dataset/rice-diseases-image-dataset.zip",
    temp_dir="/tmp",
    extract_dir="/content/rice_diseases_data"
):
    """
    Copy dataset from Google Drive and extract to local Colab storage.
    
    This avoids RAM overflow issues that can occur when directly extracting
    from Drive-mounted files.
    
    Args:
        drive_zip_path: Path to zip file in Drive (relative to /content/drive/)
        temp_dir: Temporary directory to copy zip before extraction
        extract_dir: Final extraction directory
    
    Returns:
        extract_dir: Path to extracted data
    """
    print("=" * 60)
    print("Setting up Rice Diseases Dataset")
    print("=" * 60)
    
    # Full path to Drive zip
    full_drive_path = f"/content/drive/{drive_zip_path}"
    
    # Check if Drive is mounted
    if not os.path.exists("/content/drive"):
        raise RuntimeError(
            "Google Drive is not mounted. Please run:\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')"
        )
    
    # Check if zip exists
    if not os.path.exists(full_drive_path):
        raise FileNotFoundError(
            f"Dataset zip not found at: {full_drive_path}\n"
            f"Please ensure the dataset is uploaded to Drive at:\n"
            f"{drive_zip_path}"
        )
    
    # Check if already extracted
    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
        print(f"\n✓ Dataset already extracted at: {extract_dir}")
        return extract_dir
    
    # Create temp and extract directories
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    
    # Copy zip to temp directory
    temp_zip_path = os.path.join(temp_dir, "rice_diseases.zip")
    
    print(f"\n[1/2] Copying dataset from Drive to local storage...")
    print(f"      Source: {full_drive_path}")
    print(f"      Temp: {temp_zip_path}")
    
    # Get file size for progress bar
    file_size = os.path.getsize(full_drive_path)
    
    # Copy with progress bar
    with open(full_drive_path, 'rb') as src:
        with open(temp_zip_path, 'wb') as dst:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Copying") as pbar:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"      ✓ Copy complete")
    
    # Extract zip
    print(f"\n[2/2] Extracting dataset...")
    print(f"      Destination: {extract_dir}")
    
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        # Get list of files for progress bar
        file_list = zip_ref.namelist()
        
        with tqdm(total=len(file_list), desc="Extracting") as pbar:
            for file in file_list:
                zip_ref.extract(file, extract_dir)
                pbar.update(1)
    
    print(f"      ✓ Extraction complete")
    
    # Clean up temp zip
    print(f"\n[Cleanup] Removing temporary zip file...")
    os.remove(temp_zip_path)
    print(f"      ✓ Cleanup complete")
    
    print("\n" + "=" * 60)
    print("Dataset setup complete!")
    print("=" * 60)
    print(f"Data location: {extract_dir}")
    
    return extract_dir


def verify_installation():
    """
    Verify that all required packages are installed correctly.
    
    Returns:
        bool: True if all packages are available
    """
    print("Verifying installation...")
    print("-" * 40)
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('skimage', 'scikit-image'),
        ('torch', 'PyTorch'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('networkx', 'NetworkX'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    
    all_installed = True
    
    for module_name, display_name in required_packages:
        try:
            __import__(module_name)
            print(f"✓ {display_name:20s} - OK")
        except ImportError:
            print(f"✗ {display_name:20s} - MISSING")
            all_installed = False
    
    print("-" * 40)
    
    if all_installed:
        print("✓ All packages are installed correctly!")
    else:
        print("✗ Some packages are missing. Please install them.")
    
    return all_installed


def check_preprocessed_cache(cache_dir="/content/rice_diseases_graphs"):
    """
    Check if preprocessed graphs already exist.
    
    Args:
        cache_dir: Directory where preprocessed graphs are stored
    
    Returns:
        bool: True if cache exists and is not empty
    """
    if not os.path.exists(cache_dir):
        return False
    
    # Check if cache has any .pt files
    pt_files = list(Path(cache_dir).rglob("*.pt"))
    
    if len(pt_files) == 0:
        return False
    
    print(f"✓ Found {len(pt_files)} preprocessed graph files in cache")
    return True


def get_dataset_structure(data_dir="/content/rice_diseases_data"):
    """
    Analyze the dataset directory structure.
    
    Args:
        data_dir: Root directory of extracted dataset
    
    Returns:
        dict: Information about dataset structure
    """
    print("\nAnalyzing dataset structure...")
    print("-" * 60)
    
    structure = {
        'root': data_dir,
        'subdirs': [],
        'classes': set(),
        'image_count': 0
    }
    
    # Search for image directories
    for root, dirs, files in os.walk(data_dir):
        # Count image files
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            rel_path = os.path.relpath(root, data_dir)
            class_name = os.path.basename(root)
            
            structure['subdirs'].append({
                'path': root,
                'rel_path': rel_path,
                'class': class_name,
                'count': len(image_files)
            })
            
            structure['classes'].add(class_name)
            structure['image_count'] += len(image_files)
            
            print(f"  {rel_path:40s} - {len(image_files):4d} images - Class: {class_name}")
    
    print("-" * 60)
    print(f"Total classes: {len(structure['classes'])}")
    print(f"Total images: {structure['image_count']}")
    print(f"Classes: {sorted(structure['classes'])}")
    print("-" * 60)
    
    return structure


if __name__ == "__main__":
    # Example usage
    print("Rice Diseases Dataset - Colab Setup Utilities")
    print("\nThis module provides utilities for:")
    print("  1. Copying dataset from Google Drive")
    print("  2. Extracting to local Colab storage")
    print("  3. Verifying installation")
    print("  4. Checking preprocessed cache")
    print("\nUse these functions in your notebook or script.")
