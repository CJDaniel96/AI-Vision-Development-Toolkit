#!/usr/bin/env python3
import os
import sys
from PIL import Image
import argparse
from pathlib import Path

def rotate_images(input_folder, output_base_folder):
    """
    Rotate images in input folder (recursively) by 90, 180, 270 degrees and save to separate folders
    
    Args:
        input_folder: Path to folder containing input images (will search recursively)
        output_base_folder: Base path where rotated images will be saved
    """
    
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    
    # Create output folders
    rotation_folders = {
        90: os.path.join(output_base_folder, '90_degrees'),
        180: os.path.join(output_base_folder, '180_degrees'), 
        270: os.path.join(output_base_folder, '270_degrees')
    }
    
    # Create directories if they don't exist
    for folder in rotation_folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Process each image in input folder
    processed_count = 0
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        return
        
    # Recursively find all image files
    input_path = Path(input_folder)
    
    for file_path in input_path.rglob('*'):
        if not file_path.is_file():
            continue
            
        # Check if file has supported extension
        file_ext = file_path.suffix.lower()
        if file_ext not in supported_extensions:
            continue
            
        filename = file_path.name
            
        try:
            # Open and process image
            with Image.open(str(file_path)) as img:
                # Convert to RGB if necessary (for JPEG compatibility)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Rotate and save images
                for degrees, output_folder in rotation_folders.items():
                    rotated_img = img.rotate(-degrees, expand=True)  # Negative for clockwise rotation
                    output_path = os.path.join(output_folder, filename)
                    rotated_img.save(output_path)
                    
            processed_count += 1
            print(f"Processed: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Total images processed: {processed_count}")
    print(f"Output folders:")
    for degrees, folder in rotation_folders.items():
        print(f"  {degrees}°: {folder}")

def main():
    parser = argparse.ArgumentParser(description='Rotate images by 90, 180, 270 degrees')
    parser.add_argument('input_folder', help='Path to input folder containing images')
    parser.add_argument('output_folder', help='Base path for output folders')
    
    args = parser.parse_args()
    
    rotate_images(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()