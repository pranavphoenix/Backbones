import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

def split_data(input_dir, output_dir, test_size=0.2, val_size=0.0, random_state=42):
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    val_dir = os.path.join(output_dir, 'val') if val_size > 0 else None
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    if val_dir:
        os.makedirs(val_dir, exist_ok=True)
    
    # Loop through each class folder
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if os.path.isdir(class_dir):
            # Create class subdirectories in train, test, and validation directories
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name) if val_dir else None
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)
            if val_class_dir:
                os.makedirs(val_class_dir, exist_ok=True)
            
            # List all files in the class directory
            file_list = os.listdir(class_dir)
            
            # Split the data
            train_files, temp_files = train_test_split(file_list, test_size=(test_size + val_size), random_state=random_state)
            if val_size > 0:
                val_files, test_files = train_test_split(temp_files, test_size=test_size/(test_size + val_size), random_state=random_state)
            else:
                test_files = temp_files
                val_files = []
            
            # Copy files to the respective directories
            for file_name in train_files:
                src_file = os.path.join(class_dir, file_name)
                dst_file = os.path.join(train_class_dir, file_name)
                shutil.copy2(src_file, dst_file)
                
            for file_name in test_files:
                src_file = os.path.join(class_dir, file_name)
                dst_file = os.path.join(test_class_dir, file_name)
                shutil.copy2(src_file, dst_file)
            
            if val_files:
                for file_name in val_files:
                    src_file = os.path.join(class_dir, file_name)
                    dst_file = os.path.join(val_class_dir, file_name)
                    shutil.copy2(src_file, dst_file)
                
    print("Data split complete.")

def main():
    parser = argparse.ArgumentParser(description="Split data into train, test, and validation sets.")
    parser.add_argument('--input_dir', type=str, help="Path to the input directory with class folders.")
    parser.add_argument('--output_dir', type=str, help="Path to the output directory where splits will be saved.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of data to be used for testing.")
    parser.add_argument('--val_size', type=float, default=0.0, help="Proportion of data to be used for validation.")
    parser.add_argument('--random_state', type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    split_data(args.input_dir, args.output_dir, test_size=args.test_size, val_size=args.val_size, random_state=args.random_state)

if __name__ == "__main__":
    main()
