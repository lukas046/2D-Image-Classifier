import os
import time
import torch
import pydicom
import numpy as np
import pandas as pd
from PIL import Image

def norm(x):
    min_ = x.min()
    return (x - min_)/(x.max()-min_)

def main():
    # Transforms the raw full dataset into reduced size npy dataset and pytorch dataset
    
    dataset_dir = '../dataset'
    
    # ../dataset/train/...
    input_train_dir = os.path.join(dataset_dir, 'train')
    
    # ../dataset/preprocessed
    preprocessed_dir = os.path.join(dataset_dir, 'preprocessed')

    # ../dataset/preprocessed/train
    output_dir = os.path.join(preprocessed_dir, 'train')
    
    assert os.path.exists(dataset_dir) and os.path.exists(input_train_dir), "Raw dataset must be downloaded first. Run the download_original_dataset file in get_data.py first."

    # Set up environment to write to
    if not os.path.exists(preprocessed_dir):
        os.mkdir(preprocessed_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    count = 1
    time_estimate = 0
    mean_time = 0

    # Open the raw downloaded files and preprocess them to smaller files for training
    for root_dir, _, files in os.walk(input_train_dir):
        for file in files:
            if file.endswith('.dcm'):
                print(f"Working on train preprocessing {count:0>4}/{1738}. Time remaining={int(time_estimate//60)}:{int(time_estimate%60//1):0>2}", end='\r')
                output_path = os.path.join(output_dir, file.removesuffix(".dcm"))
                start = time.time()
                
                if not os.path.exists(output_path):
                    img = pydicom.dcmread(os.path.join(root_dir, file)).pixel_array
                    float_img = norm(img)
                    smaller = np.array(Image.fromarray(float_img).resize((224,224)))
                    normed = (smaller - smaller.mean()) / smaller.std()
                    low = np.percentile(normed, 1)
                    high = np.percentile(normed, 99)
                    clipped = np.clip(normed, low, high)
                    output = (clipped - low) / (high - low)
                    np.save(output_path, output)
                
                end = time.time()
                
                mean_time = (end - start) + mean_time * count
                count += 1
                mean_time = mean_time / count
                time_estimate = (1738 - count) * mean_time
    print()

    # Create a torch dataset file
    df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    data_df = pd.DataFrame({"SOPInstanceUID":pd.Series(dtype='str'), "data":pd.Series(dtype='object')})

    for fp in os.listdir(output_dir):
        row_df = pd.DataFrame({"SOPInstanceUID":fp.removesuffix("-c.npy"), "data":[np.load(f"{output_dir}/{fp}")]})
        data_df = pd.concat([data_df,row_df])
    
    joined_df = pd.merge(df, data_df)
    
    # Select only the first label since some of them are multi label
    joined_df['label'] = joined_df['Target'].apply(lambda x: x.strip().split(' ')[0])

    # Format and save the data to a pytorch dataset file
    data_tensor = torch.tensor(joined_df['data'].array)
    label_tensor = torch.tensor((joined_df['label'].apply(lambda x: int(x))).array)

    dataset_data = {
        'features':data_tensor,
        'labels':label_tensor
    }

    torch.save(dataset_data, os.path.join(dataset_dir, 'processed_dataset.pt'))

if __name__ == '__main__':
    main()