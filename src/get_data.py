import os
import gdown
import shutil

# def download_original_dataset():
#     import kagglehub
#     # Note: this downloads the full resolution dataset from the kaggle competition.
#     # This is unnecessary but included in case you want to recreate the processed_dataset.pt file.
#     # The original dataset is ~30GB. Thus we have a reduced size one provided in the google drive.

#     # This will prompt you to input your Kaggle API Token
#     # API tokens can be generated at https://www.kaggle.com/settings under API
#     # Please accept the conditions and join the competition, then run this script.
#     kagglehub.login()

#     data_directory = "../dataset"

#     global_data_dir = os.path.join(os.curdir, data_directory)

#     if not os.path.exists(global_data_dir):
#         os.mkdir(global_data_dir)
    
#     # if os.path.exists(global_data_dir)
#     # Download the directory
#     try:
#         download_dir = kagglehub.competition_download('unifesp-x-ray-body-part-classifier', output_dir=global_data_dir)
#     except FileExistsError:
#         print("Dataset files already downloaded.")
#     except Exception as e:
#         print(f"Unexpected error occurred: {e}")

def main():
    temp_dir = '../temp'
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    gdown.download_folder("https://drive.google.com/drive/folders/1ptwOH3FbOxwhTLPVAKhpawDIsNJs5ZqO", output=temp_dir)

    dataset_dir = '../dataset'
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    gdown.download_folder("https://drive.google.com/drive/folders/1io2yKOQI6JfDI5ORq1CGx1tBddf5tOyO?usp=sharing", output=dataset_dir)

    shutil.unpack_archive(os.path.join(temp_dir, 'best_model.zip'), extract_dir='2d_image_classifier')
    shutil.unpack_archive(os.path.join(temp_dir, 'data.zip'), extract_dir='../')


if __name__ == '__main__':
    main()