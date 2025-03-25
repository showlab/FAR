import subprocess
from glob import glob
from os import path as osp


def update_sha(paths):
    print('# Update SHA for model files...')
    for idx, path in enumerate(paths):
        print(f'{idx+1:03d}: Processing {path}')

        # Calculate SHA-256 hash (first 8 characters)
        sha = subprocess.check_output(['sha256sum', path]).decode()[:8]

        basename = osp.basename(path)
        if '-' in basename:
            # If file already has a SHA in name, extract the part before it
            base_name = path.split('-')[0]
            final_file = f'{base_name}-{sha}.pth'
        else:
            # If no SHA in filename, add it before the extension
            final_file = path.split('.pth')[0] + f'-{sha}.pth'

        print(f'\tRenaming: {path} → {final_file}')
        subprocess.Popen(['mv', path, final_file])


if __name__ == '__main__':

    # Path to the folder containing model files
    model_folder = 'experiments/pretrained_models/FAR_Models/long_video_prediction/*.pth'
    paths = glob(model_folder)

    # Add SHA to model filenames
    update_sha(paths)
