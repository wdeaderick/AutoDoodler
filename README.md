# Using a CycleGAN to translate between rough-hand sketches and flourishy doodle artwork.
1. CycleGAN implementation:
Implement off the shelf CycleGAN - pytorch or tensorflow, whichever is easy to adapt.
If learned weights are available, start off with visualizing the generated data from the Quickdraw dataset. Otherwise, we’ll train the CycleGAN
2. QuickDraw dataset download:
Google quickdraw dataset download: https://quickdraw.withgoogle.com/data
Filter out the classes which won’t provide pattern transfer (intuitive)
Adjust the resolution- decide uniform across dataset
3. Pattern Dataset download:
Google images search - patterned doodle dataset.
Figure out scripting to download a bunch of such images
4. Preprocessing:
Patch sampling the images having high resolution to generated 5-10 images out of a single ones, deciding the base resolution [Preferably the same as the quickdraw resolution decided]
Once the dataset is downloaded, figure out the organization so that it’s easy to schedule.

DATA WHEREABOUTS:
The data "QuickDraw" is downloaded in .npy format and stored in the the shared Google Drive. 
Folder CS236>data_npy
The script for processing the .npy files is in data_processing.py


# Processing Steps
python data_to_img.py

for generating trainA, testA dataset

python generate_patches.py

for generating trainB, testB dataset
