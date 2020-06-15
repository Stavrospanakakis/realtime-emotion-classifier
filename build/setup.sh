# /bin/bash

echo "Downloading the requirements"
pip install -r requirements.txt

echo "Creating Dataset folder"
cd ..
mkdir dataset 
cd dataset

echo "Downloading the dataset"
kaggle datasets download -d debanga/facial-expression-recognition-challenge

echo "Unziping the dataset"
unzip \*.zip  && rm *.zip
cd ../

echo "Done."