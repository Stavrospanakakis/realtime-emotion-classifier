# Real Time Emotion Classifier

By using this app you can: 
- Train the model
- See the progress of loss and accuracy
- Recognise the emotions of a person using your camera

Model needs improvement but I have so much fun creating this, so I uploaded it.

## Demo

![Demo](demo.gif)

## Installation

You should have a Kaggle Acount and download your API info. Place the kaggle.json to /home/USER/.kaggle. Then type:
```
$ cd build
$ chmod +x ./setup.sh
$ ./setup.sh
```

## Run

To train the model type
```
$ python train.py
```

To predict emotions through the camera type
```
$ python predict.py
```