import cv2
import numpy as np
from keras.models import load_model

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = {0: "Angry", 1: "Fear", 2: "Happy", 3: "Sad", 4: "Surprise", 5: "Neutral" }

# load and compile the model
loaded_model = load_model("./models/model.h5")
loaded_model.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy'],
)

# camera number
camera = 0

# Set the video
cap = cv2.VideoCapture(camera)

while True:

    # Start the video
    ret, image = cap.read()

    # Convert the image to grayscale
    grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect face from the image
    faces = face_cascade.detectMultiScale(grayscaled_image, 1.3, 5)

    for (x,y,width,height) in faces:
        
        # Draw a rectangle to the face  
        cv2.rectangle(image,(x,y),(x+width,y+height),(255,0,0),2)

        # Crop the face
        croped_image = grayscaled_image[y:y+height, x:x+width]
                
        # resize the image with the face to 48x48
        resized_image = cv2.resize(croped_image, (48, 48))  

        # Reshape the image
        resized_image = np.array(resized_image).reshape(-1, 48, 48, 1)

        # Predict the emotion
        result = loaded_model.predict(resized_image) * 100
        index_max = np.argmax(result[0])

        # Add the emotion as text to the image
        cv2.putText(image, str(emotions[index_max]), (x+width,y+height+40), cv2.FONT_HERSHEY_SIMPLEX,  
                      1, (255,0,0), 2, cv2.LINE_AA) 
    
    # Show the image
    cv2.imshow('Emotion Classifier',image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close
cap.release()
cv2.destroyAllWindows()