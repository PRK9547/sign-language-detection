Sign Language Detection using Deep Learning:

This project is a real-time Sign Language Recognition System built with Deep Learning and an interactive Streamlit UI.
It allows users to either capture a hand sign through their camera or upload an image, and then predicts the corresponding English alphabet (A–Z).

Features:

1.Deep Learning Model trained to recognize hand signs from A–Z.
   
2.Camera Input and Image Upload support for predictions.

3.Voice Feedback using pyttsx3 (text-to-speech).

4.Streamlit Web App for a simple and intuitive user interface.

5.Real-time prediction and easy deployment.

🧩Technologies Used:

1.Python

2.TensorFlow / Keras

3.OpenCV

4.NumPy

5.Streamlit

6.pyttsx3

How It Works:

1.Load the pre-trained model (sign_language_model.h5)

2.Capture or upload an image of a hand sign

3.Preprocess the image (resize, normalize)

4.Predict the corresponding alphabet using the model

5.Display the result and optionally speak it aloud
