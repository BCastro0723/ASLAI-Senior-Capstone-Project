from flask import Flask, render_template, Response, request, jsonify
from flask import jsonify
import cv2
from mediapipe.python.solutions.hands import Hands
import mediapipe as mp
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models
from autocorrect import Speller
import pyttsx3
# from googletrans import Translator
from deep_translator import GoogleTranslator
import threading

app = Flask(__name__)

# Initialize the VideoCapture object
cap = cv2.VideoCapture(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hands = mp.solutions.hands.Hands()

def load_model():
    # Load the pre-trained ResNet-18 model
    model = models.resnet50(pretrained=True)
    # Modify the last layer to match the number of classes in your dataset
    num_classes = len(class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes) 
    # Load the trained weights
    state_dict = torch.load('10resnet24.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    # Set the model to evaluation mode
    model.eval()
    return model

class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

# Load the model
model = load_model()

def draw_bounding_box(frame, landmarks, offset=30):
    # Get bounding box coordinates from hand landmarks
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for landmark in landmarks:
        x, y = landmark.x * frame.shape[1], landmark.y * frame.shape[0]
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
    
    # Extend the bounding box coordinates
    min_x -= offset
    min_y -= offset
    max_x += offset
    max_y += offset
    
    # Ensure the bounding box does not extend beyond the frame boundaries
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(frame.shape[1], max_x)
    max_y = min(frame.shape[0], max_y)
    
    # Draw bounding box
    cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)

    return frame, (min_x, min_y, max_x - min_x, max_y - min_y)

def preprocess_cropped_region(frame, bbox):
    x, y, w, h = bbox
    # Convert bounding box coordinates to integers
    x, y, w, h = int(x), int(y), int(w), int(h)
    def preprocess_transform(image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    # Crop the region within the bounding box
    cropped_region = frame[y:y+h, x:x+w]
    # Convert numpy array to PIL Image
    cropped_region_pil = Image.fromarray(cropped_region)

    # Apply preprocessing transforms
    preprocessed_region = preprocess_transform(cropped_region_pil)

    
    # Add a batch dimension
    preprocessed_region = preprocessed_region.unsqueeze(0)
    
    return preprocessed_region

word_stream = ""
correct_words = ""
# translator = Translator()
translator = GoogleTranslator()
word_lock = threading.Lock()
condition = threading.Condition()
spell = Speller(lang='en')

def producer(word):
    global word_stream
    with word_lock:
        word_stream += word + " "
        #print(f"Producer produced: {word_stream}")
    with condition:
        condition.notify()

def consumer():
    global word_stream
    global correct_words
    while True:
        with condition:
            # Wait for the condition to be notified
            condition.wait()
            words = word_stream.strip().split(" ")
            for word in words:
                # Spell check each word and print the corrected word
                corrected_word = spell(word)
                print(f"Consumer consumed: {word} (corrected to: {corrected_word})")
            # Clear the word stream
            word_stream = ""
            correct_words += corrected_word + " "




def gen_frames(model, producer_func, consecutive_count_threshold=5):
    cap = cv2.VideoCapture(0)
    consecutive_count = 0
    current_prediction = None
    word = ""

    
    # Create and start the consumer thread
    consumer_thread = threading.Thread(target=consumer)
    consumer_thread.start()

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0].landmark
            frame, bbox = draw_bounding_box(frame, hand_landmarks)

            if bbox is not None:
                preprocessed_region = preprocess_cropped_region(frame, bbox)

                with torch.no_grad():
                    model.eval()
                    output = model(preprocessed_region.to(device))
                
                _, predicted = torch.max(output, 1)
                predicted_class = predicted.item()
                class_name = class_names[predicted_class]

                # Update consecutive count and current prediction
                if class_name == current_prediction:
                    consecutive_count += 1
                else:
                    consecutive_count = 0
                    current_prediction = class_name

                # Check if consecutive count meets threshold
                if consecutive_count >= consecutive_count_threshold:
                    # Append current prediction to the word
                    word += current_prediction
                    # Update last prediction after processing prediction
                    last_prediction = current_prediction
                    # Reset consecutive count after processing prediction
                    consecutive_count = 0

                cv2.putText(frame, word, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, class_name, (frame.shape[1] - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        else:  # Hand is out of view
            if word:
                # Send the formed word to the producer function
                producer_func(word)
                # Reset the word
                word = ""


        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Concatenate frame one by one and show result
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/video_feed')
def video_feed_page():
    return render_template('video_feed.html')

@app.route('/video_stream')
def video_stream():
    return Response(gen_frames(model, producer), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_word')
def latest_word():
    # This should return the latest corrected word
    return correct_words

@app.route('/translate_word')
def translate_word():
    try:
        translate_words = translator.translate(correct_words, dest='es')
        # translated_text = translate_words.text
        # return jsonify({'translation': translated_text})
        translated_text = GoogleTranslator(source='auto', target='es').translate(correct_words)
        return jsonify({'translation': translated_text})

    except Exception as e:
        # Handle translation error
        error_message = str(e)
        return jsonify({'error': error_message}), 500

@app.route('/clear_word')
def clear_word():
    global correct_words
    correct_words = ""  # Clear the content of correct_words
    return correct_words

@app.route('/speak_text', methods=['POST'])
def speak_text():
    try:
        # Get the text to speak from the request data
        text = request.form.get('text')
        
        # Initialize pyttsx3 engine
        engine = pyttsx3.init()
        
        # Set properties (optional)
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 1)   # Volume (0.0 to 1.0)
        
        # Speak the text
        engine.say(text)
        engine.runAndWait()
        
        # Return a response
        return 'Text spoken successfully'
    except Exception as e:
        # Handle text-to-speech error
        error_message = str(e)
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)