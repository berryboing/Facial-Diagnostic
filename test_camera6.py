import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models import vit_b_16

def load_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vit_b_16() 
    model.heads.head = nn.Sequential(
        nn.BatchNorm1d(768),
        nn.Dropout(0.5),       
        nn.Linear(768, 7)      
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()      
    model.to(device)  
    return model, device

# --- HELPER FUNCTION: Wraps long sentences cleanly in OpenCV ---
def draw_wrapped_text(img, text, position, max_width, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, color=(255, 255, 255), thickness=1):
    words = text.split(' ')
    x, y = position
    line = ''
    for word in words:
        test_line = line + word + ' '
        size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        if size[0] > max_width and line != '':
            cv2.putText(img, line, (x, y), font, font_scale, color, thickness)
            line = word + ' '
            y += size[1] + 10 # Move down a line
        else:
            line = test_line
    cv2.putText(img, line, (x, y), font, font_scale, color, thickness)
    return y + 30 # Return the next Y coordinate for the next paragraph

# --- HELPER FUNCTION: Generates the sentence based on your 3 levels ---
def get_diagnosis_sentence(dimension, score):
    if score < 25.0:
        return f"Low levels: Based on your facial scanning, there were only mild signs of {dimension} and it's up to you if you want to schedule for a consultation."
    elif score < 50.0:
        return f"Slight level increase: Based on your facial scanning, there were slight elevation of {dimension} and we encourage you to look after yourself and/or schedule for a consultation."
    else:
        return f"Increased levels: Based on your facial scanning, there were elevations of {dimension} and we recommend that you ask a professional for guidance."

def main():
    model_path = 'student_emotion_resnet34_best.pth' 
    model, device = load_model(model_path)

    # Note: Keep the exact order of your trained classes
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] 

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    
    print("========================================")
    print(" THESIS CAMERA ACTIVE - DASS-21 MODE")
    print(" - Press SPACEBAR to capture and analyze")
    print(" - Press SPACEBAR again to resume live feed")
    print(" - Press 'q' to quit")
    print("========================================")

    mode = "LIVE"
    captured_frame = None
    results_window_name = "DASS-21 Assessment Results"

    while True:
        if mode == "LIVE":
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                cv2.putText(frame, "Press SPACE to Scan", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Thesis Camera', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                break
            elif key == 32: # SPACEBAR pressed
                mode = "CAPTURED"
                captured_frame = frame.copy() 
                
                gray_cap = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
                faces_cap = face_cascade.detectMultiScale(gray_cap, scaleFactor=1.3, minNeighbors=5)

                if len(faces_cap) > 0:
                    x, y, w, h = faces_cap[0] # Just grab the first face detected
                    cv2.rectangle(captured_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    face_crop = captured_frame[y:y+h, x:x+w]
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)

                    input_tensor = transform(pil_img).unsqueeze(0).to(device)

                    with torch.no_grad(): 
                        output = model(input_tensor)
                        probabilities = F.softmax(output, dim=1)[0] 

                    scores = probabilities.cpu().numpy() * 100

                    # --- MAP TO DASS-21 ---
                    # index 4 = Sad, index 2 = Fear, index 0 = Angry, index 1 = Disgust
                    dep_score = scores[4] 
                    anx_score = scores[2] 
                    # Stress is a combination of frustration (Angry) and aversion (Disgust)
                    str_score = min(scores[0] + scores[1], 100.0) 

                    # Generate the sentences
                    dep_text = get_diagnosis_sentence("Depression", dep_score)
                    anx_text = get_diagnosis_sentence("Anxiety", anx_score)
                    str_text = get_diagnosis_sentence("Stress", str_score)
                    precaution = "PRECAUTION: This is just an initial assessment using facial scanning technology and is not a clinical diagnosis."

                    # --- CREATE THE NEW RESULTS WINDOW ---
                    # Create a blank dark gray canvas (Height 600, Width 800)
                    results_img = np.zeros((500, 800, 3), dtype=np.uint8)
                    results_img[:] = (30, 30, 30) # Dark gray background

                    # Title
                    cv2.putText(results_img, "DASS-21 Facial Scanning Report", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    
                    # Print Depression Results
                    y_offset = 90
                    cv2.putText(results_img, f"DEPRESSION SCORE: {dep_score:.1f}%", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
                    y_offset = draw_wrapped_text(results_img, dep_text, (20, y_offset + 30), 760)

                    # Print Anxiety Results
                    cv2.putText(results_img, f"ANXIETY SCORE: {anx_score:.1f}%", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
                    y_offset = draw_wrapped_text(results_img, anx_text, (20, y_offset + 30), 760)

                    # Print Stress Results
                    cv2.putText(results_img, f"STRESS SCORE: {str_score:.1f}%", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
                    y_offset = draw_wrapped_text(results_img, str_text, (20, y_offset + 30), 760)

                    # Print Disclaimer/Precaution in Yellow at the bottom
                    draw_wrapped_text(results_img, precaution, (20, 450), 760, color=(0, 255, 255), thickness=2)

                    # Show the new window
                    cv2.imshow(results_window_name, results_img)

        elif mode == "CAPTURED":
            cv2.imshow('Thesis Camera', captured_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                break
            elif key == 32: 
                mode = "LIVE" 
                # Close the results window when returning to live view
                try:
                    cv2.destroyWindow(results_window_name)
                except:
                    pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()