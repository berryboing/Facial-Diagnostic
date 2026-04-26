import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# 1. NEW IMPORT: Vision Transformer instead of ResNet
from torchvision.models import vit_b_16 

def load_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on: {device}")

    # 2. Initialize the Vision Transformer (ViT-Base with 16x16 patches)
    model = vit_b_16() 

    # 3. Reconstruct the Kaggle author's custom final classification head
    # The error log showed BatchNorm1d at index 0 and Linear at index 2.
    # The hidden dimension for a standard ViT-Base is 768.
    model.heads.head = nn.Sequential(
        nn.BatchNorm1d(768),
        nn.Dropout(0.5),       # Index 1 (weights aren't saved for Dropout, which is why 1 was "missing")
        nn.Linear(768, 7)      # Index 2 (Outputs to your 7 emotion classes)
    )

    # 4. Load the weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()      
    model.to(device)  
    
    return model, device

def main():
    model_path = 'student_emotion_resnet34_best.pth' 
    model, device = load_model(model_path)

    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] 

    # Transformers also use the standard ImageNet 224x224 pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    
    print("========================================")
    print(" THESIS CAMERA ACTIVE - PHOTO BOOTH MODE")
    print(" MODEL: Vision Transformer (ViT)")
    print(" - Press SPACEBAR to capture and analyze")
    print(" - Press SPACEBAR again to resume live feed")
    print(" - Press 'q' to quit")
    print("========================================")

    mode = "LIVE"
    captured_frame = None

    while True:
        if mode == "LIVE":
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                cv2.putText(frame, "Press SPACE to Scan", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Thesis PC Test - Vision Transformer', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                break
            elif key == 32: # SPACEBAR pressed
                mode = "CAPTURED"
                captured_frame = frame.copy() 
                
                gray_cap = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
                faces_cap = face_cascade.detectMultiScale(gray_cap, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces_cap:
                    cv2.rectangle(captured_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    face_crop = captured_frame[y:y+h, x:x+w]
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)

                    input_tensor = transform(pil_img).unsqueeze(0).to(device)

                    with torch.no_grad(): 
                        output = model(input_tensor)
                        probabilities = F.softmax(output, dim=1)[0] 

                    scores = probabilities.cpu().numpy() * 100

                    text_x = x + w + 10
                    text_y = y + 20
                    
                    cv2.rectangle(captured_frame, (text_x - 5, text_y - 20), (text_x + 180, text_y + (len(classes)*25)), (0, 0, 0), -1)

                    for i, (emotion, score) in enumerate(zip(classes, scores)):
                        text = f"{emotion}: {score:.1f}%"
                        color = (0, 255, 0) if score == max(scores) else (255, 255, 255)
                        cv2.putText(captured_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                        text_y += 25 

        elif mode == "CAPTURED":
            cv2.imshow('Thesis PC Test - Vision Transformer', captured_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                break
            elif key == 32: 
                mode = "LIVE" 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()