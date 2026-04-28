import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageGrab  
import numpy as np
from torchvision.models import vit_b_16
import tkinter as tk
from tkinter import messagebox
from tkinter import font as tkfont
import os
import time

import firebase_admin
from firebase_admin import credentials, firestore

# ==========================================
# 1. FIREBASE INITIALIZATION
# ==========================================
try:
    cred = credentials.Certificate('firebase_key.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase Database connected successfully!")
except Exception as e:
    print(f"❌ Firebase setup failed. Check your JSON key: {e}")


# ==========================================
# 2. AI MODEL SETUP
# ==========================================
def load_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on: {device}")
    
    model = vit_b_16() 
    model.heads.head = nn.Sequential(
        nn.BatchNorm1d(768),
        nn.Dropout(0.5),       
        nn.Linear(768, 7)      
    )
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("✅ Model weights loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        
    model.eval()      
    model.to(device)  
    return model, device


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_diagnosis_sentence(dimension, score):
    if score < 40.0:
        return f"Low levels: Based on your facial scanning, there were only mild signs of {dimension} and it's up to you if you want to schedule for a consultation."
    elif score < 65.0:
        return f"Slight level increase: Based on your facial scanning, there were slight elevations of {dimension} and we encourage you to look after yourself and/or schedule for a consultation."
    else:
        return f"Increased levels: Based on your facial scanning, there were high elevations of {dimension} and we recommend that you ask a professional for guidance."


# ==========================================
# 4. DATABASE & DUAL-SNAPSHOT SAVING LOGIC 
# ==========================================
def save_to_database(email, dep, anx, str_score, frame, window, crop_y):
    if not email or "@" not in email:
        messagebox.showwarning("Invalid Input", "Please enter a valid email address.")
        return

    save_dir = "snapshots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = int(time.time())

    # --- SAVE RAW FACE FRAME ---
    face_filename = f"face_{timestamp}.jpg"
    face_path = os.path.join(save_dir, face_filename)
    cv2.imwrite(face_path, frame)

    # --- SAVE PRECISE DIAGNOSIS RESULT SNAPSHOT ---
    result_filename = f"result_{timestamp}.jpg"
    result_path = os.path.join(save_dir, result_filename)
    
    try:
        window.update()
        x = window.winfo_rootx()
        y = window.winfo_rooty()
        width = window.winfo_width() 
        y_bottom_abs = y + crop_y
        screenshot = ImageGrab.grab(bbox=(x, y, x + width, y_bottom_abs))
        screenshot.save(result_path)
    except Exception as e:
        messagebox.showwarning("Screenshot Error", f"Could not precisely capture the UI window: {e}\nOnly the face snapshot was saved.")
        result_filename = None 

    # --- SAVE DATA TO FIREBASE FIRESTORE ---
    try:
        doc_ref = db.collection('assessment_results').document()
        doc_ref.set({
            'email': email,
            'depression': float(dep),
            'anxiety': float(anx),
            'stress': float(str_score),
            'face_filename': face_filename,
            'result_filename': result_filename if result_filename else "capture_failed", 
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        
        success_msg = f"Data successfully saved to Firebase!\nFace snapshot saved locally as '{face_filename}'.\n"
        if result_filename:
            success_msg += f"Precise result snapshot saved locally as '{result_filename}'."
            
        messagebox.showinfo("Success", success_msg)
        window.destroy()

    except Exception as e:
        messagebox.showerror("Firebase Error", f"Failed to upload data to Firestore: {e}")


# ==========================================
# 5. USER INTERFACE (TKINTER) 
# ==========================================
def show_results_ui(dep_score, anx_score, str_score, dep_text, anx_text, str_text, captured_frame):
    root = tk.Tk()
    root.title("DASS-21 Facial Scanning Report")
    root.geometry("650x600")
    root.configure(bg="#FFFFFF") 

    title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
    header_font = tkfont.Font(family="Helvetica", size=11, weight="bold")
    body_font = tkfont.Font(family="Helvetica", size=10)
    
    text_dark = "#2C3E50"  
    text_blue = "#2980B9"  
    bg_gray = "#F8F9F9"    

    tk.Label(root, text="Initial Assessment Results", font=title_font, bg="#FFFFFF", fg=text_dark).pack(pady=(20, 10))

    def create_result_block(parent, title, score, sentence):
        frame = tk.Frame(parent, bg=bg_gray, padx=15, pady=10, relief=tk.FLAT, borderwidth=1)
        frame.pack(fill=tk.X, padx=30, pady=5)
        tk.Label(frame, text=f"{title}: {score:.1f}%", font=header_font, bg=bg_gray, fg=text_blue).pack(anchor="w")
        tk.Label(frame, text=sentence, font=body_font, bg=bg_gray, fg=text_dark, wraplength=550, justify=tk.LEFT).pack(anchor="w", pady=(5,0))

    create_result_block(root, "DEPRESSION", dep_score, dep_text)
    create_result_block(root, "ANXIETY", anx_score, anx_text)
    create_result_block(root, "STRESS", str_score, str_text)

    precaution_statement = "PRECAUTION: This is just an initial assessment using facial scanning technology and is not a clinical diagnosis."
    precaution_label = tk.Label(root, text=precaution_statement, font=tkfont.Font(family="Helvetica", size=9, slant="italic"), bg="#FFFFFF", fg="#E67E22")
    precaution_label.pack(pady=(15, 10))

    save_frame = tk.Frame(root, bg="#FFFFFF")
    save_frame.pack(pady=20)

    tk.Label(save_frame, text="Enter email to save results and snapshots to database:", font=body_font, bg="#FFFFFF", fg=text_dark).pack()
    
    email_entry = tk.Entry(save_frame, width=40, font=body_font, bg="#ECF0F1", relief=tk.FLAT)
    email_entry.pack(pady=10, ipady=5) 

    root.update()
    
    crop_y_bottom = precaution_label.winfo_y() + precaution_label.winfo_height()

    submit_btn = tk.Button(save_frame, text="Save to Database", font=header_font, bg="#3498DB", fg="white", 
                           activebackground="#2980B9", activeforeground="white", relief=tk.FLAT, padx=20, pady=5,
                           command=lambda: save_to_database(email_entry.get(), dep_score, anx_score, str_score, captured_frame, root, crop_y_bottom))
    submit_btn.pack()

    root.mainloop()


# ==========================================
# 6. MAIN CAMERA LOOP
# ==========================================
def main():
    model_path = 'student_emotion_resnet34_best.pth' 
    model, device = load_model(model_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    
    print("\n========================================")
    print(" THESIS CAMERA ACTIVE")
    print(" - Press SPACEBAR to capture and analyze")
    print(" - Press 'q' to quit")
    print("========================================\n")

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Failed to grab frame. Exiting...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(frame, "Press SPACE to Scan", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Thesis Camera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == 32: 
            gray_cap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_cap = face_cascade.detectMultiScale(gray_cap, scaleFactor=1.3, minNeighbors=5)

            if len(faces_cap) > 0:
                clean_frame = frame.copy() 

                x, y, w, h = faces_cap[0] 
                
                face_crop = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(face_rgb)

                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad(): 
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1)[0] 

                scores = probabilities.cpu().numpy() * 100

                str_multiplier = 1.4  
                anx_multiplier = 1.3  
                dep_multiplier = 0.6  

                raw_dep = scores[4] 
                raw_anx = scores[2] 
                raw_str = scores[0] + scores[1] 

                dep_score = min(raw_dep * dep_multiplier, 100.0)
                anx_score = min(raw_anx * anx_multiplier, 100.0)
                str_score = min(raw_str * str_multiplier, 100.0)

                dep_text = get_diagnosis_sentence("Depression", dep_score)
                anx_text = get_diagnosis_sentence("Anxiety", anx_score)
                str_text = get_diagnosis_sentence("Stress", str_score)

                show_results_ui(dep_score, anx_score, str_score, dep_text, anx_text, str_text, clean_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()