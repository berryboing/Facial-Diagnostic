import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models import vit_b_16
import tkinter as tk
from tkinter import messagebox
from tkinter import font as tkfont

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

# --- HELPER FUNCTION: Thresholds (0-40, 40-65, 65+) ---
def get_diagnosis_sentence(dimension, score):
    if score < 40.0:
        return f"Low levels: Based on your facial scanning, there were only mild signs of {dimension} and it's up to you if you want to schedule for a consultation."
    elif score < 65.0:
        return f"Slight level increase: Based on your facial scanning, there were slight elevations of {dimension} and we encourage you to look after yourself and/or schedule for a consultation."
    else:
        return f"Increased levels: Based on your facial scanning, there were high elevations of {dimension} and we recommend that you ask a professional for guidance."


# ==========================================
# DATABASE SUBMISSION LOGIC
# ==========================================
def save_to_database(email, dep, anx, str_score, window):
    if not email or "@" not in email:
        messagebox.showwarning("Invalid Input", "Please enter a valid email address.")
        return

    # ---------------------------------------------------------
    # TODO: DATABASE INTEGRATION GOES HERE
    # Since you are using XAMPP, you will likely use MySQL.
    # 
    # 1. Open your terminal and run: pip install mysql-connector-python
    # 2. Uncomment and update the code below:
    # ---------------------------------------------------------
    """
    import mysql.connector

    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",          # Your XAMPP mysql username
            password="",          # Your XAMPP mysql password
            database="thesis_db"  # The name of your database
        )
        mycursor = mydb.cursor()
        
        sql = "INSERT INTO assessment_results (email, depression, anxiety, stress) VALUES (%s, %s, %s, %s)"
        val = (email, dep, anx, str_score)
        
        mycursor.execute(sql, val)
        mydb.commit()
        
        print(f"Record inserted for {email}")
        
    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Failed to connect: {err}")
        return
    """
    
    # Show success message and close the UI
    messagebox.showinfo("Success", f"Results securely saved for:\n{email}")
    window.destroy()


# ==========================================
# MODERN UI GENERATOR
# ==========================================
def show_results_ui(dep_score, anx_score, str_score, dep_text, anx_text, str_text):
    # Initialize the modern popup window
    root = tk.Tk()
    root.title("DASS-21 Facial Scanning Report")
    root.geometry("650x600")
    root.configure(bg="#FFFFFF") # Clean white background

    # Professional Fonts
    title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
    header_font = tkfont.Font(family="Helvetica", size=11, weight="bold")
    body_font = tkfont.Font(family="Helvetica", size=10)
    
    # Psychologically friendly color scheme
    text_dark = "#2C3E50"  # Professional Dark Blue/Slate
    text_blue = "#2980B9"  # Calming Blue for scores
    bg_gray = "#F8F9F9"    # Very soft gray for result boxes

    # Title
    tk.Label(root, text="Initial Assessment Results", font=title_font, bg="#FFFFFF", fg=text_dark).pack(pady=(20, 10))

    # Helper function to create result blocks
    def create_result_block(parent, title, score, sentence):
        frame = tk.Frame(parent, bg=bg_gray, padx=15, pady=10, relief=tk.FLAT, borderwidth=1)
        frame.pack(fill=tk.X, padx=30, pady=5)
        
        # Header (e.g., DEPRESSION: 42.5%)
        tk.Label(frame, text=f"{title}: {score:.1f}%", font=header_font, bg=bg_gray, fg=text_blue).pack(anchor="w")
        # Body text with automatic word wrapping
        tk.Label(frame, text=sentence, font=body_font, bg=bg_gray, fg=text_dark, wraplength=550, justify=tk.LEFT).pack(anchor="w", pady=(5,0))

    # Create the 3 blocks
    create_result_block(root, "DEPRESSION", dep_score, dep_text)
    create_result_block(root, "ANXIETY", anx_score, anx_text)
    create_result_block(root, "STRESS", str_score, str_text)

    # Precautionary statement
    precaution = "PRECAUTION: This is just an initial assessment using facial scanning technology and is not a clinical diagnosis."
    tk.Label(root, text=precaution, font=tkfont.Font(family="Helvetica", size=9, slant="italic"), bg="#FFFFFF", fg="#E67E22").pack(pady=(15, 10))

    # Save Results Section
    save_frame = tk.Frame(root, bg="#FFFFFF")
    save_frame.pack(pady=20)

    tk.Label(save_frame, text="If you want to save these results, please enter your email:", font=body_font, bg="#FFFFFF", fg=text_dark).pack()
    
    email_entry = tk.Entry(save_frame, width=40, font=body_font, bg="#ECF0F1", relief=tk.FLAT)
    email_entry.pack(pady=10, ipady=5) # ipady makes the text box a bit taller/modern

    # Submit Button
    submit_btn = tk.Button(save_frame, text="Save to Database", font=header_font, bg="#3498DB", fg="white", 
                           activebackground="#2980B9", activeforeground="white", relief=tk.FLAT, padx=20, pady=5,
                           command=lambda: save_to_database(email_entry.get(), dep_score, anx_score, str_score, root))
    submit_btn.pack()

    # Pause execution here until the user closes the window
    root.mainloop()


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
    
    print("========================================")
    print(" THESIS CAMERA ACTIVE")
    print(" - Press SPACEBAR to capture and analyze")
    print(" - Press 'q' to quit")
    print("========================================")

    while True:
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
            # Grab the current frame for analysis
            gray_cap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_cap = face_cascade.detectMultiScale(gray_cap, scaleFactor=1.3, minNeighbors=5)

            if len(faces_cap) > 0:
                x, y, w, h = faces_cap[0] 
                
                face_crop = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(face_rgb)

                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad(): 
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1)[0] 

                scores = probabilities.cpu().numpy() * 100

                # SENSITIVITY MULTIPLIERS (Adjust these for defense!)
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

                # Open the clean UI window! 
                # (The camera will pause in the background until this window is closed)
                show_results_ui(dep_score, anx_score, str_score, dep_text, anx_text, str_text)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()