import streamlit as st
import streamlit.components.v1 as components
import random
import time
import requests
import plotly.graph_objects as go
import speech_recognition as sr
import datetime
from datetime import timedelta
import hashlib
import json
import os
import sqlite3
import re
import cv2
import numpy as np
import smtplib
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
sqlite3.enable_callback_tracebacks(True)

# NOUVEAUX IMPORTS POUR L'ANALYSE FACIALE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

# ==================== CONFIGURATION ====================
APP_CONFIG = {
    "APP_NAME": "StressApp",
    "APP_URL": "http://localhost:8501",
}

# Configuration Gmail
EMAIL_CONFIG = {
    "SMTP_SERVER": "smtp.gmail.com",
    "SMTP_PORT": 587,
    "SENDER_EMAIL": "lafdiliaya8@gmail.com",
    "SENDER_PASSWORD": "mdvspoblizcfrfoc",
    "EMAIL_ENABLED": True,
    "SMTP_TIMEOUT": 20,
}

# ==================== COMPTES ADMIN PRÉ-DÉFINIS ====================
ADMIN_ACCOUNTS = [
    {
        "email": "shsbasma@gmail.com",
        "username": "basma",
        "password": "basbas2006",
        "full_name": "Basma Shs",
        "gender": "Femme"
    },
    {
        "email": "lafdiliaya8@gmail.com",
        "username": "aya",
        "password": "ayaton2006",
        "full_name": "Aya Lafdili",
        "gender": "Femme"
    }
]

# ==================== SYSTÈME DE COMPARAISON D'ÉMOTIONS OPTIMISÉ ====================

class EmotionComparisonSystem:
    def __init__(self, happy_path, sad_path, max_images_per_class=500):
        """
        Initialiser le système avec les chemins des dossiers Happy et Sad
        max_images_per_class: nombre maximum d'images à charger par catégorie (500 par défaut)
        """
        self.happy_path = happy_path
        self.sad_path = sad_path
        self.max_images_per_class = max_images_per_class
        self.model = None
        self.scaler = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def extract_face_features(self, image):
        """Extraire les caractéristiques du visage pour la comparaison"""
        # Convertir en niveaux de gris
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Redimensionner pour normaliser (plus petit pour accélérer)
        face_resized = cv2.resize(gray, (64, 64))
        
        # Caractéristiques simplifiées pour accélérer
        # 1. Histogramme des pixels
        hist = cv2.calcHist([face_resized], [0], None, [32], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-6)
        
        # 2. Caractéristiques de texture simples
        mean = np.mean(face_resized)
        std = np.std(face_resized)
        
        # 3. Caractéristiques de gradient simples
        grad_x = cv2.Sobel(face_resized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(face_resized, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        mean_grad = np.mean(grad_magnitude)
        
        # Combiner les caractéristiques
        features = np.concatenate([hist, [mean, std, mean_grad]])
        
        return features
    
    def train_model_from_database(self):
        """
        Entraîner le modèle avec un nombre limité d'images pour accélérer
        """
        features_list = []
        labels_list = []
        
        # Créer les barres de progression
        progress_bar_happy = st.progress(0, text="Chargement des images Happy...")
        progress_bar_sad = st.progress(0, text="Chargement des images Sad...")
        status_text = st.empty()
        
        # Charger les images Happy (0 = stress bas)
        status_text.info(f"📂 Chargement des images Happy...")
        happy_images = self._load_images_from_folder_optimized(
            self.happy_path, 
            self.max_images_per_class,
            progress_bar_happy
        )
        
        for i, img in enumerate(happy_images):
            features = self.extract_face_features(img)
            features_list.append(features)
            labels_list.append(0)  # 0 = Happy / Stress bas
        
        # Charger les images Sad (1 = stress élevé)
        status_text.info(f"📂 Chargement des images Sad...")
        sad_images = self._load_images_from_folder_optimized(
            self.sad_path, 
            self.max_images_per_class,
            progress_bar_sad
        )
        
        for i, img in enumerate(sad_images):
            features = self.extract_face_features(img)
            features_list.append(features)
            labels_list.append(1)  # 1 = Sad / Stress élevé
        
        # Nettoyer les barres de progression
        progress_bar_happy.empty()
        progress_bar_sad.empty()
        status_text.empty()
        
        if len(features_list) < 10:
            st.error(f"❌ Pas assez d'images valides trouvées. Happy: {len(happy_images)}, Sad: {len(sad_images)}")
            return False
        
        # Convertir en numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Normaliser les caractéristiques
        with st.spinner("🔄 Normalisation des données..."):
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        
        # Entraîner le modèle SVM (plus rapide avec moins d'images)
        with st.spinner("🤖 Entraînement du modèle..."):
            self.model = SVC(kernel='linear', probability=True, random_state=42, C=1.0)
            self.model.fit(X_scaled, y)
        
        st.success(f"✅ Modèle entraîné avec {len(features_list)} images:")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"😊 Images Happy: {len(happy_images)}")
        with col2:
            st.info(f"😢 Images Sad: {len(sad_images)}")
        
        return True
    
    def _load_images_from_folder_optimized(self, folder_path, max_images, progress_bar):
        """Charger les images de façon optimisée avec limite"""
        images = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        try:
            # Lister tous les fichiers
            all_files = [f for f in os.listdir(folder_path) 
                        if any(f.lower().endswith(ext) for ext in valid_extensions)]
            
            # Prendre un échantillon aléatoire si trop d'images
            if len(all_files) > max_images:
                import random
                all_files = random.sample(all_files, max_images)
            
            total_files = len(all_files)
            
            for idx, file in enumerate(all_files):
                # Mettre à jour la progression
                progress = (idx + 1) / total_files
                progress_bar.progress(progress, text=f"Chargement: {idx+1}/{total_files}")
                
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    # Détecter le visage dans l'image
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                    
                    for (x, y, w, h) in faces:
                        # Extraire uniquement le visage
                        face = img[y:y+h, x:x+w]
                        images.append(face)
                        break  # Prendre le premier visage seulement
                        
        except Exception as e:
            st.error(f"Erreur lors du chargement de {folder_path}: {e}")
        
        return images
    
    def predict_emotion_from_face(self, face_image):
        """Prédire l'émotion à partir d'une image de visage"""
        if self.model is None:
            raise ValueError("⚠️ Modèle non entraîné !")
        
        # Extraire les caractéristiques
        features = self.extract_face_features(face_image)
        features_scaled = self.scaler.transform([features])
        
        # Prédire
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Calculer le score de stress
        if prediction == 0:  # Happy
            stress_score = 20 + (1 - probabilities[0]) * 20
            emotion = "😊 Heureux/Serein"
        else:  # Sad
            stress_score = 70 + probabilities[1] * 30
            emotion = "😟 Triste/Stressé"
        
        confidence = max(probabilities) * 100
        
        return stress_score, emotion, confidence
    
    def analyze_live_face_with_comparison(self, duration=8):
        """Analyser le visage en direct"""
        if self.model is None:
            st.error("❌ Modèle non entraîné !")
            return 50, "medium", "Erreur"
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Impossible d'accéder à la caméra")
            return 50, "medium", "Erreur"
        
        # Variables pour l'analyse
        happy_count = 0
        sad_count = 0
        total_frames = 0
        stress_scores = []
        
        start_time = time.time()
        
        # Éléments Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_placeholder = st.empty()
        
        try:
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                
                # Détecter les visages
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60)
                )
                
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        # Extraire le visage
                        face = frame[y:y+h, x:x+w]
                        
                        try:
                            stress_score, emotion, confidence = self.predict_emotion_from_face(face)
                            
                            stress_scores.append(stress_score)
                            
                            if stress_score < 40:
                                happy_count += 1
                                color = (0, 255, 0)  # Vert
                            elif stress_score > 60:
                                sad_count += 1
                                color = (0, 0, 255)  # Rouge
                            else:
                                color = (255, 255, 0)  # Jaune
                            
                            # Dessiner le rectangle
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            
                            # Afficher le score
                            cv2.putText(
                                frame, 
                                f"Stress: {stress_score:.1f}%", 
                                (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, 
                                color, 
                                2
                            )
                            
                        except Exception as e:
                            continue
                
                # Afficher les stats en direct
                status_text.markdown(f"""
                **Analyse en direct:**
                - 😊 Happy: {happy_count} frames
                - 😢 Sad: {sad_count} frames
                - 📊 Total: {total_frames} frames
                """)
                
                # Afficher le frame (redimensionné pour accélérer)
                frame_small = cv2.resize(frame, (640, 480))
                frame_placeholder.image(frame_small, channels="BGR", use_container_width=True)
                
                # Progression
                elapsed = time.time() - start_time
                progress = elapsed / duration
                progress_bar.progress(min(progress, 1.0))
                
                # Petit délai pour ne pas surcharger
                cv2.waitKey(1)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            progress_bar.empty()
            status_text.empty()
            frame_placeholder.empty()
        
        # Calculer le score final
        if total_frames == 0:
            return 50, "medium", "Aucune analyse"
        
        avg_stress = np.mean(stress_scores) if stress_scores else 50
        
        # Déterminer le niveau
        if avg_stress < 34:
            level = "low"
        elif avg_stress < 67:
            level = "medium"
        else:
            level = "high"
        
        return avg_stress, level, f"Happy: {happy_count}, Sad: {sad_count}"

# ==================== ANALYSE DES ÉMOTIONS FACIALES OPTIMISÉE ====================

def analyze_facial_emotions(duration=8):
    """
    Analyser les émotions faciales avec optimisation
    """
    # Chemins vers vos dossiers
    HAPPY_PATH = r"C:\Users\pc\Downloads\archive (5)\Data\Happy"
    SAD_PATH = r"C:\Users\pc\Downloads\archive (5)\Data\Sad"
    
    # Vérifier que les dossiers existent
    if not os.path.exists(HAPPY_PATH):
        st.error(f"❌ Dossier Happy introuvable")
        return simple_face_detection(duration)
    
    if not os.path.exists(SAD_PATH):
        st.error(f"❌ Dossier Sad introuvable")
        return simple_face_detection(duration)
    
    # Compter les images (rapide sans charger)
    n_happy = len([f for f in os.listdir(HAPPY_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    n_sad = len([f for f in os.listdir(SAD_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    st.info(f"📊 **Base de données trouvée:** {n_happy} images Happy, {n_sad} images Sad")
    
    # Demander à l'utilisateur combien d'images utiliser
    col1, col2 = st.columns(2)
    with col1:
        max_happy = st.number_input(
            "😊 Max images Happy à utiliser",
            min_value=50,
            max_value=min(1000, n_happy),
            value=min(300, n_happy),
            step=50
        )
    with col2:
        max_sad = st.number_input(
            "😢 Max images Sad à utiliser",
            min_value=50,
            max_value=min(1000, n_sad),
            value=min(300, n_sad),
            step=50
        )
    
    st.warning(f"⏱️ Entraînement avec {max_happy + max_sad} images (cela prendra {((max_happy + max_sad)//100)*2} secondes environ)")
    
    # Créer le système avec limite d'images
    emotion_system = EmotionComparisonSystem(
        HAPPY_PATH, 
        SAD_PATH, 
        max_images_per_class=max(max_happy, max_sad)
    )
    
    # Entraîner le modèle
    success = emotion_system.train_model_from_database()
    
    if not success:
        st.warning("⚠ Utilisation de l'analyse simple")
        return simple_face_detection(duration)
    
    # Analyser le visage
    st.success("✅ Modèle prêt! Analyse du visage en cours...")
    stress_score, level, stats = emotion_system.analyze_live_face_with_comparison(duration)
    
    # Afficher les statistiques
    with st.expander("📊 Statistiques détaillées", expanded=True):
        st.write(stats)
        
        # Extraire les comptes
        try:
            happy_count = int(stats.split("Happy:")[1].split(",")[0].strip())
            sad_count = int(stats.split("Sad:")[1].strip())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("😊 Frames correspondant à Happy", happy_count)
            with col2:
                st.metric("😟 Frames correspondant à Sad", sad_count)
        except:
            pass
    
    return stress_score, level

def simple_face_detection(duration=8):
    """Détection simple de visage avec OpenCV (fallback)"""
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Impossible d'accéder à la caméra")
        return 50, "medium"
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    face_detections = 0
    total_frames = 0
    
    start_time = datetime.datetime.now()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_placeholder = st.empty()

    try:
        while (datetime.datetime.now() - start_time).seconds < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            total_frames += 1
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                face_detections += 1
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.putText(frame, f"Visages: {len(faces)}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)

            elapsed = (datetime.datetime.now() - start_time).seconds
            progress = elapsed / duration
            progress_bar.progress(progress)
            status_text.text(f"⏱️ {elapsed}/{duration} secondes")
            
            cv2.waitKey(1)
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return 50, "medium"
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        progress_bar.empty()
        status_text.empty()
        frame_placeholder.empty()

    if total_frames == 0:
        return 50, "medium"
    
    face_detection_rate = face_detections / total_frames if total_frames > 0 else 0
    
    if face_detection_rate > 0.7:
        stress_score = 20
    elif face_detection_rate > 0.4:
        stress_score = 50
    else:
        stress_score = 80
    
    import random
    stress_score = min(100, max(0, stress_score + random.randint(-10, 10)))
    
    if stress_score < 34:
        level = "low"
    elif stress_score < 67:
        level = "medium"
    else:
        level = "high"

    return stress_score, level

def detect_stress_from_face(duration=8):
    """Wrapper pour l'analyse faciale avec base de données"""
    return analyze_facial_emotions(duration)

# ==================== FONCTIONS EXISTANTES ====================
def recognize_mixed_speech(recognizer, audio):
    results = {}
    languages = {
        "ar-MA": "Darija",
        "fr-FR": "French",
        "en-US": "English",
        "ar-ar": "Arabic"
    }

    for lang_code, lang_name in languages.items():
        try:
            text = recognizer.recognize_google(audio, language=lang_code)
            results[lang_name] = text
        except:
            pass

    if not results:
        return None, None

    best_lang = max(results, key=lambda k: len(results[k]))
    return results[best_lang], best_lang

# ==================== FONCTIONS EMAIL ====================

def send_welcome_email(recipient_email, username, full_name):
    """Envoyer un email de bienvenue"""
    if not EMAIL_CONFIG["EMAIL_ENABLED"]:
        return False, "Email non configuré"
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"🎉 Bienvenue sur {APP_CONFIG['APP_NAME']} !"
        msg['From'] = f"{APP_CONFIG['APP_NAME']} <{EMAIL_CONFIG['SENDER_EMAIL']}>"
        msg['To'] = recipient_email
        
        signup_date = datetime.datetime.now().strftime("%d/%m/%Y à %H:%M")
        
        html_content = f"""
<!DOCTYPE html>
<html>
  <body style="margin:0;padding:0;background-color:#eaf4ff;font-family:Arial,sans-serif;">
    <div style="max-width:600px;margin:40px auto;background:#ffffff;border-radius:16px;
                box-shadow:0 10px 30px rgba(0,0,0,0.1);padding:32px;text-align:center;">
      <div style="width:72px;height:72px;margin:0 auto 20px;background:#e0f0ff;border-radius:50%;
                  display:flex;align-items:center;justify-content:center;
                  font-size:32px;color:#2f80ed;">
        🔐
      </div>
      <h2 style="margin:0 0 10px;color:#2f80ed;font-size:22px;">
        Vérification de votre email
      </h2>
      <p style="margin:0 0 24px;color:#555;font-size:15px;">
        Utilisez le code ci-dessous pour confirmer votre adresse email.
      </p>
      <div style="display:inline-block;padding:14px 24px;font-size:28px;
                  letter-spacing:10px;font-weight:bold;color:#2f80ed;
                  background:#f2f8ff;border-radius:10px;margin-bottom:20px;">
        {otp_code}
      </div>
      <p style="margin:16px 0 6px;color:#666;font-size:14px;">
        ⏳ Ce code est valable pendant <strong>10 minutes</strong>.
      </p>
      <p style="margin:0 0 24px;color:#666;font-size:14px;">
        Après validation, vous pourrez créer votre code permanent.
      </p>
      <hr style="border:none;border-top:1px solid #e0e0e0;margin:24px 0;">
      <p style="font-size:12px;color:#999;margin:0;">
        Si vous n'êtes pas à l'origine de cette demande, ignorez ce message.<br>
        © {APP_CONFIG['APP_NAME']}
      </p>
    </div>
  </body>
</html>
"""
        
        text_content = f"""
        BIENVENUE SUR {APP_CONFIG['APP_NAME']}
        
        Bonjour {full_name},
        
        Votre compte a été créé avec succès !
        
        Informations :
        - Nom complet : {full_name}
        - Nom d'utilisateur : {username}
        - Email : {recipient_email}
        - Date : {signup_date}
        
        Accédez à votre compte : {APP_CONFIG['APP_URL']}
        
        © 2024 {APP_CONFIG['APP_NAME']}
        """
        
        msg.attach(MIMEText(text_content, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))
        
        server = smtplib.SMTP(EMAIL_CONFIG["SMTP_SERVER"], EMAIL_CONFIG["SMTP_PORT"], timeout=EMAIL_CONFIG.get("SMTP_TIMEOUT", 20))
        try:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_CONFIG["SENDER_EMAIL"], EMAIL_CONFIG["SENDER_PASSWORD"])
            server.send_message(msg)
        finally:
            try:
                server.quit()
            except Exception:
                pass
        
        return True, "Email envoyé avec succès"
        
    except Exception as e:
        print("❌ ERREUR SMTP (welcome):", e)
        print(traceback.format_exc())
        return False, f"Erreur d'envoi d'email: {str(e)}"
# ==================== FONCTION DE NORMALISATION DU TEXTE ====================

def normalize_speech_text(text):
    """
    Normaliser le texte de la voix pour qu'il corresponde au texte écrit
    """
    if not text:
        return text
    
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer les mots de remplissage courants en Darija/Français
    filler_words = [
        'euh', 'heu', 'hum', 'ah', 'oh', 'bah', 'ben', 'quoi', 'enfin',
        'du coup', 'en fait', 'genre', 'style', 'voilà', 'voila',
        'hmm', 'mmm', 'eh', 'hé', 'ha', 'hi', 'ho'
    ]
    
    for filler in filler_words:
        text = text.replace(filler, '')
    
    # Supprimer les répétitions (ex: "je je suis" -> "je suis")
    words = text.split()
    unique_words = []
    for i, word in enumerate(words):
        if i == 0 or word != words[i-1]:
            unique_words.append(word)
    
    text = ' '.join(unique_words)
    
    # Supprimer les espaces multiples
    text = ' '.join(text.split())
    
    return text

def compare_texts(text1, text2):
    """
    Comparer deux textes pour voir s'ils sont similaires
    """
    from difflib import SequenceMatcher
    
    # Nettoyer les deux textes
    text1_clean = normalize_speech_text(text1)
    text2_clean = normalize_speech_text(text2)
    
    # Calculer la similarité
    similarity = SequenceMatcher(None, text1_clean, text2_clean).ratio()
    
    return similarity, text1_clean, text2_clean
# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Stress Detection Dashboard",
    page_icon="🔷",
    layout="centered",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:5000/predict"

# ==================== BASE DE DONNÉES ====================
def _table_columns(conn, table_name: str):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cur.fetchall()}

def init_database():
    """Initialiser / migrer la base de données SQLite"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Table USERS - Version corrigée avec password_hash nullable
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            pin_hash TEXT,
            full_name TEXT,
            email TEXT UNIQUE,
            gender TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            role TEXT DEFAULT 'user',
            is_active INTEGER DEFAULT 1
        )
    ''')

    # Table OTP
    c.execute('''
        CREATE TABLE IF NOT EXISTS email_otps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            otp_hash TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Table pour les tokens de réinitialisation
    c.execute('''
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            token_hash TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Migration douce
    cols = _table_columns(conn, "users")
    if "pin_hash" not in cols:
        c.execute("ALTER TABLE users ADD COLUMN pin_hash TEXT")
    if "gender" not in cols:
        c.execute("ALTER TABLE users ADD COLUMN gender TEXT")
    if "email" not in cols:
        c.execute("ALTER TABLE users ADD COLUMN email TEXT")
    if "role" not in cols:
        c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
    if "is_active" not in cols:
        c.execute("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1")
    
    conn.commit()
    conn.close()
    print("✅ Base de données initialisée / migrée")

def create_admin_accounts():
    """Créer les comptes admin pour Basma et Aya"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    for admin in ADMIN_ACCOUNTS:
        try:
            # Vérifier si l'admin existe déjà
            c.execute("SELECT COUNT(*) FROM users WHERE email = ?", (admin["email"],))
            if c.fetchone()[0] == 0:
                # Créer le compte admin
                c.execute('''
                    INSERT INTO users (
                        username, 
                        password_hash, 
                        pin_hash, 
                        full_name, 
                        email, 
                        gender, 
                        role, 
                        is_active,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'admin', 1, CURRENT_TIMESTAMP)
                ''', (
                    admin["username"],
                    hash_password(admin["password"]),  # password_hash
                    hash_password(admin["password"]),  # pin_hash (même valeur)
                    admin["full_name"],
                    admin["email"],
                    admin["gender"]
                ))
                print(f"✅ Compte admin créé: {admin['username']} - {admin['email']}")
            else:
                # Mettre à jour le rôle admin si nécessaire
                c.execute("UPDATE users SET role='admin', pin_hash=?, password_hash=? WHERE email=?", 
                         (hash_password(admin["password"]), hash_password(admin["password"]), admin["email"]))
                print(f"✅ Compte existant promu admin: {admin['username']}")
        except Exception as e:
            print(f"❌ Erreur création admin {admin['username']}: {e}")
    
    conn.commit()
    conn.close()

def debug_check_accounts():
    """Vérifier que les comptes admin sont bien dans la base"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    print("🔍 VÉRIFICATION DES COMPTES ADMIN:")
    for admin in ADMIN_ACCOUNTS:
        c.execute("SELECT id, username, email, role, pin_hash FROM users WHERE email = ?", (admin["email"],))
        result = c.fetchone()
        if result:
            print(f"✅ Compte trouvé: {result[1]} ({result[2]}) - Rôle: {result[3]}")
            print(f"   PIN hash: {result[4][:20]}...")
        else:
            print(f"❌ Compte NON trouvé: {admin['email']}")
    
    # Lister tous les utilisateurs
    print("\n📋 TOUS LES UTILISATEURS:")
    c.execute("SELECT id, username, email, role FROM users")
    for row in c.fetchall():
        print(f"   - {row[1]} ({row[2]}) - {row[3]}")
    
    conn.close()

# Initialiser la base de données
init_database()

# Créer les comptes admin
create_admin_accounts()

# Vérifier les comptes
debug_check_accounts()

# ==================== ADMINS (FORCÉS) ====================
ADMINS_EMAILS = {
    "shsbasma@gmail.com",     # Basma
    "lafdiliaya8@gmail.com",  # Aya
}

def promote_admins_by_email():
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        for email in ADMINS_EMAILS:
            c.execute("UPDATE users SET role='admin' WHERE email=?", (email,))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Erreur promotion admins:", e)

# Appliquer au démarrage
promote_admins_by_email()

# ==================== FONCTIONS POUR MOT DE PASSE OUBLIÉ ====================
def send_password_reset_email(email: str, reset_token: str):
    """Envoyer un email de réinitialisation de mot de passe"""
    if not EMAIL_CONFIG.get("EMAIL_ENABLED", False):
        return False, "Email désactivé (EMAIL_ENABLED=False)"

    try:
        reset_url = f"{APP_CONFIG['APP_URL']}?reset_token={reset_token}"
        expiration_hours = 24
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"🔐 Réinitialisation de votre mot de passe - {APP_CONFIG['APP_NAME']}"
        msg['From'] = f"{APP_CONFIG['APP_NAME']} <{EMAIL_CONFIG['SENDER_EMAIL']}>"
        msg['To'] = email

        text_content = f"""Bonjour,

Vous avez demandé à réinitialiser votre mot de passe pour {APP_CONFIG['APP_NAME']}.

Pour réinitialiser votre mot de passe, cliquez sur le lien ci-dessous :

{reset_url}

Ce lien est valable pendant {expiration_hours} heures.

Si vous n'êtes pas à l'origine de cette demande, vous pouvez ignorer ce message en toute sécurité.

— L'équipe {APP_CONFIG['APP_NAME']}
"""

        html_content = f"""
        <html>
          <body style="font-family: Arial, sans-serif;">
            <div style="max-width:600px;margin:auto;border:1px solid #eee;padding:20px;border-radius:8px;">
              <h2 style="margin-top:0;color:#2f80ed;">🔐 Réinitialisation de mot de passe</h2>
              <p>Vous avez demandé à réinitialiser votre mot de passe pour <strong>{APP_CONFIG['APP_NAME']}</strong>.</p>
              
              <div style="text-align:center;margin:30px 0;">
                <a href="{reset_url}" 
                   style="background-color:#2f80ed;color:white;padding:12px 24px;
                          text-decoration:none;border-radius:6px;font-weight:bold;
                          display:inline-block;">
                  Réinitialiser mon mot de passe
                </a>
              </div>
              
              <p style="color:#666;margin-top:20px;">
                Vous pouvez également copier ce lien :<br>
                <code style="background:#f5f5f5;padding:8px;border-radius:4px;word-break:break-all;">
                  {reset_url}
                </code>
              </p>
              
              <p style="color:#777;">
                ⏳ Ce lien est valable pendant <strong>{expiration_hours} heures</strong>.
              </p>
              
              <hr style="border:none;border-top:1px solid #eee;margin:20px 0;">
              <p style="color:#999;font-size:12px;">
                Si vous n'êtes pas à l'origine de cette demande, ignorez ce message.<br>
                © {APP_CONFIG['APP_NAME']}
              </p>
            </div>
          </body>
        </html>
        """

        msg.attach(MIMEText(text_content, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))

        server = smtplib.SMTP(EMAIL_CONFIG["SMTP_SERVER"], EMAIL_CONFIG["SMTP_PORT"], timeout=EMAIL_CONFIG.get("SMTP_TIMEOUT", 20))
        try:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_CONFIG["SENDER_EMAIL"], EMAIL_CONFIG["SENDER_PASSWORD"])
            server.send_message(msg)
        finally:
            try:
                server.quit()
            except Exception:
                pass

        return True, "Email de réinitialisation envoyé."
    except Exception as e:
        print("❌ ERREUR SMTP (reset):", e)
        print(traceback.format_exc())
        return False, f"Erreur d'envoi d'email: {str(e)}"

def create_reset_token(email: str):
    """Créer un token de réinitialisation sécurisé"""
    import secrets
    import string
    
    token = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expires_at = int(time.time()) + (24 * 60 * 60)
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM password_reset_tokens WHERE email = ?", (email,))
    c.execute(
        "INSERT INTO password_reset_tokens (email, token_hash, expires_at) VALUES (?, ?, ?)",
        (email, token_hash, expires_at)
    )
    conn.commit()
    conn.close()
    
    return token

def verify_reset_token(email: str, token: str):
    """Vérifier la validité d'un token de réinitialisation"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute(
        "SELECT token_hash, expires_at FROM password_reset_tokens WHERE email = ? ORDER BY created_at DESC LIMIT 1",
        (email,)
    )
    row = c.fetchone()
    conn.close()
    
    if not row:
        return False, "Aucun token de réinitialisation trouvé."
    
    token_hash, expires_at = row
    if int(time.time()) > int(expires_at):
        return False, "Le lien de réinitialisation a expiré."
    
    if hashlib.sha256(token.encode()).hexdigest() != token_hash:
        return False, "Token de réinitialisation invalide."
    
    return True, "Token valide."

def delete_reset_token(email: str):
    """Supprimer le token après utilisation"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM password_reset_tokens WHERE email = ?", (email,))
    conn.commit()
    conn.close()

# ==================== VÉRIFICATION DES LIENS DE RÉINITIALISATION DANS L'URL ====================
def check_reset_token_from_url():
    """Vérifier si un token de réinitialisation est présent dans l'URL"""
    try:
        query_params = st.query_params.to_dict()
        
        if 'reset_token' in query_params:
            token = query_params['reset_token']
            
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT email, expires_at FROM password_reset_tokens")
            rows = c.fetchall()
            conn.close()
            
            for row in rows:
                email, expires_at = row
                token_hash = hashlib.sha256(token.encode()).hexdigest()
                
                c2 = conn = sqlite3.connect('users.db')
                c2.execute("SELECT token_hash FROM password_reset_tokens WHERE email = ?", (email,))
                stored_hash = c2.fetchone()
                c2.close()
                
                if stored_hash and token_hash == stored_hash[0]:
                    if int(time.time()) <= int(expires_at):
                        st.session_state.reset_stage = "reset"
                        st.session_state.reset_email = email
                        st.session_state.reset_token = token
                        st.session_state.auth_view = "login"
                        st.rerun()
                        return True
                    else:
                        st.error("❌ Le lien de réinitialisation a expiré.")
                        return False
            
            st.error("❌ Lien de réinitialisation invalide.")
            return False
    except Exception as e:
        print(f"Erreur lors de la vérification du token: {e}")
    return False

# ---------------- SESSION STATE ----------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "history" not in st.session_state:
    st.session_state.history = []
if "audio_text" not in st.session_state:
    st.session_state.audio_text = ""
if "recording_status" not in st.session_state:
    st.session_state.recording_status = "ready"
if "show_connection_info" not in st.session_state:
    st.session_state.show_connection_info = False
if "show_info_page" not in st.session_state:
    st.session_state.show_info_page = False
if "show_edit_profile" not in st.session_state:
    st.session_state.show_edit_profile = False
if "show_manage_users" not in st.session_state:
    st.session_state.show_manage_users = False

# OTP flow states
if "otp_stage" not in st.session_state:
    st.session_state.otp_stage = "enter_email"
if "otp_email" not in st.session_state:
    st.session_state.otp_email = ""
if "otp_dev_code" not in st.session_state:
    st.session_state.otp_dev_code = None

# Reset password flow states
if "reset_stage" not in st.session_state:
    st.session_state.reset_stage = None
if "reset_email" not in st.session_state:
    st.session_state.reset_email = ""
if "reset_token" not in st.session_state:
    st.session_state.reset_token = ""

# ---------------- LANGUAGE DISPLAY MAP ----------------
lang_map = {
    "french": "Français 🇫🇷",
    "english": "English 🇬🇧",
    "arabic": "العربية 🇲🇦",
    "darija_lat": "Darija (Latin) 🇲🇦",
    "mixed": "Mixte 🌍"
}

# ==================== FONCTIONS UTILITAIRES ====================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email: str):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email or ""):
        return True, "Email valide"
    return False, "Format d'email invalide"

def validate_pin(pin: str):
    if not pin or len(pin) < 4:
        return False, "Le code permanent doit contenir au moins 4 caractères"
    return True, "Code permanent valide"

def validate_password(password: str):
    if len(password) < 6:
        return False, "Le mot de passe doit contenir au moins 6 caractères"
    if not any(char.isdigit() for char in password):
        return False, "Le mot de passe doit contenir au moins un chiffre"
    if not any(char.isalpha() for char in password):
        return False, "Le mot de passe doit contenir au moins une lettre"
    return True, "Mot de passe valide"

def validate_username(username: str):
    if len((username or "").strip()) < 3:
        return False, "Le nom d'utilisateur doit contenir au moins 3 caractères"
    if not username.isalnum():
        return False, "Le nom d'utilisateur ne doit contenir que des lettres et chiffres"
    return True, "Nom d'utilisateur valide"

def validate_full_name(full_name: str):
    name = (full_name or "").strip()
    if len(name) < 2:
        return False, "Le nom complet est requis"
    if not re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ\s'\-]{2,}$", name):
        return False, "Le nom complet ne doit contenir que des lettres (accents ok), espaces, tirets ou apostrophes"
    return True, "Nom complet valide"

# ==================== EMAIL OTP ====================
def _generate_otp() -> str:
    return f"{random.randint(0, 999999):06d}"

def _save_otp(email: str, otp_code: str, ttl_seconds: int = 10 * 60):
    expires_at = int(time.time()) + ttl_seconds
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM email_otps WHERE email = ?", (email,))
    c.execute(
        "INSERT INTO email_otps (email, otp_hash, expires_at) VALUES (?, ?, ?)",
        (email, hash_password(otp_code), expires_at),
    )
    conn.commit()
    conn.close()
    return expires_at

def _verify_otp(email: str, otp_code: str):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute(
        "SELECT otp_hash, expires_at FROM email_otps WHERE email = ? ORDER BY id DESC LIMIT 1",
        (email,),
    )
    row = c.fetchone()
    conn.close()

    if not row:
        return False, "Aucun code OTP trouvé."
    otp_hash, expires_at = row
    if int(time.time()) > int(expires_at):
        return False, "Le code OTP a expiré."
    if hash_password(otp_code) != otp_hash:
        return False, "Code OTP incorrect."
    return True, "OTP vérifié."

def send_otp_email(recipient_email: str, otp_code: str):
    if not EMAIL_CONFIG.get("EMAIL_ENABLED", False):
        return False, "Email désactivé"

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Votre code OTP - {APP_CONFIG['APP_NAME']}"
        msg['From'] = f"{APP_CONFIG['APP_NAME']} <{EMAIL_CONFIG['SENDER_EMAIL']}>"
        msg['To'] = recipient_email

        text_content = f"""Bonjour,

Voici votre code de vérification à usage unique :

{otp_code}

Ce code est valable pendant 10 minutes.

— L'équipe StressApp
"""

        html_content = f"""
        <html>
          <body style="font-family: Arial, sans-serif;">
            <div style="max-width:600px;margin:auto;border:1px solid #eee;padding:20px;border-radius:8px;">
              <h2 style="margin-top:0;">🔐 Code de vérification</h2>
              <p>Voici votre code de vérification à usage unique :</p>
              <div style="font-size:28px;font-weight:bold;letter-spacing:6px;padding:12px 16px;background:#f5f5f5;display:inline-block;border-radius:8px;">
                {otp_code}
              </div>
              <p style="color:#666;margin-top:16px;">⏳ Ce code est valable pendant <strong>10 minutes</strong>.</p>
              <hr style="border:none;border-top:1px solid #eee;margin:20px 0;">
              <p style="color:#999;font-size:12px">© {APP_CONFIG['APP_NAME']}</p>
            </div>
          </body>
        </html>
        """

        msg.attach(MIMEText(text_content, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))

        server = smtplib.SMTP(EMAIL_CONFIG["SMTP_SERVER"], EMAIL_CONFIG["SMTP_PORT"], timeout=EMAIL_CONFIG.get("SMTP_TIMEOUT", 20))
        try:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_CONFIG["SENDER_EMAIL"], EMAIL_CONFIG["SENDER_PASSWORD"])
            server.send_message(msg)
        finally:
            try:
                server.quit()
            except Exception:
                pass

        return True, "OTP envoyé."
    except Exception as e:
        print("❌ ERREUR SMTP (otp):", e)
        print(traceback.format_exc())
        return False, f"Erreur d'envoi d'OTP: {str(e)}"

# ==================== AUTH ====================
def _get_user_by_username(username: str):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        SELECT id, username, password_hash, pin_hash, full_name, email, gender, created_at, role
        FROM users
        WHERE username = ? AND is_active = 1
    ''', (username,))
    user = c.fetchone()
    conn.close()
    return user

def _get_user_by_email(email: str):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        SELECT id, username, password_hash, pin_hash, full_name, email, gender, created_at, role
        FROM users
        WHERE email = ? AND is_active = 1
    ''', (email,))
    user = c.fetchone()
    conn.close()
    return user

def authenticate_classic(username: str, password: str):
    try:
        user = _get_user_by_username(username)
        if not user:
            return False, None
        user_id, username_db, password_hash_db, pin_hash_db, full_name, email, gender, created_at, role = user
        if not password_hash_db:
            return False, None
        if hash_password(password) != password_hash_db:
            return False, None
        return True, {
            "id": user_id,
            "username": username_db,
            "full_name": full_name,
            "email": email,
            "gender": gender,
            "role": role,
            "created_at": created_at,
        }
    except Exception as e:
        print(f"Erreur d'authentification (classic): {e}")
        return False, None

def authenticate_email_pin(email: str, pin: str):
    try:
        user = _get_user_by_email(email)
        if not user:
            print(f"❌ Compte introuvable pour l'email: {email}")
            return False, None, "Compte introuvable."
        
        user_id, username_db, password_hash_db, pin_hash_db, full_name, email_db, gender, created_at, role = user
        
        print(f"🔐 Tentative de connexion pour {email_db}")
        print(f"   PIN fourni: {pin}")
        print(f"   PIN hash DB: {pin_hash_db}")
        print(f"   PIN hash calculé: {hash_password(pin)}")
        
        if not pin_hash_db:
            return False, None, "Aucun code permanent enregistré."
        
        if hash_password(pin) != pin_hash_db:
            print("❌ Code permanent incorrect")
            return False, None, "Code permanent incorrect."
        
        print("✅ Authentification réussie!")
        return True, {
            "id": user_id,
            "username": username_db,
            "full_name": full_name,
            "email": email_db,
            "gender": gender,
            "role": role,
            "created_at": created_at,
        }, "OK"
    except Exception as e:
        print(f"❌ Erreur d'authentification: {e}")
        import traceback
        traceback.print_exc()
        return False, None, "Erreur technique."

def upsert_user_with_pin(email: str, pin: str, full_name: str = None, username: str = None, gender: str = None):
    """Amélioré avec meilleure gestion des erreurs"""
    
    # Validation de l'email
    if not email or not email.strip():
        return False, "❌ L'email est requis"
    
    email_valid, email_msg = validate_email(email)
    if not email_valid:
        return False, f"❌ {email_msg}"
    
    # Validation du PIN
    if not pin:
        return False, "❌ Le code permanent est requis"
    
    pin_valid, pin_msg = validate_pin(pin)
    if not pin_valid:
        return False, f"❌ {pin_msg}"
    
    # Nettoyer l'email
    email = email.strip().lower()
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        # Vérifier si l'utilisateur existe
        c.execute('SELECT id, username FROM users WHERE email = ?', (email,))
        existing = c.fetchone()
        
        if existing:
            # Mise à jour de l'utilisateur existant
            c.execute('''
                UPDATE users 
                SET pin_hash = ?,
                    full_name = COALESCE(?, full_name),
                    gender = COALESCE(?, gender)
                WHERE email = ?
            ''', (
                hash_password(pin),
                full_name.strip() if full_name else None,
                gender,
                email
            ))
            message = "✅ Code permanent mis à jour avec succès"
            
        else:
            # Validation des champs requis pour nouvel utilisateur
            if not full_name or not full_name.strip():
                conn.close()
                return False, "❌ Le nom complet est requis"
            
            full_name_valid, full_name_msg = validate_full_name(full_name)
            if not full_name_valid:
                conn.close()
                return False, f"❌ {full_name_msg}"
            
            if not username or not username.strip():
                conn.close()
                return False, "❌ Le nom d'utilisateur est requis"
            
            # Nettoyer le username
            username = username.strip().lower()
            
            # Vérifier que le username est valide
            username_valid, username_msg = validate_username(username)
            if not username_valid:
                conn.close()
                return False, f"❌ {username_msg}"
            
            # Vérifier si le username existe déjà
            c.execute('SELECT COUNT(*) FROM users WHERE username = ?', (username,))
            if c.fetchone()[0] > 0:
                conn.close()
                return False, f"❌ Le nom d'utilisateur '{username}' est déjà pris"
            
            # Vérifier si l'email existe déjà
            c.execute('SELECT COUNT(*) FROM users WHERE email = ?', (email,))
            if c.fetchone()[0] > 0:
                conn.close()
                return False, f"❌ L'email '{email}' est déjà utilisé"
            
            # Création du nouvel utilisateur - avec password_hash NULL
            c.execute('''
                INSERT INTO users (
                    username, 
                    pin_hash, 
                    password_hash,  -- On met NULL pour les utilisateurs PIN-only
                    full_name, 
                    email, 
                    gender, 
                    role, 
                    is_active,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'user', 1, CURRENT_TIMESTAMP)
            ''', (
                username,
                hash_password(pin),
                None,  # password_hash NULL pour les utilisateurs sans mot de passe
                full_name.strip(),
                email,
                gender
            ))
            message = "✅ Compte créé avec succès"
        
        conn.commit()
        conn.close()
        return True, message
        
    except sqlite3.IntegrityError as e:
        conn.close()
        error_msg = str(e)
        if "UNIQUE constraint failed: users.username" in error_msg:
            return False, "❌ Ce nom d'utilisateur est déjà pris"
        elif "UNIQUE constraint failed: users.email" in error_msg:
            return False, "❌ Cet email est déjà utilisé"
        else:
            return False, f"❌ Erreur de base de données: {error_msg}"
            
    except Exception as e:
        conn.close()
        print(f"Erreur inattendue dans upsert_user_with_pin: {e}")
        print(traceback.format_exc())
        return False, f"❌ Erreur inattendue: {str(e)}"

# ==================== PROFIL ====================
def update_user_profile(user_id, full_name=None, email=None, gender=None, current_password=None, new_password=None, current_pin=None, new_pin=None):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('SELECT password_hash, pin_hash, username FROM users WHERE id = ?', (user_id,))
        current_data = c.fetchone()
        if not current_data:
            conn.close()
            return False, "Utilisateur non trouvé"

        password_hash_db, pin_hash_db, username = current_data

        if new_password:
            if not current_password:
                conn.close()
                return False, "Mot de passe actuel requis pour changer le mot de passe"
            if not password_hash_db or hash_password(current_password) != password_hash_db:
                conn.close()
                return False, "Mot de passe actuel incorrect"
            password_valid, password_msg = validate_password(new_password)
            if not password_valid:
                conn.close()
                return False, f"Nouveau mot de passe invalide: {password_msg}"
            c.execute('UPDATE users SET password_hash = ? WHERE id = ?', (hash_password(new_password), user_id))

        if new_pin:
            if not current_pin:
                conn.close()
                return False, "Code permanent actuel requis pour changer le code"
            if not pin_hash_db or hash_password(current_pin) != pin_hash_db:
                conn.close()
                return False, "Code permanent actuel incorrect"
            pin_valid, pin_msg = validate_pin(new_pin)
            if not pin_valid:
                conn.close()
                return False, f"Nouveau code permanent invalide: {pin_msg}"
            c.execute('UPDATE users SET pin_hash = ? WHERE id = ?', (hash_password(new_pin), user_id))

        if full_name is not None:
            full_name_valid, full_name_msg = validate_full_name(full_name)
            if not full_name_valid:
                conn.close()
                return False, f"Nom complet invalide: {full_name_msg}"
            c.execute('UPDATE users SET full_name = ? WHERE id = ?', (full_name.strip(), user_id))

        if gender is not None:
            if gender not in ["Homme", "Femme"]:
                conn.close()
                return False, "Sexe invalide"
            c.execute('UPDATE users SET gender = ? WHERE id = ?', (gender, user_id))

        if email is not None:
            email_valid, email_msg = validate_email(email)
            if not email_valid:
                conn.close()
                return False, f"Email invalide: {email_msg}"
            c.execute('SELECT COUNT(*) FROM users WHERE email = ? AND id != ?', (email, user_id))
            if c.fetchone()[0] > 0:
                conn.close()
                return False, "Cet email est déjà utilisé par un autre compte"
            c.execute('UPDATE users SET email = ? WHERE id = ?', (email, user_id))

        conn.commit()
        conn.close()

        if full_name is not None:
            st.session_state.current_user['full_name'] = full_name.strip()
        if email is not None:
            st.session_state.current_user['email'] = email
        if gender is not None:
            st.session_state.current_user['gender'] = gender

        return True, "✅ Profil mis à jour avec succès !"

    except sqlite3.Error as e:
        return False, f"❌ Erreur de base de données: {str(e)}"
    except Exception as e:
        return False, f"❌ Erreur inattendue: {str(e)}"

# ==================== FONCTIONS DE GESTION DES UTILISATEURS (ADMIN) ====================
def get_all_users():
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('''
            SELECT id, username, full_name, email, gender, role, created_at, is_active
            FROM users
            ORDER BY created_at DESC
        ''')

        users = c.fetchall()
        conn.close()
        return users
    except Exception as e:
        print(f"Erreur récupération utilisateurs: {e}")
        return []

def delete_user(user_id):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('SELECT username, email FROM users WHERE id = ?', (user_id,))
        row = c.fetchone()

        if row:
            username, email = row
            if email in ADMINS_EMAILS:
                conn.close()
                return False, "❌ Impossible de supprimer les comptes admin principaux"

            c.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()
            conn.close()

            user_dir = f"user_data/{username}"
            if os.path.exists(user_dir):
                import shutil
                shutil.rmtree(user_dir)

            return True, f"✅ Utilisateur '{username}' supprimé avec succès"
        else:
            conn.close()
            return False, "❌ Utilisateur non trouvé"

    except Exception as e:
        return False, f"❌ Erreur lors de la suppression: {str(e)}"

def update_user_role(user_id, new_role):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('SELECT username, email FROM users WHERE id = ?', (user_id,))
        row = c.fetchone()

        if row:
            username, email = row
            if email in ADMINS_EMAILS:
                conn.close()
                return False, "❌ Impossible de modifier les rôles des comptes admin principaux"

            c.execute('UPDATE users SET role = ? WHERE id = ?', (new_role, user_id))
            conn.commit()
            conn.close()

            return True, f"✅ Rôle de '{username}' changé en '{new_role}'"
        else:
            conn.close()
            return False, "❌ Utilisateur non trouvé"

    except Exception as e:
        return False, f"❌ Erreur lors de la modification: {str(e)}"

def toggle_user_status(user_id):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('SELECT username, email, is_active FROM users WHERE id = ?', (user_id,))
        user_data = c.fetchone()

        if user_data:
            username, email, current_status = user_data
            if email in ADMINS_EMAILS:
                conn.close()
                return False, "❌ Impossible de désactiver les comptes admin principaux"

            new_status = 0 if current_status == 1 else 1
            status_text = "activé" if new_status == 1 else "désactivé"

            c.execute('UPDATE users SET is_active = ? WHERE id = ?', (new_status, user_id))
            conn.commit()
            conn.close()

            return True, f"✅ Utilisateur '{username}' {status_text}"
        else:
            conn.close()
            return False, "❌ Utilisateur non trouvé"

    except Exception as e:
        return False, f"❌ Erreur lors du changement de statut: {str(e)}"

def logout():
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.session_state.history = []
    st.session_state.audio_text = ""
    st.session_state.show_connection_info = False
    st.session_state.show_info_page = False
    st.session_state.show_edit_profile = False
    st.session_state.show_manage_users = False
    st.session_state.otp_stage = "enter_email"
    st.session_state.otp_email = ""
    st.session_state.otp_dev_code = None
    st.session_state.reset_stage = None
    st.session_state.reset_email = ""
    st.session_state.reset_token = ""
    st.rerun()

def reset_otp_flow():
    """Réinitialiser le flux OTP en cas de problème"""
    st.session_state.otp_stage = "enter_email"
    st.session_state.otp_email = ""
    st.session_state.otp_dev_code = None
    
    # Nettoyer les variables de formulaire
    for key in ['form_full_name', 'form_username', 'form_gender']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.rerun()

# ==================== PAGE DE LOGIN ====================
if not st.session_state.authenticated:
    check_reset_token_from_url()
    
    st.markdown("<h1 style='text-align:center; margin-bottom:0;'>Stress Detection Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#5c7fa8; margin-top:6px;'>Analyse intelligente du stress par IA</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    <style>
      @keyframes fadeSlideIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
      }
      .login-card {
        max-width: 720px;
        margin: 14px auto 0 auto;
        background: #ffffff;
        border: 1px solid rgba(47,128,237,0.14);
        border-radius: 18px;
        padding: 22px 22px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.08);
        animation: fadeSlideIn 420ms ease-out;
      }
      .seg {
        max-width: 720px;
        margin: 8px auto 16px auto;
        background: #f2f8ff;
        border: 1px solid #cfe3ff;
        padding: 10px 12px;
        border-radius: 14px;
        animation: fadeSlideIn 420ms ease-out;
      }
      .seg label { font-weight: 700; color: #2f80ed; }
      div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 12px !important;
      }
    </style>
    """, unsafe_allow_html=True)

    if "auth_view" not in st.session_state:
        st.session_state.auth_view = "login"
    if st.session_state.get("force_auth_view") in ("login", "otp"):
        st.session_state.auth_view = st.session_state.force_auth_view
        st.session_state.force_auth_view = None

    current_label = "🔑 Connectez-vous" if st.session_state.auth_view == "login" else "📩 Email OTP (1ère fois / reset)"
    if st.session_state.reset_stage:
        current_label = "🔑 Mot de passe oublié"
    
    st.markdown(f"""
      <div style="max-width:720px;margin:8px auto 14px auto;">
        <span style="display:inline-block;background:#f2f8ff;border:1px solid #cfe3ff;color:#2f80ed;
                     padding:8px 12px;border-radius:999px;font-weight:700;">
          {current_label}
        </span>
      </div>
    """, unsafe_allow_html=True)

    # Vue 1 : Connexion
    if st.session_state.auth_view == "login" and not st.session_state.reset_stage:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.subheader("Connexion")
        
       
        with st.form("login_email_pin"):
            email_login = st.text_input("📧 Email", placeholder="votre@email.com")
            pin_login = st.text_input("🔐 Code permanent (PIN / mot de passe)", type="password", placeholder="Votre code permanent")
            submit_login = st.form_submit_button("✅ Se connecter", type="primary", use_container_width=True)

        if submit_login:
            ok, user_info, msg = authenticate_email_pin(email_login.strip(), pin_login)
            if ok:
                st.session_state.authenticated = True
                st.session_state.current_user = user_info
                st.success(f"✅ Bienvenue {user_info['full_name']} !")
                st.rerun()
            else:
                st.error(f"❌ {msg}")
        
        col_forgot = st.columns([1, 2, 1])
        with col_forgot[1]:
            if st.button("🔑 Mot de passe oublié ?", use_container_width=True, type="secondary"):
                st.session_state.reset_stage = "request"
                st.rerun()
        
        st.markdown(
            "<div style='text-align:center;color:#6b6b6b;margin:6px 0 10px 0;'>"
            "Première fois ? Cliquez sur « Créer un compte » pour recevoir un OTP par email."
            "</div>",
            unsafe_allow_html=True
        )

        col_l, col_m, col_r = st.columns([1, 2, 1])
        with col_m:
            if st.button("Créer un compte", use_container_width=True):
                st.session_state.force_auth_view = "otp"
                st.session_state.otp_stage = "enter_email"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Vue 2 : OTP
    elif st.session_state.auth_view == "otp" and not st.session_state.reset_stage:
        st.subheader("Vérification par email (OTP)")
        st.caption("Flow: Email OTP (une fois) ➜ création d'un code permanent ➜ login normal.")

        if st.session_state.otp_stage == "enter_email":
            with st.form("otp_request_form"):
                otp_email = st.text_input("📧 Ton email", value=st.session_state.otp_email, placeholder="votre@email.com")
                send_btn = st.form_submit_button("📩 Envoyer le code OTP", type="primary", use_container_width=True)

            if send_btn:
                otp_email = (otp_email or "").strip()
                valid, msg = validate_email(otp_email)
                if not valid:
                    st.error(f"❌ {msg}")
                else:
                    otp_code = _generate_otp()
                    _save_otp(otp_email, otp_code, ttl_seconds=10*60)
                    sent, send_msg = send_otp_email(otp_email, otp_code)

                    st.session_state.otp_email = otp_email
                    st.session_state.otp_stage = "verify_otp"

                    if sent:
                        st.success("✅ Code OTP envoyé par email.")
                    else:
                        st.session_state.otp_dev_code = otp_code
                        st.warning(f"⚠️ {send_msg}")
                        st.info(f"🔎 Code OTP (mode dev) : **{otp_code}**")

                    st.rerun()

        elif st.session_state.otp_stage == "verify_otp":
            st.info(f"Email: **{st.session_state.otp_email}**")
            if (not EMAIL_CONFIG.get("EMAIL_ENABLED", False)) and st.session_state.otp_dev_code:
                st.info(f"🔎 Code OTP (mode dev) : **{st.session_state.otp_dev_code}**")

            st.markdown("""
            <style>
              @keyframes popIn {
                0%   { opacity: 0; transform: translateY(10px) scale(0.98); }
                100% { opacity: 1; transform: translateY(0) scale(1); }
              }
              .otp-card {
                max-width: 520px;
                margin: 18px auto 10px auto;
                background: #ffffff;
                padding: 28px 26px;
                border-radius: 18px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.10);
                text-align: center;
                animation: popIn 420ms ease-out;
              }
              .otp-icon {
                width: 72px; height: 72px;
                margin: 0 auto 16px auto;
                background: #e0f0ff;
                border-radius: 50%;
                display: flex; align-items: center; justify-content: center;
                font-size: 32px; color: #2f80ed;
              }
              .otp-title { color: #2f80ed; font-size: 22px; font-weight: 700; margin: 0 0 6px 0; }
              .otp-sub   { color: #666; font-size: 14px; margin: 0 0 18px 0; }
              .otp-row [data-testid="stTextInput"] input {
                height: 52px !important;
                text-align: center !important;
                font-size: 22px !important;
                border-radius: 12px !important;
                border: 1px solid #cfe3ff !important;
                background: #f7fbff !important;
              }
              .otp-row [data-testid="stTextInput"] input:focus {
                outline: none !important;
                border: 1px solid #2f80ed !important;
                box-shadow: 0 0 0 4px rgba(47,128,237,0.12) !important;
              }
            </style>

            <div class="otp-card">
              <div class="otp-icon">🔐</div>
              <div class="otp-title">Vérification de votre email</div>
              <div class="otp-sub">Entrez le code à 6 chiffres reçu par email</div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("otp_verify_form"):
                st.markdown('<div class="otp-row">', unsafe_allow_html=True)
                otp_cols = st.columns(6, gap="small")
                otp_digits = []
                for i, col in enumerate(otp_cols):
                    with col:
                        otp_digits.append(
                            st.text_input(
                                "",
                                max_chars=1,
                                key=f"otp_digit_{i}",
                                label_visibility="collapsed",
                            )
                        )
                st.markdown('</div>', unsafe_allow_html=True)

                otp_input = "".join([d.strip() for d in otp_digits])
                verify_btn = st.form_submit_button("✅ Vérifier", type="primary", use_container_width=True)

            components.html("""
            <script>
            (function() {
              const root = window.parent.document;

              function setNativeValue(el, value) {
                const valueSetter = Object.getOwnPropertyDescriptor(el.__proto__, 'value')?.set;
                const prototype = Object.getPrototypeOf(el);
                const prototypeValueSetter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set;
                const setter = prototypeValueSetter || valueSetter;
                if (setter) setter.call(el, value);
                else el.value = value;
                el.dispatchEvent(new Event('input', { bubbles: true }));
              }

              function getOtpInputs() {
                const row = root.querySelector('.otp-row');
                if (row) {
                  const inputs = Array.from(row.querySelectorAll('input[type="text"]'))
                    .filter(el => !el.disabled && el.offsetParent !== null);
                  if (inputs.length >= 6) return inputs.slice(0, 6);
                }

                const verifyBtns = Array.from(root.querySelectorAll('button'))
                  .filter(b => (b.innerText || '').trim() === '✅ Vérifier' && b.offsetParent !== null);
                if (!verifyBtns.length) return [];

                const verifyBtn = verifyBtns[verifyBtns.length - 1];
                const scope = verifyBtn.closest('section') || verifyBtn.closest('div') || root.body;
                const scoped = Array.from(scope.querySelectorAll('input[type="text"]'))
                  .filter(el => !el.disabled && el.offsetParent !== null);
                if (scoped.length >= 6) return scoped.slice(-6);

                return [];
              }

              function attachIfNeeded() {
                const inputs = getOtpInputs();
                if (inputs.length !== 6) return;

                inputs.forEach((inp, idx) => {
                  if (inp.dataset.otpAttached === "1") return;
                  inp.dataset.otpAttached = "1";

                  inp.setAttribute("inputmode", "numeric");
                  inp.setAttribute("pattern", "[0-9]*");
                  inp.autocomplete = "one-time-code";

                  inp.addEventListener("input", () => {
                    const digits = (inp.value || "").replace(/\D/g, "");
                    const one = digits.slice(0, 1);
                    if (inp.value !== one) setNativeValue(inp, one);

                    if (one.length === 1 && idx < inputs.length - 1) {
                      inputs[idx + 1].focus();
                      inputs[idx + 1].select();
                    }
                  });

                  inp.addEventListener("paste", (e) => {
                    const text = (e.clipboardData || window.clipboardData).getData('text') || '';
                    const digits = text.replace(/\D/g, '').slice(0, 6);
                    if (!digits) return;
                    e.preventDefault();
                    for (let i = 0; i < 6; i++) {
                      setNativeValue(inputs[i], digits[i] ? digits[i] : "");
                    }
                    const last = Math.min(digits.length, 6) - 1;
                    if (last >= 0) {
                      inputs[last].focus();
                      inputs[last].select();
                    }
                  });

                  inp.addEventListener("keydown", (e) => {
                    if (e.key === "Backspace" && (!inp.value || inp.value.length === 0) && idx > 0) {
                      inputs[idx - 1].focus();
                      inputs[idx - 1].select();
                    }
                    if (e.key === "ArrowLeft" && idx > 0) inputs[idx - 1].focus();
                    if (e.key === "ArrowRight" && idx < inputs.length - 1) inputs[idx + 1].focus();
                  });
                });
              }

              function boot() {
                attachIfNeeded();
                const obs = new MutationObserver(() => attachIfNeeded());
                obs.observe(root.body, { childList: true, subtree: true });
                let tries = 0;
                const t = setInterval(() => {
                  attachIfNeeded();
                  tries += 1;
                  if (tries > 24) clearInterval(t);
                }, 250);
              }

              boot();
            })();
            </script>
            """, height=0)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("↩️ Changer d'email", use_container_width=True):
                    st.session_state.otp_stage = "enter_email"
                    st.session_state.otp_dev_code = None
                    st.rerun()
            with col_b:
                if st.button("🔁 Renvoyer un OTP", use_container_width=True):
                    otp_code = _generate_otp()
                    _save_otp(st.session_state.otp_email, otp_code, ttl_seconds=10*60)
                    sent, send_msg = send_otp_email(st.session_state.otp_email, otp_code)
                    if sent:
                        st.success("✅ Nouveau code OTP envoyé.")
                    else:
                        st.session_state.otp_dev_code = otp_code
                        st.warning(f"⚠️ {send_msg}")
                        st.info(f"🔎 Code OTP (mode dev) : **{otp_code}**")
                    st.rerun()

            if verify_btn:
                if (not otp_input) or (len(otp_input) != 6) or (not otp_input.isdigit()):
                    st.error("❌ Veuillez saisir les 6 chiffres du code.")
                else:
                    ok, msg = _verify_otp(st.session_state.otp_email, (otp_input or "").strip())
                    if ok:
                        st.session_state.otp_stage = "set_pin"
                        st.success("✅ Vérification réussie.")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

        elif st.session_state.otp_stage == "set_pin":
            st.success("✅ Vérification réussie.")
            st.warning("🔒 **Sauvegarde ce code !** Ce sera ton code permanent pour te connecter les prochaines fois.")
            
            # Vérifier si l'utilisateur existe déjà
            existing = _get_user_by_email(st.session_state.otp_email)
            
            # Initialiser les valeurs par défaut dans session_state pour éviter les pertes
            if "form_full_name" not in st.session_state:
                st.session_state.form_full_name = existing[4] if existing else ""
            if "form_username" not in st.session_state:
                st.session_state.form_username = existing[1] if existing else ""
            if "form_gender" not in st.session_state:
                st.session_state.form_gender = existing[5] if existing and existing[5] else "Homme"
            
            # Bouton de réinitialisation
            col1, col2, col3 = st.columns(3)
            with col3:
                if st.button("↩️ Recommencer", use_container_width=True):
                    reset_otp_flow()
            
            with st.form("set_pin_form"):
                if not existing:
                    st.markdown("### Création de compte")
                    full_name = st.text_input(
                        "👤 Nom complet", 
                        value=st.session_state.form_full_name,
                        placeholder="Votre nom et prénom"
                    )
                    username = st.text_input(
                        "👥 Nom d'utilisateur", 
                        value=st.session_state.form_username,
                        placeholder="Choisissez un nom d'utilisateur (lettres et chiffres uniquement)"
                    )
                    # Validation en temps réel du nom d'utilisateur
                    if username and not username.isalnum():
                        st.warning("⚠️ Le nom d'utilisateur ne doit contenir que des lettres et chiffres")
                    if username and len(username) < 3:
                        st.warning("⚠️ Le nom d'utilisateur doit contenir au moins 3 caractères")
                else:
                    st.markdown("### Mise à jour du code permanent")
                    full_name = st.text_input(
                        "👤 Nom complet (optionnel)", 
                        value=existing[4] or ""
                    )
                    username = st.text_input(
                        "👥 Nom d'utilisateur (optionnel)", 
                        value=existing[1] or "",
                        disabled=True  # Désactiver la modification du username pour les comptes existants
                    )
                    st.caption("Le nom d'utilisateur ne peut pas être modifié")

                gender = st.radio(
                    "🚻 Sexe", 
                    ["Femme", "Homme"], 
                    horizontal=True,
                    index=0 if st.session_state.form_gender == "Femme" else 1
                )
                
                pin1 = st.text_input(
                    "🔐 Nouveau code permanent (PIN/mot de passe)", 
                    type="password",
                    placeholder="Minimum 4 caractères"
                )
                pin2 = st.text_input(
                    "🔁 Confirmer le code permanent", 
                    type="password",
                    placeholder="Confirmez votre code"
                )
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    save_pin_btn = st.form_submit_button(
                        "💾 Enregistrer", 
                        type="primary", 
                        use_container_width=True
                    )

            if save_pin_btn:
                # Validation complète
                errors = []
                
                # Validation des champs requis pour les nouveaux utilisateurs
                if not existing:
                    if not full_name or not full_name.strip():
                        errors.append("❌ Le nom complet est requis")
                    else:
                        valid, msg = validate_full_name(full_name)
                        if not valid:
                            errors.append(f"❌ {msg}")
                    
                    if not username or not username.strip():
                        errors.append("❌ Le nom d'utilisateur est requis")
                    else:
                        valid, msg = validate_username(username)
                        if not valid:
                            errors.append(f"❌ {msg}")
                
                # Validation du code permanent
                if not pin1:
                    errors.append("❌ Le code permanent est requis")
                elif len(pin1) < 4:
                    errors.append("❌ Le code permanent doit contenir au moins 4 caractères")
                elif pin1 != pin2:
                    errors.append("❌ Les codes ne correspondent pas")
                
                # Afficher les erreurs
                if errors:
                    for error in errors:
                        st.error(error)
                    
                    # Sauvegarder les valeurs pour ne pas les perdre
                    st.session_state.form_full_name = full_name
                    st.session_state.form_username = username
                    st.session_state.form_gender = gender
                else:
                    # Préparer les données pour l'enregistrement
                    with st.spinner("Enregistrement en cours..."):
                        ok, msg = upsert_user_with_pin(
                            email=st.session_state.otp_email,
                            pin=pin1,
                            full_name=full_name.strip() if full_name else None,
                            username=username.strip().lower() if username else None,  # Convertir en minuscules
                            gender=gender,
                        )
                        
                        if ok:
                            st.success("✅ Code permanent enregistré avec succès !")
                            
                            # Tentative de connexion automatique
                            ok2, user_info, msg2 = authenticate_email_pin(
                                st.session_state.otp_email, 
                                pin1
                            )
                            
                            if ok2:
                                st.session_state.authenticated = True
                                st.session_state.current_user = user_info
                                st.session_state.otp_stage = "enter_email"
                                st.session_state.otp_dev_code = None
                                
                                # Nettoyer les variables de formulaire
                                for key in ['form_full_name', 'form_username', 'form_gender']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                
                                st.success("✅ Connexion automatique réussie !")
                                st.rerun()
                            else:
                                st.info("Connectez-vous maintenant avec votre email et code permanent.")
                                st.session_state.otp_stage = "enter_email"
                                st.rerun()
                        else:
                            st.error(msg)
                            
                            # Sauvegarder les valeurs pour ne pas les perdre
                            st.session_state.form_full_name = full_name
                            st.session_state.form_username = username
                            st.session_state.form_gender = gender

    # Vue 3 : Réinitialisation de mot de passe
    elif st.session_state.reset_stage:
        st.subheader("🔑 Réinitialisation du mot de passe")
        
        if st.session_state.reset_stage == "request":
            st.info("Entrez votre email pour recevoir un lien de réinitialisation.")
            
            with st.form("reset_request_form"):
                reset_email = st.text_input("📧 Email", placeholder="votre@email.com")
                submit_request = st.form_submit_button("📩 Envoyer le lien de réinitialisation", type="primary", use_container_width=True)
            
            if submit_request:
                reset_email = reset_email.strip()
                valid, msg = validate_email(reset_email)
                if not valid:
                    st.error(f"❌ {msg}")
                else:
                    user = _get_user_by_email(reset_email)
                    if not user:
                        st.error("❌ Aucun compte trouvé avec cet email.")
                    else:
                        token = create_reset_token(reset_email)
                        sent, send_msg = send_password_reset_email(reset_email, token)
                        
                        if sent:
                            st.session_state.reset_email = reset_email
                            st.session_state.reset_stage = "verify"
                            st.success("✅ Email envoyé ! Vérifiez votre boîte de réception.")
                            st.rerun()
                        else:
                            st.error(f"❌ {send_msg}")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("↩️ Retour à la connexion", use_container_width=True):
                    st.session_state.reset_stage = None
                    st.rerun()
        
        elif st.session_state.reset_stage == "verify":
            st.info(f"Un lien de réinitialisation a été envoyé à **{st.session_state.reset_email}**")
            st.info("Cliquez sur le lien dans l'email pour continuer.")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("↩️ Retour à la connexion", use_container_width=True):
                    st.session_state.reset_stage = None
                    st.rerun()
            with col_b:
                if st.button("🔄 Renvoyer le lien", use_container_width=True):
                    token = create_reset_token(st.session_state.reset_email)
                    sent, send_msg = send_password_reset_email(st.session_state.reset_email, token)
                    if sent:
                        st.success("✅ Nouveau lien envoyé !")
                    else:
                        st.error(f"❌ {send_msg}")
        
        elif st.session_state.reset_stage == "reset":
            st.success("✅ Lien de réinitialisation valide")
            st.info("Choisissez un nouveau code permanent.")
            
            with st.form("reset_password_form"):
                new_pin1 = st.text_input("🔐 Nouveau code permanent", type="password", placeholder="Votre nouveau code")
                new_pin2 = st.text_input("🔁 Confirmer le code permanent", type="password", placeholder="Confirmez le nouveau code")
                submit_reset = st.form_submit_button("💾 Enregistrer le nouveau code", type="primary", use_container_width=True)
            
            if submit_reset:
                if new_pin1 != new_pin2:
                    st.error("❌ Les codes ne correspondent pas.")
                else:
                    pin_valid, pin_msg = validate_pin(new_pin1)
                    if not pin_valid:
                        st.error(f"❌ {pin_msg}")
                    else:
                        conn = sqlite3.connect('users.db')
                        c = conn.cursor()
                        
                        try:
                            c.execute('UPDATE users SET pin_hash = ? WHERE email = ?', 
                                     (hash_password(new_pin1), st.session_state.reset_email))
                            conn.commit()
                            
                            delete_reset_token(st.session_state.reset_email)
                            
                            st.success("✅ Code permanent mis à jour avec succès !")
                            st.info("Vous pouvez maintenant vous connecter avec votre nouveau code.")
                            
                            st.session_state.reset_stage = None
                            st.session_state.reset_email = ""
                            st.session_state.reset_token = ""
                            
                            time.sleep(3)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la mise à jour: {str(e)}")
                        finally:
                            conn.close()

    if st.button("ℹ️ Afficher les informations", type="secondary", use_container_width=True):
        st.session_state.show_info_page = not st.session_state.show_info_page
        st.rerun()

    if st.session_state.show_info_page:
        with st.expander("📋 **Informations du système**", expanded=True):
            st.subheader("ℹ️ À propos de cette application")
            st.markdown("""
            **Stress Detection Dashboard**

            **Fonctionnalités :**
            - 📝 Analyse de texte multilingue
            - 🎤 Reconnaissance vocale avancée
            - 📷 Analyse faciale avec base de données Happy/Sad
            - 📊 Visualisations interactives
            - 💡 Recommandations personnalisées

            **Développé par :** Basma & Aya
            """)

    with st.expander("ℹ️ **Comment ça marche ?**", expanded=False):
        st.info("""
        **Étapes :**
        1. **Connexion (normal)** : Email + code permanent
        2. **Première connexion** : Email → OTP → création du code permanent
        3. **Mot de passe oublié** : Email → lien de réinitialisation → nouveau code
        4. **Analyse faciale** : Compare votre visage avec les images Happy/Sad
        """)

    st.markdown("---")
    st.markdown("© 2024 Stress Detection Dashboard • Version 4.0 • Développé par Basma & Aya")

    st.stop()

# ==================== PAGE PRINCIPALE (APRÈS LOGIN) ====================
col_user1, col_user2, col_user3 = st.columns([3, 1, 1])
with col_user1:
    if st.session_state.current_user["role"] == "admin":
        st.markdown(f"### 👑 Administrateur : {st.session_state.current_user['full_name']}")
        st.caption(f"Compte créé le : {st.session_state.current_user['created_at']}")
    else:
        st.markdown(f"### 👋 Bienvenue, {st.session_state.current_user['full_name']}!")
        st.caption(f"Utilisateur depuis : {st.session_state.current_user['created_at']}")

with col_user3:
    if st.button("🚪 Déconnexion", type="secondary", use_container_width=True):
        logout()

st.title("🧠 Stress Detection Dashboard")
st.markdown("### Analyse intelligente du stress par IA")

# ==================== SIDEBAR ====================
with st.sidebar:
    if st.session_state.current_user["role"] == "admin":
        if st.session_state.current_user.get("email") in ADMINS_EMAILS:
            st.success("👑 **MODE ADMINISTRATEUR**")
        else:
            st.warning("⚠️ **ADMIN**")
    
    st.header(f"👤 Profil de {st.session_state.current_user['full_name']}")
    
    st.markdown(f"**📧 Email :** {st.session_state.current_user['email']}")
    st.markdown(f"**🚻 Sexe :** {st.session_state.current_user.get('gender', '—') or '—'}")
    st.markdown(f"**👥 Nom d'utilisateur :** `{st.session_state.current_user['username']}`")
    st.markdown(f"**🎯 Rôle :** {st.session_state.current_user['role'].capitalize()}")
    
    # Menu principal
    menu_options = []
    
    if not st.session_state.show_edit_profile and not st.session_state.show_manage_users:
        menu_options.append("📊 Tableau de bord")
    
    menu_options.append("⚙️ Modifier mon profil")
    
    if st.session_state.current_user["role"] == "admin":
        menu_options.append("👥 Gérer les utilisateurs")
    
    selected_menu = st.selectbox(
        "Navigation",
        menu_options,
        key="main_menu"
    )
    
    if selected_menu == "⚙️ Modifier mon profil":
        if not st.session_state.show_edit_profile:
            st.session_state.show_edit_profile = True
            st.session_state.show_manage_users = False
            st.rerun()
    
    elif selected_menu == "👥 Gérer les utilisateurs":
        if not st.session_state.show_manage_users:
            st.session_state.show_manage_users = True
            st.session_state.show_edit_profile = False
            st.rerun()
    
    elif selected_menu == "📊 Tableau de bord":
        if st.session_state.show_edit_profile or st.session_state.show_manage_users:
            st.session_state.show_edit_profile = False
            st.session_state.show_manage_users = False
            st.rerun()
    
    st.divider()
    
    if not st.session_state.show_edit_profile and not st.session_state.show_manage_users:
        st.header("⚙️ Paramètres")
        
        recording_duration = st.slider(
            "Durée d'enregistrement (secondes)",
            min_value=3,
            max_value=15,
            value=8
        )
        
        energy_threshold = st.slider(
            "Sensibilité au bruit (voix)",
            min_value=100,
            max_value=5000,
            value=300
        )
        
        st.divider()
        
        st.header("📊 Mes statistiques")
        total_analyses = len(st.session_state.history)
        st.metric("Analyses totales", total_analyses)
        
        if total_analyses > 0:
            avg_stress = sum(h["stress"] for h in st.session_state.history) / total_analyses
            st.metric("Stress moyen", f"{avg_stress:.1f}%")
            
            last_analysis = st.session_state.history[-1]
            st.metric("Dernier score", f"{last_analysis['stress']}%")
        
        st.divider()
        
        st.header("💡 Conseils")
        st.info("""
        **Pour une meilleure reconnaissance :**
        - 🗣️ Parlez clairement
        - 🤫 Évitez le bruit
        - 📷 Bon éclairage pour la caméra
            
        **Pour une meilleure analyse faciale :**
        - 📷 Regardez directement la caméra
        - 💡 Assurez-vous d'un bon éclairage
        - 😀 Gardez une expression naturelle
        - 🚫 Évitez les mouvements brusques
        """)
        
        if st.button("🗑️ Effacer mon historique", type="secondary", use_container_width=True):
            st.session_state.history = []
            st.success("✅ Historique effacé !")
            st.rerun()

# ==================== PAGE DE MODIFICATION DE PROFIL ====================
if st.session_state.show_edit_profile:
    st.title("⚙️ Modifier mon profil")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.info("**👤 Informations actuelles**")
        st.write(f"**Nom complet :** {st.session_state.current_user['full_name']}")
        st.write(f"**Email :** {st.session_state.current_user['email']}")
        st.write(f"**Sexe :** {st.session_state.current_user.get('gender', '—') or '—'}")
        st.write(f"**Nom d'utilisateur :** {st.session_state.current_user['username']}")
        st.write(f"**Rôle :** {st.session_state.current_user['role'].capitalize()}")

    with col2:
        st.info("**✏️ Modifier les informations**")

        with st.form("edit_profile_form"):
            new_full_name = st.text_input(
                "Nouveau nom complet",
                value=st.session_state.current_user['full_name'] or "",
                placeholder="Votre nom et prénom"
            )

            new_email = st.text_input(
                "Nouvel email",
                value=st.session_state.current_user['email'] or "",
                placeholder="votre@email.com"
            )

            current_gender = st.session_state.current_user.get("gender") or "Femme"
            gender = st.radio("🚻 Sexe", ["Femme", "Homme"], horizontal=True,
                              index=0 if current_gender == "Femme" else 1)

            st.markdown("---")

            st.subheader("🔐 Changer le code permanent (PIN)")
            st.caption("Laissez vide si vous ne voulez pas changer votre code permanent.")
            current_pin = st.text_input("Code permanent actuel", type="password", placeholder="Votre code actuel")
            new_pin = st.text_input("Nouveau code permanent", type="password", placeholder="Votre nouveau code")
            confirm_new_pin = st.text_input("Confirmer le nouveau code", type="password", placeholder="Retapez le nouveau code")

            st.markdown("---")

            st.subheader("🔑 Changer le mot de passe (legacy)")
            st.caption("Optionnel.")
            current_password = st.text_input("Mot de passe actuel", type="password", placeholder="Mot de passe actuel")
            new_password = st.text_input("Nouveau mot de passe", type="password", placeholder="Nouveau mot de passe")
            confirm_new_password = st.text_input("Confirmer le nouveau mot de passe", type="password", placeholder="Retapez le nouveau mot de passe")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                save_button = st.form_submit_button("💾 Enregistrer", type="primary", use_container_width=True)
            with col_btn2:
                cancel_button = st.form_submit_button("❌ Annuler", use_container_width=True)

    if save_button:
        if new_pin or confirm_new_pin or current_pin:
            if not new_pin:
                st.error("❌ Entrez le nouveau code permanent.")
            elif new_pin != confirm_new_pin:
                st.error("❌ Les nouveaux codes permanents ne correspondent pas.")
            elif not current_pin:
                st.error("❌ Entrez le code permanent actuel pour le changer.")
            else:
                success, message = update_user_profile(
                    user_id=st.session_state.current_user['id'],
                    full_name=new_full_name,
                    email=new_email,
                    gender=gender,
                    current_pin=current_pin,
                    new_pin=new_pin
                )
                if success:
                    st.success(message)
                    st.session_state.show_edit_profile = False
                    st.rerun()
                else:
                    st.error(message)
        elif new_password or confirm_new_password or current_password:
            if not new_password:
                st.error("❌ Entrez le nouveau mot de passe.")
            elif new_password != confirm_new_password:
                st.error("❌ Les nouveaux mots de passe ne correspondent pas.")
            elif not current_password:
                st.error("❌ Entrez le mot de passe actuel pour le changer.")
            else:
                success, message = update_user_profile(
                    user_id=st.session_state.current_user['id'],
                    full_name=new_full_name,
                    email=new_email,
                    gender=gender,
                    current_password=current_password,
                    new_password=new_password
                )
                if success:
                    st.success(message)
                    st.session_state.show_edit_profile = False
                    st.rerun()
                else:
                    st.error(message)
        else:
            success, message = update_user_profile(
                user_id=st.session_state.current_user['id'],
                full_name=new_full_name,
                email=new_email,
                gender=gender
            )
            if success:
                st.success(message)
                st.session_state.show_edit_profile = False
                st.rerun()
            else:
                st.error(message)

    if cancel_button:
        st.session_state.show_edit_profile = False
        st.rerun()

    if st.button("⬅️ Retour au tableau de bord", type="secondary"):
        st.session_state.show_edit_profile = False
        st.rerun()

    st.stop()

# ==================== PAGE DE GESTION DES UTILISATEURS (ADMIN) ====================
if st.session_state.show_manage_users and st.session_state.current_user["role"] == "admin":
    st.title("👥 Gestion des utilisateurs")
    st.markdown("---")
    
    if st.session_state.current_user.get("email") not in ADMINS_EMAILS:
        st.error("❌ Accès refusé. Seuls Basma et Aya peuvent gérer les utilisateurs.")
        if st.button("⬅️ Retour"):
            st.session_state.show_manage_users = False
            st.rerun()
        st.stop()
    
    users = get_all_users()
    
    if not users:
        st.info("📭 Aucun utilisateur trouvé")
    else:
        total_users = len(users)
        active_users = sum(1 for u in users if u[6] == 1)
        admin_users = sum(1 for u in users if u[4] == 'admin')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Total utilisateurs", total_users)
        with col2:
            st.metric("✅ Actifs", active_users)
        with col3:
            st.metric("👑 Admins", admin_users)
        
        st.divider()
        
        for user in users:
            user_id, username, full_name, email, gender, role, created_at, is_active = user
            
            with st.expander(f"{'👑' if role == 'admin' else '👤'} **{username}** - {full_name}", expanded=False):
                col_info, col_actions = st.columns([2, 1])
                
                with col_info:
                    st.write(f"**📧 Email :** {email}")
                    st.write(f"**🚻 Sexe :** {gender or '—'}")
                    st.write(f"**🎯 Rôle :** {role}")
                    st.write(f"**📅 Créé le :** {created_at}")
                    status = "✅ Actif" if is_active == 1 else "❌ Inactif"
                    st.write(f"**🔧 Statut :** {status}")
                
                with col_actions:
                    if username not in ["basma", "aya"]:
                        st.write("**Actions :**")
                        
                        new_role = st.selectbox(
                            "Rôle",
                            ["user", "admin"],
                            index=0 if role == "user" else 1,
                            key=f"role_{user_id}"
                        )
                        
                        if new_role != role:
                            if st.button("🔄 Changer rôle", key=f"changer_role_{user_id}"):
                                success, message = update_user_role(user_id, new_role)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                        
                        status_action = "🚫 Désactiver" if is_active == 1 else "✅ Activer"
                        if st.button(status_action, key=f"toggle_{user_id}"):
                            success, message = toggle_user_status(user_id)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                        
                        if st.button("🗑️ Supprimer", type="secondary", key=f"delete_{user_id}"):
                            with st.spinner("Suppression en cours..."):
                                success, message = delete_user(user_id)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                    else:
                        st.info("🔒 Compte protégé")
    
    if st.button("⬅️ Retour au tableau de bord", type="secondary"):
        st.session_state.show_manage_users = False
        st.rerun()
    
    st.stop()

# ==================== MAIN CONTENT (TABLEAU DE BORD) ====================
st.markdown("### Analysez votre niveau de stress à partir de texte, voix ou visage")

mode = st.radio(
    "**Sélectionnez le mode d'entrée :**",
    ("Texte ✍️", "Voix 🎤", "Visage 📷"),
    horizontal=True
)

text = ""

# ==================== MODE TEXTE ====================
if mode == "Texte ✍️":
    st.markdown("#### Entrez votre texte ici :")
    text = st.text_area(
        "",
        placeholder="Ex: 'Je me sens stressé ces derniers temps'...",
        height=150,
        label_visibility="collapsed"
    )
    
    if st.button("📤 Analyser le texte"):
        if not text.strip():
            st.warning("Veuillez entrer un texte à analyser.")
            st.stop()

# ==================== MODE VOIX ====================
elif mode == "Voix 🎤":
    st.markdown("#### Configuration de la reconnaissance vocale")
    
    voice_lang = st.selectbox(
        "**Sélectionnez la langue parlée :**",
        ["Darija 🇲🇦", "Français 🇫🇷", "Arabe 🇸🇦", "Anglais 🇬🇧","Mixte"],
        help="Sélectionnez la langue dans laquelle vous allez parler"
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        record_button = st.button(
            "🎤 Commencer l'enregistrement",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Effacer", use_container_width=True):
            st.session_state.audio_text = ""
            st.rerun()
    
    if record_button:
        with st.spinner(f"🎤 Enregistrement en cours... ({recording_duration}s)"):
            r = sr.Recognizer()
            r.energy_threshold = energy_threshold
            
            try:
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=2)
                    audio = r.listen(source, timeout=10, phrase_time_limit=recording_duration)
                
                lang_map_voice = {
                    "Darija 🇲🇦": "ar-MA",
                    "Français 🇫🇷": "fr-FR",
                    "Arabe 🇸🇦": "ar-SA",
                    "Anglais 🇬🇧": "en-US"
                }
                
                if voice_lang == "Mixte":
                    text, detected_lang = recognize_mixed_speech(r, audio)
                    if text:
                        st.session_state.audio_text = text
                        st.success(f"✅ Transcription réussie ! (Langue détectée: {detected_lang})")
                    else:
                        st.error("❌ Impossible de reconnaître la parole")
                else:
                    text = r.recognize_google(audio, language=lang_map_voice[voice_lang])
                    st.session_state.audio_text = text
                    st.success(f"✅ Transcription réussie !")
                    
            except sr.UnknownValueError:
                st.error("❌ Impossible de reconnaître la parole")
            except sr.RequestError as e:
                st.error(f"❌ Erreur du service: {e}")
            except Exception as e:
                st.error(f"❌ Erreur d'enregistrement: {e}")
    
    if st.session_state.audio_text:
        st.markdown("#### 📝 Texte transcrit :")
        st.info(f"**{st.session_state.audio_text}**")
        text = st.session_state.audio_text
        
        if st.button("📊 Analyser le texte transcrit", type="primary"):
            if not text.strip():
                st.warning("Le texte transcrit est vide.")
                st.stop()
    else:
        st.info("🎤 Appuyez sur 'Commencer l'enregistrement' pour parler")

# ==================== MODE VISAGE ====================
elif mode == "Visage 📷":
    st.markdown("#### 📷 Détection du stress par comparaison d'émotions")
    
    st.info("""
    **Comment ça marche :**
    - 🤗 **Images Happy** → Stress bas
    - 😔 **Images Sad** → Stress élevé
    - Le système compare votre visage avec la base de données d'images
    """)
    
    # Afficher les statistiques des dossiers
    HAPPY_PATH = r"C:\Users\pc\Downloads\archive (5)\Data\Happy"
    SAD_PATH = r"C:\Users\pc\Downloads\archive (5)\Data\Sad"
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            if os.path.exists(HAPPY_PATH):
                n_happy = len([f for f in os.listdir(HAPPY_PATH) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                st.metric("😊 Images Happy", n_happy)
            else:
                st.metric("😊 Images Happy", "❌ Non trouvé")
        except:
            st.metric("😊 Images Happy", "❌ Erreur")
    
    with col2:
        try:
            if os.path.exists(SAD_PATH):
                n_sad = len([f for f in os.listdir(SAD_PATH) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                st.metric("😢 Images Sad", n_sad)
            else:
                st.metric("😢 Images Sad", "❌ Non trouvé")
        except:
            st.metric("😢 Images Sad", "❌ Erreur")
    
    st.markdown("---")
    
    if st.button("📷 Démarrer l'analyse avec base de données", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            score, level = detect_stress_from_face(duration=recording_duration)
        
        # Sauvegarder dans l'historique
        st.session_state.history.append({
            "time": datetime.datetime.now(),
            "stress": score,
            "text": f"Analyse faciale avec base de données (Happy/Sad)",
            "lang": "face_db",
            "level": level,
            "user": st.session_state.current_user["username"]
        })
        
        st.success("✅ Analyse terminée !")
        
        # Résultat final
        st.subheader("📊 Résultat de l'analyse")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric("Score de stress", f"{score:.1f}%")
        
        with col_res2:
            level_text = {
                "low": "Faible 😊", 
                "medium": "Modéré 😐", 
                "high": "Élevé 😟"
            }.get(level, level)
            st.metric("Niveau", level_text)
        
        with col_res3:
            if level == "low":
                st.success("✅ Correspond aux images HAPPY")
            elif level == "high":
                st.error("⚠️ Correspond aux images SAD")
            else:
                st.warning("⚖️ Émotion neutre")

# ==================== ANALYSE DU TEXTE ====================
if text.strip() != "" and (mode == "Texte ✍️" or (mode == "Voix 🎤" and text)):
    
    with st.expander("📄 Texte à analyser", expanded=True):
        st.write(text)
    
    try:
        with st.spinner("🔄 Analyse en cours..."):
            response = requests.post(
                API_URL, 
                json={"text": text},
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            
            score = data.get("score", 0)
            level = data.get("level", "unknown")
            lang = data.get("lang", "unknown")
            suggestion = data.get("suggestion", "")
            
            display_lang = lang_map.get(lang, lang.capitalize())
            
            st.session_state.history.append({
                "time": datetime.datetime.now(),
                "stress": score,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "lang": lang,
                "level": level,
                "user": st.session_state.current_user["username"]
            })
            
            # Visualisations
            st.subheader("📊 Niveau de Stress")
            
            if level == "low":
                bar_color = "green"
                level_text = "Faible"
            elif level == "medium":
                bar_color = "orange"
                level_text = "Modéré"
            else:
                bar_color = "red"
                level_text = "Élevé"
            
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={
                    "font": {"size": 48, "color": bar_color},
                    "suffix": "%"
                },
                title={
                    "text": f"Stress : {level_text}",
                    "font": {"size": 20}
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "darkgray"
                    },
                    "bar": {"color": bar_color, "thickness": 0.8},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 34], "color": "lightgreen"},
                        {"range": [34, 67], "color": "lightyellow"},
                        {"range": [67, 100], "color": "lightcoral"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": score
                    }
                }
            ))
            
            gauge_fig.update_layout(
                height=300,
                margin=dict(t=50, b=10, l=10, r=10)
            )
            
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🌍 Langue", display_lang)
            
            with col2:
                st.metric("📈 Score", f"{score}%")
            
            with col3:
                st.metric("📊 Niveau", level_text)
            
            if len(st.session_state.history) > 1:
                st.subheader("📈 Évolution Historique")
                
                times = [h["time"] for h in st.session_state.history]
                stresses = [h["stress"] for h in st.session_state.history]
                texts = [h["text"] for h in st.session_state.history]
                
                line_fig = go.Figure()
                
                line_fig.add_trace(go.Scatter(
                    x=times,
                    y=stresses,
                    mode='lines+markers',
                    name='Stress Level',
                    line=dict(color='#FF4B4B', width=3),
                    marker=dict(size=10, color='white', line=dict(width=2, color='#FF4B4B')),
                    text=texts,
                    hovertemplate='<b>%{text}</b><br>Stress: %{y}%<br>Time: %{x}<extra></extra>'
                ))
                
                line_fig.add_hrect(
                    y0=0, y1=34,
                    fillcolor="green", opacity=0.1,
                    layer="below", line_width=0
                )
                
                line_fig.add_hrect(
                    y0=34, y1=67,
                    fillcolor="yellow", opacity=0.1,
                    layer="below", line_width=0
                )
                
                line_fig.add_hrect(
                    y0=67, y1=100,
                    fillcolor="red", opacity=0.1,
                    layer="below", line_width=0
                )
                
                line_fig.update_layout(
                    height=400,
                    xaxis_title="Date et Heure",
                    yaxis_title="Stress (%)",
                    yaxis=dict(range=[0, 100]),
                    hovermode="x unified",
                    plot_bgcolor='rgba(0,0,0,0.05)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(line_fig, use_container_width=True)
            
            st.subheader("💡 Recommandations Personnalisées")
            
            if suggestion:
                st.info(suggestion)
            else:
                if level == "low":
                    st.info("**Pour maintenir votre bien-être :**\n- Continuez vos activités relaxantes\n- Pratiquez la gratitude quotidienne\n- Maintenez un équilibre travail/vie personnelle")
                elif level == "medium":
                    st.info("**Pour réduire votre stress :**\n- Pratiquez 10 minutes de méditation par jour\n- Faites une promenade en pleine nature\n- Écoutez de la musique relaxante\n- Limitez votre consommation de caféine")
                else:
                    st.info("**Pour gérer le stress élevé :**\n- Consultez un professionnel de santé\n- Pratiquez des exercices de respiration profonde\n- Établissez une routine de sommeil régulière\n- Parlez de vos préoccupations à un proche")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Nouvelle Analyse", type="primary", use_container_width=True):
                    st.session_state.audio_text = ""
                    st.rerun()
            
        else:
            st.error(f"❌ Erreur du serveur : Code {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ **Impossible de se connecter au serveur.**")
        st.info("Vérifiez que le serveur Flask est en cours d'exécution : `python app.py`")
    except requests.exceptions.Timeout:
        st.error("⏱️ Délai d'attente dépassé.")
    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")

# ==================== FOOTER ====================
st.divider()
st.markdown("---")
st.markdown(f"Stress Detection Dashboard v4.0 • Connecté en tant que : **{st.session_state.current_user['full_name']}** • Administrateurs : Basma & Aya")
# Dans la sidebar ou le footer
st.markdown("---")
st.markdown("[🔒 Politique de confidentialité](https://ton-site.netlify.app/privacy.html)")
with st.expander("📦 Instructions d'installation"):
    st.markdown("""
    **Pour exécuter cette application :**
    
    1. **Installer les dépendances :**
    ```bash
    pip install streamlit requests plotly speechrecognition
    pip install opencv-python numpy scikit-learn
    pip install pyaudio
    ```
    
    2. **Démarrer le serveur Flask :**
    ```bash
    python app.py
    ```
    
    3. **Démarrer le dashboard :**
    ```bash
    streamlit run dashboard.py
    ```
    
    4. **Ouvrir votre navigateur à :** `http://localhost:8501`
    """)