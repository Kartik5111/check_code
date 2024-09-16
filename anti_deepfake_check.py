import cv2
import numpy as np
from cryptography.fernet import Fernet
import os

class AntiDeepfakeVideoProcessor:
    def __init__(self, key):
        self.key = key
        self.fernet = Fernet(key)

    def load_user_photo(self, image_path):

        user_image = cv2.imread(image_path)
        if user_image is None:
            raise ValueError("Image not found or invalid format")
        return user_image

    def resize_watermark(self, watermark, frame_shape):
        
        height, width = frame_shape[:2]
        return cv2.resize(watermark, (width, height))

    def embed_watermark(self, frame, watermark):
        
        return cv2.addWeighted(frame, 1, watermark, 0.3, 0)  

    def add_noise(self, frame):
        
        noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
        return cv2.add(frame, noise)

    def encrypt_metadata(self, metadata):
        
        return self.fernet.encrypt(metadata.encode())

    def process_video(self, input_path, output_path, metadata, user_photo_path=None):
    
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if user_photo_path:
            
            watermark = self.load_user_photo(user_photo_path)
            watermark = self.resize_watermark(watermark, (height, width))
        else:
        
            watermark = np.zeros((height, width, 3), np.uint8)
            cv2.putText(watermark, "Protected", (width//2 - 100, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        encrypted_metadata = self.encrypt_metadata(metadata)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break


            frame = self.embed_watermark(frame, watermark)
            frame = self.add_noise(frame)

            
            cv2.imshow("Watermarked Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break

            
            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()


key = Fernet.generate_key()
processor = AntiDeepfakeVideoProcessor(key)


processor.process_video("video.mp4/IMG-20240915-WA0008.jpg", "output.mp4", 
                        "Original video by John Doe", 
                        user_photo_path="video.mp4/IMG-20240915-WA0008.jpg")
