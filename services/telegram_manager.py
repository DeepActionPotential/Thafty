import requests
from dataclasses import dataclass
import numpy as np
from PIL import Image
import tempfile


from schemas.services_schemas import NoFaceMatchingAlarm, FoundBodyNoFaceAlarm, FaceMatchingAlarm
from utils.utils import ndarray_2_pilimage


class TelegramNotificationManager:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
    
    def send_telegram_photo(self, image: Image.Image, caption: str = "") -> dict:
        """Helper method to send photos through Telegram API"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array and then to BGR (OpenCV format)
            import numpy as np
            img_array = np.array(image)
            
            # Convert RGB to BGR
            img_bgr = img_array[..., ::-1]
            
            # Convert back to PIL Image
            from PIL import Image as PILImage
            img_bgr_pil = PILImage.fromarray(img_bgr)
            
            # Save as JPEG with maximum quality
            img_bgr_pil.save(temp_file, format="JPEG", quality=100, subsampling=0, optimize=True)
            temp_file.seek(0)  # Rewind the file
            
            files = {"photo": temp_file}
            data = {"chat_id": self.chat_id, "caption": caption}
            
            try:
                response = requests.post(url, files=files, data=data, timeout=10)
                response.raise_for_status()  # Raise an exception for bad status codes
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error sending photo to Telegram: {e}")
                return {"ok": False, "error": str(e)}

    def send_telegram_message(self, text: str) -> dict:
        """Send plain text message"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, data=payload)
        return response.json()

    def send_alarm(self, alarm_data) -> None:
        """Handle different alarm types and send appropriate notifications"""
        if isinstance(alarm_data, NoFaceMatchingAlarm):
            self.handle_no_face_matching(alarm_data)
        elif isinstance(alarm_data, FoundBodyNoFaceAlarm):
            self.handle_body_no_face(alarm_data)
        elif isinstance(alarm_data, FaceMatchingAlarm):
            self.handle_face_matching(alarm_data)
        else:
            return

    def send_alarms(self, alarms_data) -> None:
        """Send multiple alarms"""
        for alarm_data in alarms_data:
            self.send_alarm(alarm_data)
    

    def handle_face_matching(self, alarm: FaceMatchingAlarm) -> None:
        """Process FaceMatchingAlarm"""
        # Convert numpy arrays to PIL Images
        known_img = ndarray_2_pilimage(alarm.known_face)
        matched_img = ndarray_2_pilimage(alarm.extracted_face)
        
        # Send notification with images
        message = f"✅ Someone is present, but no need to worry — they're recognized from the database."

        self.send_telegram_message(message)
        
        # Send known face
        self.send_telegram_photo(known_img, "Matched Face")
        
        # Send matched face
        self.send_telegram_photo(matched_img, "Extracted Face")

    def handle_no_face_matching(self, alarm: NoFaceMatchingAlarm) -> None:
        """Process NoFaceMatchingAlarm"""
        # Convert numpy arrays to PIL Images
        unmatched_img = ndarray_2_pilimage(alarm.extracted_face)
        
        # Send notification with images
        message = f"⚠️ Unrecognized face detected — no match in the system."
        
        # Send unmatched face
        self.send_telegram_photo(unmatched_img, message + "Unmatched Face Detected")
        
      

    def handle_body_no_face(self, alarm: FoundBodyNoFaceAlarm) -> None:
        """Process FoundBodyNoFaceAlarm"""
        annotated_img = ndarray_2_pilimage(alarm.annotated_image)
        message = "⚠️ Human body detected — facial features not visible."
        self.send_telegram_photo(annotated_img, message)



