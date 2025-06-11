from dataclasses import dataclass


@dataclass
class DefaultCfg:

    face_model_path: str = "./models/det_10g.onnx"
    face_threshold_probability: float = 0.5
    face_nms: float = 0.5
    
    person_model_path: str = "./models/yolov8n.pt"
    person_confidence_threshold: float = 0.5

    embedding_model_path: str = "./models/efficientnet_b0.pth"
    embedding_device: str = "cpu"

    deepface_models_path: str = './models'
    deepface_model_name: str = 'VGG-Face'
    deepface_dimension: int = 4096


    matching_top_k: int = 2
    faiss_manager_top_k: int = 2
    found_coverd_body_alarm_wait: int = 3


    frame_analysis_interval_wait_in_seconds = 5
    known_people_info_path: str = './data/images'


    telegram_bot_token: str = "your_telegram_token"
    notification_chat_id: str = 'the id for that telegram user that will be notiified when-there is unusual even' # example: '113xxxxxxx'


