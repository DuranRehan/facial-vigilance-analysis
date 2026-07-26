from datetime import datetime

class CheckInRecordBuilder:
   
    @staticmethod
    def build_record(features: dict, mood: int) -> dict:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "timestamp": timestamp,
            "mood": mood
        }
        record.update(features)
        return record
        