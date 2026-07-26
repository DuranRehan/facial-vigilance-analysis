import os
import csv


class CSVWriter:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def append(self, record: dict):
            fieldnames = list(record.keys())
            
            file_exists = os.path.exists(self.file_path)
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            with open(self.file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()  
                    
                writer.writerow(record)