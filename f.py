import cv2
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import dropbox

class AdvancedTransferData:
    def __init__(self, access_token):
        self.access_token = access_token

    def process_frame_with_tensorflow(self, frame, model, color_threshold=50):
        frame = cv2.resize(frame, (640, 480))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pixels = frame_rgb.reshape((-1, 3))
        kmeans = KMeans(n_clusters=1).fit(pixels)
        dominant_color = kmeans.cluster_centers_[0]

        lower_bound = np.array(dominant_color - color_threshold, dtype=np.uint8)
        upper_bound = np.array(dominant_color + color_threshold, dtype=np.uint8)

        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)
        result = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)

        difference = frame_rgb - result
        processed_frame = np.where(mask[:, :, np.newaxis] != 0, difference, frame_rgb)

        return cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

    def upload_file_to_dropbox(self, source_file_path, destination_file_path):
        dbx = dropbox.Dropbox(self.access_token)

        with open(source_file_path, 'rb') as file:
            dbx.files_upload(file.read(), destination_file_path)

    def main(self):
        # GUI to select background image
        root = tk.Tk()
        root.withdraw()
        source_path = filedialog.askopenfilename(title="Select File to Transfer")

        video_capture = cv2.VideoCapture(0)

        while True:
            ret, current_frame = video_capture.read()

            if not ret:
                break

            processed_frame = self.process_frame_with_tensorflow(current_frame, model=None)

            cv2.imshow("Video", current_frame)
            cv2.imshow("Processed Frame", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

        # Uploading the processed frame to Dropbox
        destination_path = input("Enter the full path to upload to Dropbox: ")
        self.upload_file_to_dropbox(source_path, destination_path)
        print("File has been processed and uploaded to Dropbox!")

if __name__ == "__main__":
    access_token = 'sl.A3SzaxDevBfFDENlvQynfnFAzhUe9axnUeI6rD6IaAr5SXW1dfGKn6VezA8H1h6qtdSwcVQJm1CjfEGHpEyEPy5x3c08lhDtwr09oqr7YuEaO_oaEnaXYVFXg9fy1TTjbFiaHpU'
    data_transfer = AdvancedTransferData(access_token)
    data_transfer.main()
