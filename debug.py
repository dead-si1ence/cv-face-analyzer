import cv2
from deepface import DeepFace
import numpy as np
from datetime import datetime
import time
import tensorflow as tf


def real_time_analysis():
    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Analysis frequency control
    analysis_interval = 1  # seconds
    last_analysis_time = time.time()
    last_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # Perform analysis every [analysis_interval] seconds
        if current_time - last_analysis_time >= analysis_interval:
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["age", "gender", "emotion", "race"],
                    enforce_detection=False,
                )
                last_results = results[0]
                last_analysis_time = current_time
            except Exception as e:
                print(f"Analysis error: {str(e)}")

        # Display results if available
        if last_results:
            # Draw rectangle around face
            if "region" in last_results:
                x, y, w, h, *_ = last_results["region"].values()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display analysis results
            info = [
                f"Age: {last_results['age']:.0f}",
                f"Gender: {max(last_results['gender'], key=last_results['gender'].get)}",
                f"Dominant emotion: {last_results['dominant_emotion']}",
                f"Dominant race: {last_results['dominant_race']}",
            ]

            # Add text to frame
            for i, text in enumerate(info):
                cv2.putText(
                    frame,
                    text,
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Display frame
        cv2.imshow("Face Analysis", frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_analysis()
