import cv2
import numpy as np
import time
import threading
from queue import Queue
from deepface import DeepFace
import tensorflow as tf
import logging
from collections import deque


class FaceAnalyzerApp:
    """
    FaceAnalyzerApp - Real-time face analysis application with DeepFace and OpenCV

    This application uses DeepFace for face analysis and OpenCV for face detection.
    It provides enhanced visualization and performance optimizations for real-time
    face analysis applications.

    Usage:
        app = FaceAnalyzerApp()
        app.run()
    """

    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize camera with optimal settings
        self._initializeCamera()

        # Threading and processing controls
        self.frameQueue = Queue(maxsize=4)
        self.resultQueue = Queue(maxsize=4)
        self.isProcessing = False
        self.isRunning = True

        # Load OpenCV's face detector as fallback
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Analysis and display parameters
        self.lastResults = None
        self.frameTimes = deque(maxlen=30)  # Using deque for better performance

        # Result smoothing
        self.smoothingWindow = 5
        self.ageBuffer = deque(maxlen=self.smoothingWindow)
        self.emotionBuffer = deque(maxlen=self.smoothingWindow)

        # Face tracking
        self.tracker = cv2.legacy.TrackerKCF_create()  # tracker creation with legacy
        self.trackingBox = None
        self.trackingSuccess = False
        self.detectionInterval = 15  # Frames between full detections
        self.frameCount = 0

        # Configure GPU
        self._configureGPU()

        # Start analysis thread
        self.analysisThread = threading.Thread(target=self._analysisWorker)
        self.analysisThread.daemon = True  # Ensure thread terminates with main program
        self.analysisThread.start()

    def _initializeCamera(self):
        """Initialize camera with optimal settings"""
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Hi-Res for better analysis
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")

    def _configureGPU(self):
        """Configure GPU settings for optimal performance"""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(
                    f"GPU configuration successful: {len(gpus)} GPUs found"
                )
            else:
                self.logger.warning("No GPUs found, using CPU")
        except Exception as e:
            self.logger.error(f"GPU configuration failed: {str(e)}")

    def _smoothResults(self, results):
        """Smooth analysis results to reduce jitter"""
        self.ageBuffer.append(results["age"])
        results["age"] = sum(self.ageBuffer) / len(self.ageBuffer)

        # Smooth emotions
        if not self.emotionBuffer:
            self.emotionBuffer.append(results["emotion"])
        else:
            smoothed_emotions = {}
            for emotion in results["emotion"].keys():
                values = [buf[emotion] for buf in self.emotionBuffer]
                smoothed_emotions[emotion] = sum(values) / len(values)
            results["emotion"] = smoothed_emotions

        self.emotionBuffer.append(results["emotion"])
        return results

    def _analysisWorker(self):
        while self.isRunning:
            try:
                if not self.isProcessing and not self.frameQueue.empty():
                    self.isProcessing = True
                    frame = self.frameQueue.get()

                    # Try with OpenCV detector first
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        face_frame = frame[y : y + h, x : x + w]
                        results = DeepFace.analyze(
                            img_path=cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB),
                            actions=["age", "gender", "emotion", "race"],
                            enforce_detection=False,
                            detector_backend="opencv",
                            align=True,
                            silent=True,
                        )

                        if isinstance(results, list) and len(results) > 0:
                            result = results[0]  # Get the first result in the list
                            result["region"] = {"x": x, "y": y, "w": w, "h": h}
                            smoothed_results = self._smoothResults(result)
                            if self.resultQueue.full():
                                self.resultQueue.get()
                            self.resultQueue.put(smoothed_results)

                    self.isProcessing = False

            except Exception as e:
                self.logger.error(f"Analysis error: {str(e)}")
                self.isProcessing = False

            time.sleep(0.01)

    def _drawResults(self, frame, results):
        """Draw analysis results with enhanced visualization"""
        if "region" in results:
            x, y, w, h = results["region"].values()

            # Draw face rectangle with gradient
            gradient = np.linspace(0, 1, 4)
            for t, alpha in zip(range(4), gradient):
                cv2.rectangle(
                    frame, (x - t, y - t), (x + w + t, y + h + t), (0, 255, 0), 1
                )

        # Create formatted info strings
        info = [
            f"Age: {results['age']:.1f} years",
            f"Gender: {max(results['gender'], key=results['gender'].get)}",
            f"Emotion: {results['dominant_emotion']}",
            f"Race: {results['dominant_race']}",
        ]

        # Draw emotion bars
        emotions = results["emotion"]
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        bar_width = 150
        for i, (emotion, value) in enumerate(sorted_emotions[:5]):
            bar_pos = (10, 160 + i * 25)
            cv2.rectangle(
                frame,
                bar_pos,
                (bar_pos[0] + int(bar_width * value / 100), bar_pos[1] + 20),
                (120, 255, 120),
                -1,
            )
            self._drawTextWithBackground(
                frame,
                f"{emotion}: {value:.1f}%",
                (bar_pos[0] + bar_width + 10, bar_pos[1] + 15),
            )

        # Draw main info
        for i, text in enumerate(info):
            self._drawTextWithBackground(frame, text, (10, 30 + i * 30))

    def _drawTextWithBackground(self, frame, text, position):
        """Enhanced text rendering with better visibility"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        padding = 5

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )

        # Draw background with alpha blending
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (position[0] - padding, position[1] - text_height - padding),
            (position[0] + text_width + padding, position[1] + padding),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text
        cv2.putText(frame, text, position, font, scale, (255, 255, 255), thickness)

    def run(self):
        """Main application loop with enhanced error handling"""
        try:
            while True:
                startTime = time.time()

                ret, frame = self.camera.read()
                if not ret:
                    self.logger.error("Failed to read frame from camera")
                    break

                # Update face tracking
                if self.trackingSuccess:
                    self.trackingSuccess, self.trackingBox = self.tracker.update(frame)

                # Queue frame for analysis
                if not self.isProcessing and self.frameQueue.empty():
                    if self.frameCount % self.detectionInterval == 0:
                        self.frameQueue.put(frame.copy())

                # Get and display results
                if not self.resultQueue.empty():
                    self.lastResults = self.resultQueue.get()
                    if "region" in self.lastResults:
                        bbox = self.lastResults["region"]
                        self.tracker = (
                            cv2.legacy.TrackerKCF_create()
                        )  # Re-initialize tracker
                        self.tracker.init(
                            frame, (bbox["x"], bbox["y"], bbox["w"], bbox["h"])
                        )
                        self.trackingSuccess = True

                if self.lastResults:
                    self._drawResults(frame, self.lastResults)

                self._updateFPS(frame, startTime)

                cv2.imshow("Face Analysis", frame)
                self.frameCount += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as e:
            self.logger.error(f"Application error: {str(e)}")
        finally:
            self._cleanup()

    def _updateFPS(self, frame, startTime):
        """Display FPS with better accuracy"""
        elapsedTime = time.time() - startTime
        fps = 1 / elapsedTime
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    def _cleanup(self):
        """Release resources gracefully"""
        self.camera.release()
        cv2.destroyAllWindows()
        self.isRunning = False


if __name__ == "__main__":
    app = FaceAnalyzerApp()
    app.run()
