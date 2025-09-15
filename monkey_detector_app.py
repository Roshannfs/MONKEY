#!/usr/bin/env python3
"""
AI-Powered Monkey Detection System - Fixed Detection Logic
Prevents counting the same monkey multiple times
"""

import sys
import cv2
import numpy as np
import serial
import time
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                           QWidget, QPushButton, QLabel, QTextEdit, QSlider,
                           QGroupBox, QGridLayout, QComboBox, QSpinBox,
                           QFileDialog, QMessageBox, QFrame)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage, QFont

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

class FixedMonkeyDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🐒 AI Monkey Detection System - Professional Edition")
        self.setGeometry(100, 100, 1000, 700)

        # Core variables
        self.model = None
        self.arduino = None
        self.camera = None
        self.detection_active = False
        self.arduino_connected = False
        self.confidence_threshold = 0.5
        self.detection_count = 0

        # IMPROVED Detection state tracking
        self.monkey_present = False
        self.detection_start_time = None
        self.no_detection_frames = 0
        self.min_gap_frames = 30  # 30 frames (~3 seconds) gap before counting new detection

        # Setup GUI
        self.setup_ui()
        self.setup_styling()

        # Status timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(2000)

        self.log_message("🚀 Monkey Detection System Initialized")

    def setup_ui(self):
        """Setup user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header
        self.create_header(main_layout)

        # Main content area
        content_layout = QHBoxLayout()

        # Left side - Camera and detection
        self.create_camera_section(content_layout)

        # Right side - Controls and status
        self.create_controls_section(content_layout)

        main_layout.addLayout(content_layout)

        # Footer
        self.create_footer(main_layout)

    def create_header(self, parent_layout):
        """Create header"""
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #1e1e1e; border-radius: 8px; padding: 10px;")
        header_layout = QHBoxLayout(header_frame)

        title_label = QLabel("🐒 AI-Powered Monkey Detection System")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #4CAF50;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self.system_status_label = QLabel("System Ready")
        self.system_status_label.setStyleSheet("font-size: 12px; color: #888;")
        header_layout.addWidget(self.system_status_label)

        parent_layout.addWidget(header_frame)

    def create_camera_section(self, parent_layout):
        """Create camera display and controls"""
        left_panel = QVBoxLayout()

        # Camera display
        camera_group = QGroupBox("📹 Live Detection Feed")
        camera_layout = QVBoxLayout(camera_group)

        self.camera_label = QLabel("Click 'Start Camera' to begin")
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("""
            border: 2px solid #444; 
            background-color: #000; 
            color: #fff;
            font-size: 14px;
        """)
        self.camera_label.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.camera_label)

        # Camera controls
        camera_controls = QHBoxLayout()

        self.start_camera_btn = QPushButton("📹 Start Camera")
        self.start_detection_btn = QPushButton("🚀 Start Detection")
        self.stop_btn = QPushButton("⏹ Stop All")

        self.start_camera_btn.clicked.connect(self.start_camera)
        self.start_detection_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_all)

        camera_controls.addWidget(self.start_camera_btn)
        camera_controls.addWidget(self.start_detection_btn)
        camera_controls.addWidget(self.stop_btn)

        camera_layout.addLayout(camera_controls)
        left_panel.addWidget(camera_group)

        parent_layout.addLayout(left_panel, 2)

    def create_controls_section(self, parent_layout):
        """Create controls"""
        right_panel = QVBoxLayout()

        # Model selection
        model_group = QGroupBox("🧠 AI Model")
        model_layout = QVBoxLayout(model_group)

        self.load_model_btn = QPushButton("📂 Load Trained Model")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)

        self.model_status_label = QLabel("❌ No model loaded")
        model_layout.addWidget(self.model_status_label)

        right_panel.addWidget(model_group)

        # Arduino controls
        arduino_group = QGroupBox("🔌 Arduino Alert System")
        arduino_layout = QGridLayout(arduino_group)

        arduino_layout.addWidget(QLabel("COM Port:"), 0, 0)
        self.com_port_combo = QComboBox()
        self.com_port_combo.addItems(["COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8"])
        self.com_port_combo.setCurrentText("COM3")
        arduino_layout.addWidget(self.com_port_combo, 0, 1)

        arduino_buttons = QHBoxLayout()
        self.connect_arduino_btn = QPushButton("🔗 Connect")
        self.test_buzzer_btn = QPushButton("🔔 Test")

        self.connect_arduino_btn.clicked.connect(self.connect_arduino)
        self.test_buzzer_btn.clicked.connect(self.test_buzzer)

        arduino_buttons.addWidget(self.connect_arduino_btn)
        arduino_buttons.addWidget(self.test_buzzer_btn)
        arduino_layout.addLayout(arduino_buttons, 1, 0, 1, 2)

        self.arduino_status_label = QLabel("❌ Arduino: Not connected")
        arduino_layout.addWidget(self.arduino_status_label, 2, 0, 1, 2)

        right_panel.addWidget(arduino_group)

        # Detection settings
        settings_group = QGroupBox("⚙️ Detection Settings")
        settings_layout = QGridLayout(settings_group)

        settings_layout.addWidget(QLabel("Confidence:"), 0, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 95)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        settings_layout.addWidget(self.confidence_slider, 0, 1)

        self.confidence_label = QLabel("0.50")
        settings_layout.addWidget(self.confidence_label, 0, 2)

        right_panel.addWidget(settings_group)

        # System status
        status_group = QGroupBox("📊 Detection Status")
        status_layout = QVBoxLayout(status_group)

        self.camera_status_label = QLabel("❌ Camera: Disconnected")
        self.detection_count_label = QLabel("Unique Detections: 0")
        self.current_status_label = QLabel("Status: No Detection")
        self.last_detection_label = QLabel("Last Alert: Never")

        status_layout.addWidget(self.camera_status_label)
        status_layout.addWidget(self.detection_count_label)
        status_layout.addWidget(self.current_status_label)
        status_layout.addWidget(self.last_detection_label)

        right_panel.addWidget(status_group)

        # Activity log
        log_group = QGroupBox("📝 Activity Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        self.clear_log_btn = QPushButton("🗑 Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(self.clear_log_btn)

        right_panel.addWidget(log_group)

        parent_layout.addLayout(right_panel, 1)

    def create_footer(self, parent_layout):
        """Create footer"""
        footer_frame = QFrame()
        footer_frame.setStyleSheet("background-color: #1e1e1e; border-radius: 8px; padding: 8px;")
        footer_layout = QHBoxLayout(footer_frame)

        status_label = QLabel("AI-Powered Detection System - Ready for Operation")
        status_label.setStyleSheet("color: #888; font-size: 10px;")
        footer_layout.addWidget(status_label)

        footer_layout.addStretch()

        version_label = QLabel("v3.0 - Fixed Detection Logic")
        version_label.setStyleSheet("color: #888; font-size: 10px;")
        footer_layout.addWidget(version_label)

        parent_layout.addWidget(footer_frame)

    def setup_styling(self):
        """Apply styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }

            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 15px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }

            QPushButton:hover {
                background-color: #45a049;
            }

            QPushButton:pressed {
                background-color: #3d8b40;
            }

            QComboBox, QSpinBox {
                padding: 8px;
                border: 1px solid #555;
                border-radius: 4px;
                background-color: #3c3c3c;
                color: #ffffff;
                font-size: 11px;
            }

            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                font-size: 11px;
            }

            QSlider::groove:horizontal {
                border: 1px solid #999;
                height: 6px;
                background: #666;
                border-radius: 3px;
            }

            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #4CAF50;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }

            QLabel {
                font-size: 11px;
            }
        """)

    # Core functionality
    def update_confidence(self, value):
        """Update confidence threshold"""
        self.confidence_threshold = value / 100.0
        self.confidence_label.setText(f"{self.confidence_threshold:.2f}")

    def start_camera(self):
        """Start camera feed"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.show_error("Failed to open camera")
                return

            self.camera_thread = CameraThread(self.camera)
            self.camera_thread.frame_ready.connect(self.update_camera_display)
            self.camera_thread.start()

            self.camera_status_label.setText("✅ Camera: Connected")
            self.log_message("Camera started successfully")

        except Exception as e:
            self.show_error(f"Camera error: {str(e)}")

    def start_detection(self):
        """Start AI detection"""
        if not self.model:
            self.show_error("Please load an AI model first!")
            return

        if not self.camera or not self.camera.isOpened():
            self.show_error("Please start camera first!")
            return

        self.detection_active = True
        # Reset detection state
        self.monkey_present = False
        self.detection_start_time = None
        self.no_detection_frames = 0

        self.detection_thread = ProperDetectionThread(
            self.camera, self.model, self.confidence_threshold, self
        )
        self.detection_thread.detection_result.connect(self.handle_detection)
        self.detection_thread.start()

        self.log_message("🚀 AI detection started with improved logic")
        self.system_status_label.setText("Detection Active")

    def stop_all(self):
        """Stop all operations"""
        self.detection_active = False
        self.monkey_present = False
        self.no_detection_frames = 0

        # Stop detection thread
        if hasattr(self, 'detection_thread'):
            self.detection_thread.stop()
            self.detection_thread.wait()

        # Stop camera
        if hasattr(self, 'camera_thread'):
            self.camera_thread.stop()
            self.camera_thread.wait()

        if self.camera:
            self.camera.release()
            self.camera = None

        # Turn off Arduino buzzer
        if self.arduino_connected and self.arduino:
            try:
                self.arduino.write(b'STOP_ALERT\n')
            except:
                pass

        self.camera_status_label.setText("❌ Camera: Disconnected")
        self.current_status_label.setText("Status: Stopped")
        self.camera_label.setText("Click 'Start Camera' to begin")
        self.camera_label.setStyleSheet("border: 2px solid #444; background-color: #000; color: #fff; font-size: 14px;")
        self.system_status_label.setText("System Ready")
        self.log_message("All operations stopped")

    def load_model(self):
        """Load trained AI model"""
        if not YOLO_AVAILABLE:
            self.show_error("ultralytics package not installed!")
            return

        try:
            # Try to load the trained model first
            trained_model_path = "runs/detect/monkey_detector_v1/weights/best.pt"
            if Path(trained_model_path).exists():
                self.model = YOLO(trained_model_path)
                self.model_status_label.setText("✅ Custom trained model loaded (88.4% mAP)")
                self.log_message("Custom trained model loaded successfully")
                return

            # Otherwise, let user select model
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load YOLO Model", "", "Model files (*.pt);;All files (*.*)"
            )

            if file_path:
                self.model = YOLO(file_path)
                self.model_status_label.setText("✅ AI Model: Loaded")
                self.log_message(f"Model loaded: {Path(file_path).name}")

        except Exception as e:
            self.show_error(f"Failed to load model: {str(e)}")

    def connect_arduino(self):
        """Connect to Arduino"""
        try:
            com_port = self.com_port_combo.currentText()
            self.arduino = serial.Serial(com_port, 9600, timeout=2)
            time.sleep(2)

            # Test connection
            self.arduino.write(b'STATUS\n')
            time.sleep(0.5)

            self.arduino_connected = True
            self.arduino_status_label.setText(f"✅ Arduino: Connected ({com_port})")
            self.log_message(f"Arduino connected on {com_port}")

        except Exception as e:
            self.show_error(f"Arduino connection failed: {str(e)}")

    def test_buzzer(self):
        """Test Arduino buzzer"""
        if self.arduino_connected and self.arduino:
            try:
                self.arduino.write(b'TEST\n')
                self.log_message("🔔 Buzzer test sent to Arduino")
            except Exception as e:
                self.show_error(f"Buzzer test failed: {str(e)}")
        else:
            self.show_error("Arduino not connected")

    def handle_detection(self, detected, confidence, frame):
        """IMPROVED detection handling - prevents counting same monkey multiple times"""

        if detected:
            # Reset no-detection counter
            self.no_detection_frames = 0

            if not self.monkey_present:
                # NEW monkey detected - count it and trigger alert
                self.monkey_present = True
                self.detection_count += 1
                self.detection_start_time = time.time()

                # Update UI for NEW detection
                self.detection_count_label.setText(f"Unique Detections: {self.detection_count}")
                self.current_status_label.setText(f"🐒 NEW MONKEY DETECTED! ({confidence:.2f})")
                self.last_detection_label.setText(f"Last Alert: {datetime.now().strftime('%H:%M:%S')}")

                # Arduino alert for NEW detection only
                if self.arduino_connected and self.arduino:
                    try:
                        self.arduino.write(b'MONKEY_DETECTED\n')
                    except:
                        pass

                # Log NEW detection
                self.log_message(f"🐒 NEW MONKEY DETECTED! Count: {self.detection_count}, Confidence: {confidence:.2f}")
            else:
                # SAME monkey still present - just update confidence, no new count
                self.current_status_label.setText(f"🐒 Tracking Same Monkey ({confidence:.2f})")

        else:
            # No monkey detected in this frame
            self.no_detection_frames += 1

            if self.monkey_present and self.no_detection_frames >= self.min_gap_frames:
                # Monkey has been gone long enough - reset state
                self.monkey_present = False
                self.current_status_label.setText("Status: Monitoring...")

                # Turn off Arduino buzzer
                if self.arduino_connected and self.arduino:
                    try:
                        self.arduino.write(b'STOP_ALERT\n')
                    except:
                        pass

                self.log_message("Monkey left area - Ready for new detection")
            elif not self.monkey_present:
                # No monkey present, just monitoring
                self.current_status_label.setText("Status: Monitoring...")

    def update_camera_display(self, frame):
        """Update camera display"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)

    def update_status(self):
        """Update system status"""
        pass

    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)

        # Keep log manageable
        if self.log_text.document().lineCount() > 50:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()

    def clear_log(self):
        """Clear log"""
        self.log_text.clear()

    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        self.log_message(f"ERROR: {message}")

    def closeEvent(self, event):
        """Clean shutdown"""
        self.stop_all()
        if self.arduino:
            self.arduino.close()
        event.accept()

# PROPER detection thread with improved logic
class ProperDetectionThread(QThread):
    detection_result = pyqtSignal(bool, float, np.ndarray)

    def __init__(self, camera, model, confidence_threshold, parent):
        super().__init__()
        self.camera = camera
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.parent = parent
        self.running = False

    def run(self):
        self.running = True
        while self.running and self.parent.detection_active:
            ret, frame = self.camera.read()
            if ret:
                try:
                    results = self.model(frame, verbose=False)

                    monkey_detected = False
                    max_confidence = 0

                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                confidence = box.conf.item()
                                if confidence > self.confidence_threshold:
                                    monkey_detected = True
                                    max_confidence = max(max_confidence, confidence)

                                    # Draw bounding box
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                                    # Color based on confidence
                                    if confidence > 0.8:
                                        color = (0, 255, 0)  # Green
                                    elif confidence > 0.6:
                                        color = (0, 255, 255)  # Yellow
                                    else:
                                        color = (0, 165, 255)  # Orange

                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(frame, f'Monkey: {confidence:.2f}', 
                                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.6, color, 2)

                    # Update display
                    self.parent.update_camera_display(frame)

                    # Emit detection result
                    self.detection_result.emit(monkey_detected, max_confidence, frame)

                except Exception as e:
                    print(f"Detection error: {e}")

            self.msleep(100)  # ~10 FPS

    def stop(self):
        self.running = False

# Simple camera thread
class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                self.frame_ready.emit(frame)
            self.msleep(33)  # ~30 FPS

    def stop(self):
        self.running = False

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI Monkey Detection System")

    window = FixedMonkeyDetectorGUI()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
