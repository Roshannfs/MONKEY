
"""
AI-Powered Monkey Detection System with Arduino Integration
Complete system with PyQt5 GUI, live webcam detection, and hardware alerts
"""

import sys
import cv2
import numpy as np
import serial
import time
import threading
from pathlib import Path
import json
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QPushButton, QLabel, QTextEdit, QSlider, 
                           QGroupBox, QGridLayout, QComboBox, QSpinBox,
                           QProgressBar, QTabWidget, QFileDialog, QMessageBox,
                           QFrame, QCheckBox, QLineEdit)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

class MonkeyDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Powered Monkey Detection System v2.0")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize variables
        self.model = None
        self.arduino = None
        self.camera = None
        self.detection_thread = None
        self.detection_active = False
        self.arduino_connected = False

        # Detection settings
        self.confidence_threshold = 0.5
        self.detection_count = 0
        self.last_detection_time = 0
        self.alert_cooldown = 5  # seconds

        # Setup GUI
        self.setup_ui()
        self.setup_styling()

        # Setup timers
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)

        print("🚀 Monkey Detection System Initialized")

    def setup_ui(self):
        """Setup the main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create tab widget
        self.tabs = QTabWidget()
        central_widget.setLayout(QVBoxLayout())
        central_widget.layout().addWidget(self.tabs)

        # Create tabs
        self.create_detection_tab()
        self.create_training_tab()
        self.create_settings_tab()
        self.create_logs_tab()

    def create_detection_tab(self):
        """Create the main detection interface"""
        detection_tab = QWidget()
        layout = QHBoxLayout(detection_tab)

        # Left panel - Camera feed
        left_panel = QVBoxLayout()

        # Camera display
        camera_group = QGroupBox("🔴 Live Camera Feed")
        camera_layout = QVBoxLayout(camera_group)

        self.camera_label = QLabel("Camera feed will appear here")
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.camera_label)

        # Camera controls
        camera_controls = QHBoxLayout()
        self.start_camera_btn = QPushButton("📹 Start Camera")
        self.stop_camera_btn = QPushButton("⏹ Stop Camera")
        self.capture_btn = QPushButton("📸 Capture")

        self.start_camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.capture_btn.clicked.connect(self.capture_frame)

        camera_controls.addWidget(self.start_camera_btn)
        camera_controls.addWidget(self.stop_camera_btn)
        camera_controls.addWidget(self.capture_btn)
        camera_layout.addLayout(camera_controls)

        left_panel.addWidget(camera_group)

        # Detection controls
        detection_group = QGroupBox("🧠 AI Detection Controls")
        detection_layout = QGridLayout(detection_group)

        # Model selection
        detection_layout.addWidget(QLabel("AI Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["None", "Load Custom Model...", "Train New Model"])
        detection_layout.addWidget(self.model_combo, 0, 1)

        # Confidence threshold
        detection_layout.addWidget(QLabel("Confidence:"), 1, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 95)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        detection_layout.addWidget(self.confidence_slider, 1, 1)

        self.confidence_label = QLabel("0.50")
        detection_layout.addWidget(self.confidence_label, 1, 2)

        # Detection buttons
        detection_buttons = QHBoxLayout()
        self.start_detection_btn = QPushButton("🚀 Start Detection")
        self.stop_detection_btn = QPushButton("⏹ Stop Detection")

        self.start_detection_btn.clicked.connect(self.start_detection)
        self.stop_detection_btn.clicked.connect(self.stop_detection)

        detection_buttons.addWidget(self.start_detection_btn)
        detection_buttons.addWidget(self.stop_detection_btn)
        detection_layout.addLayout(detection_buttons, 2, 0, 1, 3)

        left_panel.addWidget(detection_group)

        # Right panel - Status and controls
        right_panel = QVBoxLayout()

        # System status
        status_group = QGroupBox("📊 System Status")
        status_layout = QGridLayout(status_group)

        self.camera_status_label = QLabel("❌ Camera: Disconnected")
        self.model_status_label = QLabel("❌ AI Model: Not loaded")
        self.arduino_status_label = QLabel("❌ Arduino: Disconnected")

        status_layout.addWidget(self.camera_status_label, 0, 0)
        status_layout.addWidget(self.model_status_label, 1, 0)
        status_layout.addWidget(self.arduino_status_label, 2, 0)

        right_panel.addWidget(status_group)

        # Detection statistics
        stats_group = QGroupBox("📈 Detection Statistics")
        stats_layout = QGridLayout(stats_group)

        self.detection_count_label = QLabel("Detections: 0")
        self.last_detection_label = QLabel("Last Detection: Never")
        self.fps_label = QLabel("FPS: 0")

        stats_layout.addWidget(self.detection_count_label, 0, 0)
        stats_layout.addWidget(self.last_detection_label, 1, 0)
        stats_layout.addWidget(self.fps_label, 2, 0)

        right_panel.addWidget(stats_group)

        # Arduino controls
        arduino_group = QGroupBox("🔌 Arduino Controls")
        arduino_layout = QGridLayout(arduino_group)

        arduino_layout.addWidget(QLabel("COM Port:"), 0, 0)
        self.com_port_combo = QComboBox()
        self.com_port_combo.addItems(["COM1", "COM2", "COM3", "COM4", "COM5"])
        arduino_layout.addWidget(self.com_port_combo, 0, 1)

        arduino_buttons = QHBoxLayout()
        self.connect_arduino_btn = QPushButton("🔗 Connect")
        self.test_buzzer_btn = QPushButton("🔔 Test Buzzer")

        self.connect_arduino_btn.clicked.connect(self.connect_arduino)
        self.test_buzzer_btn.clicked.connect(self.test_buzzer)

        arduino_buttons.addWidget(self.connect_arduino_btn)
        arduino_buttons.addWidget(self.test_buzzer_btn)
        arduino_layout.addLayout(arduino_buttons, 1, 0, 1, 2)

        # Alert settings
        arduino_layout.addWidget(QLabel("Alert Cooldown (s):"), 2, 0)
        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(1, 60)
        self.cooldown_spin.setValue(5)
        self.cooldown_spin.valueChanged.connect(self.update_cooldown)
        arduino_layout.addWidget(self.cooldown_spin, 2, 1)

        right_panel.addWidget(arduino_group)

        # Detection log
        log_group = QGroupBox("📝 Detection Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        log_buttons = QHBoxLayout()
        self.clear_log_btn = QPushButton("🗑 Clear Log")
        self.save_log_btn = QPushButton("💾 Save Log")

        self.clear_log_btn.clicked.connect(self.clear_log)
        self.save_log_btn.clicked.connect(self.save_log)

        log_buttons.addWidget(self.clear_log_btn)
        log_buttons.addWidget(self.save_log_btn)
        log_layout.addLayout(log_buttons)

        right_panel.addWidget(log_group)

        # Add panels to main layout
        layout.addLayout(left_panel, 2)  # 2/3 width
        layout.addLayout(right_panel, 1)  # 1/3 width

        self.tabs.addTab(detection_tab, "🔍 Detection")

    def create_training_tab(self):
        """Create the AI model training interface"""
        training_tab = QWidget()
        layout = QVBoxLayout(training_tab)

        # Dataset section
        dataset_group = QGroupBox("📁 Dataset Management")
        dataset_layout = QGridLayout(dataset_group)

        dataset_layout.addWidget(QLabel("Dataset Folder:"), 0, 0)
        self.dataset_path_edit = QLineEdit("monkey_dataset")
        dataset_layout.addWidget(self.dataset_path_edit, 0, 1)

        self.browse_dataset_btn = QPushButton("📂 Browse")
        self.browse_dataset_btn.clicked.connect(self.browse_dataset)
        dataset_layout.addWidget(self.browse_dataset_btn, 0, 2)

        dataset_buttons = QHBoxLayout()
        self.create_dataset_btn = QPushButton("📋 Create Dataset Structure")
        self.collect_images_btn = QPushButton("📸 Collect Images")
        self.annotate_images_btn = QPushButton("🏷 Annotate Images")

        self.create_dataset_btn.clicked.connect(self.create_dataset_structure)
        self.collect_images_btn.clicked.connect(self.collect_images)
        self.annotate_images_btn.clicked.connect(self.annotate_images)

        dataset_buttons.addWidget(self.create_dataset_btn)
        dataset_buttons.addWidget(self.collect_images_btn)
        dataset_buttons.addWidget(self.annotate_images_btn)
        dataset_layout.addLayout(dataset_buttons, 1, 0, 1, 3)

        layout.addWidget(dataset_group)

        # Training section
        training_group = QGroupBox("🧠 Model Training")
        training_layout = QGridLayout(training_group)

        # Training parameters
        training_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(100)
        training_layout.addWidget(self.epochs_spin, 0, 1)

        training_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(16)
        training_layout.addWidget(self.batch_spin, 0, 3)

        training_layout.addWidget(QLabel("Image Size:"), 1, 0)
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["416", "640", "832", "1024"])
        self.imgsz_combo.setCurrentText("640")
        training_layout.addWidget(self.imgsz_combo, 1, 1)

        # Training controls
        training_buttons = QHBoxLayout()
        self.start_training_btn = QPushButton("🚀 Start Training")
        self.stop_training_btn = QPushButton("⏹ Stop Training")
        self.load_model_btn = QPushButton("📂 Load Model")

        self.start_training_btn.clicked.connect(self.start_training)
        self.stop_training_btn.clicked.connect(self.stop_training)
        self.load_model_btn.clicked.connect(self.load_model)

        training_buttons.addWidget(self.start_training_btn)
        training_buttons.addWidget(self.stop_training_btn)
        training_buttons.addWidget(self.load_model_btn)
        training_layout.addLayout(training_buttons, 2, 0, 1, 4)

        # Training progress
        self.training_progress = QProgressBar()
        training_layout.addWidget(self.training_progress, 3, 0, 1, 4)

        # Training log
        self.training_log = QTextEdit()
        self.training_log.setMaximumHeight(200)
        self.training_log.setReadOnly(True)
        training_layout.addWidget(self.training_log, 4, 0, 1, 4)

        layout.addWidget(training_group)

        self.tabs.addTab(training_tab, "🧠 Training")

    def create_settings_tab(self):
        """Create the settings interface"""
        settings_tab = QWidget()
        layout = QVBoxLayout(settings_tab)

        # Camera settings
        camera_group = QGroupBox("📹 Camera Settings")
        camera_layout = QGridLayout(camera_group)

        camera_layout.addWidget(QLabel("Camera Index:"), 0, 0)
        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 10)
        camera_layout.addWidget(self.camera_index_spin, 0, 1)

        camera_layout.addWidget(QLabel("Resolution:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        self.resolution_combo.setCurrentText("640x480")
        camera_layout.addWidget(self.resolution_combo, 1, 1)

        layout.addWidget(camera_group)

        # Detection settings
        detection_group = QGroupBox("🔍 Detection Settings")
        detection_layout = QGridLayout(detection_group)

        detection_layout.addWidget(QLabel("Auto-save detections:"), 0, 0)
        self.auto_save_checkbox = QCheckBox()
        detection_layout.addWidget(self.auto_save_checkbox, 0, 1)

        detection_layout.addWidget(QLabel("Save folder:"), 1, 0)
        self.save_folder_edit = QLineEdit("detections")
        detection_layout.addWidget(self.save_folder_edit, 1, 1)

        layout.addWidget(detection_group)

        # System settings
        system_group = QGroupBox("⚙️ System Settings")
        system_layout = QGridLayout(system_group)

        system_layout.addWidget(QLabel("Enable sound alerts:"), 0, 0)
        self.sound_alerts_checkbox = QCheckBox()
        self.sound_alerts_checkbox.setChecked(True)
        system_layout.addWidget(self.sound_alerts_checkbox, 0, 1)

        system_layout.addWidget(QLabel("Log level:"), 1, 0)
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["INFO", "DEBUG", "WARNING", "ERROR"])
        system_layout.addWidget(self.log_level_combo, 1, 1)

        layout.addWidget(system_group)

        # Save settings button
        self.save_settings_btn = QPushButton("💾 Save Settings")
        self.save_settings_btn.clicked.connect(self.save_settings)
        layout.addWidget(self.save_settings_btn)

        layout.addStretch()

        self.tabs.addTab(settings_tab, "⚙️ Settings")

    def create_logs_tab(self):
        """Create the system logs interface"""
        logs_tab = QWidget()
        layout = QVBoxLayout(logs_tab)

        # System log
        self.system_log = QTextEdit()
        self.system_log.setReadOnly(True)
        layout.addWidget(self.system_log)

        # Log controls
        log_controls = QHBoxLayout()
        self.clear_system_log_btn = QPushButton("🗑 Clear Log")
        self.save_system_log_btn = QPushButton("💾 Save Log")
        self.refresh_log_btn = QPushButton("🔄 Refresh")

        self.clear_system_log_btn.clicked.connect(self.clear_system_log)
        self.save_system_log_btn.clicked.connect(self.save_system_log)
        self.refresh_log_btn.clicked.connect(self.refresh_log)

        log_controls.addWidget(self.clear_system_log_btn)
        log_controls.addWidget(self.save_system_log_btn)
        log_controls.addWidget(self.refresh_log_btn)
        log_controls.addStretch()

        layout.addLayout(log_controls)

        self.tabs.addTab(logs_tab, "📝 Logs")

        # Initialize system log
        self.log_message("System initialized successfully")

    def setup_styling(self):
        """Apply custom styling to the GUI"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 10px;
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
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #45a049;
            }

            QPushButton:pressed {
                background-color: #3d8b40;
            }

            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }

            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }

            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
            }

            QComboBox, QSpinBox, QLineEdit {
                padding: 5px;
                border: 1px solid #555;
                border-radius: 4px;
                background-color: #3c3c3c;
                color: #ffffff;
            }

            QTabWidget::pane {
                border: 1px solid #555;
                border-radius: 4px;
            }

            QTabBar::tab {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                padding: 8px 12px;
                margin-right: 2px;
                border-radius: 4px 4px 0px 0px;
            }

            QTabBar::tab:selected {
                background-color: #4CAF50;
            }
        """)

    # Event handlers and methods
    def update_confidence(self, value):
        """Update confidence threshold"""
        self.confidence_threshold = value / 100.0
        self.confidence_label.setText(f"{self.confidence_threshold:.2f}")

    def update_cooldown(self, value):
        """Update alert cooldown"""
        self.alert_cooldown = value

    def start_camera(self):
        """Start camera feed"""
        try:
            camera_index = self.camera_index_spin.value()
            self.camera = cv2.VideoCapture(camera_index)

            if not self.camera.isOpened():
                self.show_error("Failed to open camera")
                return

            # Set resolution
            resolution = self.resolution_combo.currentText()
            width, height = map(int, resolution.split('x'))
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Start camera thread
            self.camera_thread = CameraThread(self.camera)
            self.camera_thread.frame_ready.connect(self.update_camera_display)
            self.camera_thread.start()

            self.camera_status_label.setText("✅ Camera: Connected")
            self.log_message("Camera started successfully")

        except Exception as e:
            self.show_error(f"Camera error: {str(e)}")

    def stop_camera(self):
        """Stop camera feed"""
        if hasattr(self, 'camera_thread'):
            self.camera_thread.stop()
            self.camera_thread.wait()

        if self.camera:
            self.camera.release()
            self.camera = None

        self.camera_status_label.setText("❌ Camera: Disconnected")
        self.camera_label.setText("Camera feed stopped")
        self.camera_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.log_message("Camera stopped")

    def capture_frame(self):
        """Capture current frame"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                self.log_message(f"Frame captured: {filename}")

    def start_detection(self):
        """Start AI detection"""
        if not self.model:
            self.show_error("No AI model loaded. Please load a model first.")
            return

        if not self.camera or not self.camera.isOpened():
            self.show_error("Camera not started. Please start camera first.")
            return

        self.detection_active = True
        self.detection_thread = DetectionThread(
            self.camera, self.model, self.confidence_threshold, self
        )
        self.detection_thread.detection_result.connect(self.handle_detection)
        self.detection_thread.start()

        self.log_message("AI detection started")

    def stop_detection(self):
        """Stop AI detection"""
        self.detection_active = False
        if hasattr(self, 'detection_thread') and self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.wait()

        self.log_message("AI detection stopped")

    def connect_arduino(self):
        """Connect to Arduino"""
        try:
            com_port = self.com_port_combo.currentText()
            self.arduino = serial.Serial(com_port, 9600, timeout=2)
            time.sleep(2)

            # Test connection
            self.arduino.write(b'STATUS\n')
            response = self.arduino.readline().decode().strip()

            if "ONLINE" in response or "Ready" in response:
                self.arduino_connected = True
                self.arduino_status_label.setText(f"✅ Arduino: Connected ({com_port})")
                self.log_message(f"Arduino connected on {com_port}")
            else:
                self.show_error("Arduino not responding correctly")

        except Exception as e:
            self.show_error(f"Arduino connection failed: {str(e)}")

    def test_buzzer(self):
        """Test Arduino buzzer"""
        if self.arduino_connected and self.arduino:
            try:
                self.arduino.write(b'MONKEY_DETECTED\n')
                self.log_message("Buzzer test sent to Arduino")
            except Exception as e:
                self.show_error(f"Buzzer test failed: {str(e)}")
        else:
            self.show_error("Arduino not connected")

    def load_model(self):
        """Load AI model"""
        if not YOLO_AVAILABLE:
            self.show_error("ultralytics package not installed. Run: pip install ultralytics")
            return

        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load YOLO Model", "", "Model files (*.pt);;All files (*.*)"
            )

            if file_path:
                self.model = YOLO(file_path)
                self.model_status_label.setText("✅ AI Model: Loaded")
                self.log_message(f"Model loaded: {file_path}")

        except Exception as e:
            self.show_error(f"Failed to load model: {str(e)}")

    def handle_detection(self, detected, confidence, frame):
        """Handle detection results"""
        if detected:
            current_time = time.time()
            if (current_time - self.last_detection_time) > self.alert_cooldown:
                self.detection_count += 1
                self.last_detection_time = current_time

                # Update UI
                self.detection_count_label.setText(f"Detections: {self.detection_count}")
                self.last_detection_label.setText(f"Last Detection: {datetime.now().strftime('%H:%M:%S')}")

                # Send Arduino alert
                if self.arduino_connected and self.arduino:
                    try:
                        self.arduino.write(b'MONKEY_DETECTED\n')
                    except:
                        pass

                # Log detection
                self.log_message(f"🐒 MONKEY DETECTED! Confidence: {confidence:.2f}")

                # Save frame if enabled
                if self.auto_save_checkbox.isChecked():
                    save_folder = Path(self.save_folder_edit.text())
                    save_folder.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = save_folder / f"monkey_detection_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)

    def update_camera_display(self, frame):
        """Update camera display in GUI"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale image to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)

    def update_status(self):
        """Update system status periodically"""
        # Update FPS if camera is running
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            fps = getattr(self.camera_thread, 'fps', 0)
            self.fps_label.setText(f"FPS: {fps:.1f}")

    def log_message(self, message):
        """Add message to detection log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        self.system_log.append(formatted_message)
        print(formatted_message)

    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        self.log_message(f"ERROR: {message}")

    def clear_log(self):
        """Clear detection log"""
        self.log_text.clear()

    def save_log(self):
        """Save detection log to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text files (*.txt);;All files (*.*)"
        )

        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.log_text.toPlainText())
            self.log_message(f"Log saved: {file_path}")

    # Training tab methods (simplified)
    def browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.dataset_path_edit.setText(folder)

    def create_dataset_structure(self):
        dataset_path = Path(self.dataset_path_edit.text())
        try:
            # Create folder structure
            folders = [
                "images/train", "images/val", "labels/train", "labels/val"
            ]
            for folder in folders:
                (dataset_path / folder).mkdir(parents=True, exist_ok=True)

            # Create data.yaml
            config = f"""path: {dataset_path}
train: images/train
val: images/val
nc: 1
names: ['monkey']
"""
            with open(dataset_path / "data.yaml", "w") as f:
                f.write(config)

            self.log_message("Dataset structure created successfully")

        except Exception as e:
            self.show_error(f"Failed to create dataset structure: {str(e)}")

    def collect_images(self):
        self.show_error("Image collection feature coming soon! Use webcam capture for now.")

    def annotate_images(self):
        self.show_error("Use LabelImg or Roboflow for image annotation.")

    def start_training(self):
        if not YOLO_AVAILABLE:
            self.show_error("ultralytics package not installed")
            return

        self.show_error("Training feature coming soon! Use command line training for now.")

    def stop_training(self):
        pass

    # Settings methods
    def save_settings(self):
        settings = {
            'camera_index': self.camera_index_spin.value(),
            'resolution': self.resolution_combo.currentText(),
            'auto_save': self.auto_save_checkbox.isChecked(),
            'save_folder': self.save_folder_edit.text(),
            'sound_alerts': self.sound_alerts_checkbox.isChecked(),
            'log_level': self.log_level_combo.currentText()
        }

        with open('settings.json', 'w') as f:
            json.dump(settings, f, indent=2)

        self.log_message("Settings saved")

    # Log methods
    def clear_system_log(self):
        self.system_log.clear()

    def save_system_log(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save System Log", f"system_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text files (*.txt);;All files (*.*)"
        )

        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.system_log.toPlainText())
            self.log_message(f"System log saved: {file_path}")

    def refresh_log(self):
        self.log_message("Log refreshed")

    def closeEvent(self, event):
        """Handle application closing"""
        # Stop all threads
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()

        if hasattr(self, 'detection_thread') and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread.wait()

        # Close hardware connections
        if self.camera:
            self.camera.release()

        if self.arduino:
            self.arduino.close()

        event.accept()

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                self.frame_ready.emit(frame)
                self.frame_count += 1

                # Calculate FPS
                elapsed = time.time() - self.start_time
                if elapsed > 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()

            self.msleep(30)  # ~30 FPS

    def stop(self):
        self.running = False

class DetectionThread(QThread):
    detection_result = pyqtSignal(bool, float, np.ndarray)  # detected, confidence, frame

    def __init__(self, camera, model, confidence_threshold, parent):
        super().__init__()
        self.camera = camera
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.parent = parent
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            if not self.parent.detection_active:
                break

            ret, frame = self.camera.read()
            if ret:
                # Run detection
                results = self.model(frame)

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

                                # Draw bounding box on frame
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f'Monkey: {confidence:.2f}', 
                                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.6, (0, 255, 0), 2)

                # Update camera display with detection overlay
                self.parent.update_camera_display(frame)

                # Emit detection result
                if monkey_detected:
                    self.detection_result.emit(True, max_confidence, frame)

            self.msleep(100)  # ~10 FPS for detection

    def stop(self):
        self.running = False

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI Monkey Detection System")

    # Set application icon (if available)
    # app.setWindowIcon(QIcon("icon.png"))

    window = MonkeyDetectionGUI()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
