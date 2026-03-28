"""Webcam + MediaPipe face landmark streaming utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp

from src.config import RuntimeConfig
from src.runtime_types import FrameData, LandmarkPoint


class LandmarkStreamError(RuntimeError):
    """Raised when camera or face-mesh streaming fails."""


@dataclass
class StreamStats:
    frames_seen: int = 0
    frames_with_face: int = 0
    dropped_frames: int = 0


class LandmarkStream:
    """Captures frames from webcam or video file and yields timestamped landmarks."""

    def __init__(self, cfg: RuntimeConfig, video_file_path: Optional[str] = None) -> None:
        self.cfg = cfg
        self.video_file_path = video_file_path
        self.capture: Optional[cv2.VideoCapture] = None
        self.face_mesh = None
        self.stats = StreamStats()
        self._video_frame_index = 0

    def __enter__(self) -> "LandmarkStream":
        # Open video file if provided, otherwise open webcam
        if self.video_file_path:
            self.capture = cv2.VideoCapture(self.video_file_path)
            if self.capture is None or not self.capture.isOpened():
                raise LandmarkStreamError(f"Unable to open video file: {self.video_file_path}")
        else:
            self.capture = cv2.VideoCapture(self.cfg.camera_index)
            if self.capture is None or not self.capture.isOpened():
                raise LandmarkStreamError(
                    f"Unable to open camera index {self.cfg.camera_index}."
                )

            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.cfg.max_faces,
            refine_landmarks=True,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        cv2.destroyAllWindows()

    @staticmethod
    def _extract_landmark_tuples(face_landmarks) -> List[LandmarkPoint]:
        points: List[LandmarkPoint] = []
        for lm in face_landmarks.landmark:
            points.append((float(lm.x), float(lm.y), float(lm.z)))
        return points

    def frames(self) -> Generator[Tuple[FrameData, object], None, None]:
        """Yield (FrameData, BGR frame) for each detected face frame.
        
        For video files: stops when file ends.
        For webcam: continues indefinitely until interrupted.
        """
        if self.capture is None or self.face_mesh is None:
            raise LandmarkStreamError("Stream is not initialized. Use context manager.")

        while True:
            ok, frame = self.capture.read()
            self.stats.frames_seen += 1
            if not ok or frame is None:
                self.stats.dropped_frames += 1
                # For video files, stop when we reach the end
                if self.video_file_path:
                    break
                # For webcam, continue (might be temporary frame drop)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                continue

            face_landmarks = results.multi_face_landmarks[0]
            landmark_tuples = self._extract_landmark_tuples(face_landmarks)
            if len(landmark_tuples) < 468:
                self.stats.dropped_frames += 1
                continue

            self.stats.frames_with_face += 1
            if self.video_file_path:
                # Use deterministic video time instead of wall-clock so temporal
                # features (latency, slope, velocity) remain meaningful offline.
                pos_msec = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
                if pos_msec > 0.0:
                    timestamp = pos_msec / 1000.0
                else:
                    self._video_frame_index += 1
                    timestamp = float(self._video_frame_index) / float(max(self.cfg.target_fps, 1e-6))
            else:
                timestamp = time.time()
            yield FrameData(timestamp=timestamp, landmarks=landmark_tuples), frame


def draw_face_landmarks(frame, face_landmarks, thickness: int = 1) -> None:
    """Optional helper for visual sanity checks on captured frames."""
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    spec = mp_drawing.DrawingSpec(color=(80, 220, 100), thickness=thickness, circle_radius=1)
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=spec,
    )
