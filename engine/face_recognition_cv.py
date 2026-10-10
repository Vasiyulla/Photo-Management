"""
Local Facial Recognition Engine (YuNet + SFace)
================================================
Architecture mimicking digiKam's face management pipeline:
- Face Detection: OpenCV YuNet ('FaceDetectorYN')
- Face Recognition: OpenCV SFace ('FaceRecognizerSF')

Strictly uses native OpenCV (cv2.dnn) without external face_recognition libraries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import urllib.request

import cv2
import numpy as np


# ==============================================================================
# Constants & Model URLs (Official OpenCV Zoo)
# ==============================================================================

DEFAULT_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
DEFAULT_SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# Recommended SFace cosine threshold according to OpenCV documentation:
# - Cosine similarity >= 0.36 indicates the same identity.
# - L2 norm distance <= 1.128 indicates the same identity.
DEFAULT_SFACE_COSINE_THRESHOLD: float = 0.36


# ==============================================================================
# Custom Exceptions
# ==============================================================================

class FaceRecognitionError(Exception):
    """Base exception for face recognition pipeline errors."""
    pass


class ModelNotFoundError(FaceRecognitionError, FileNotFoundError):
    """Raised when an ONNX model file does not exist."""
    pass


class ImageReadError(FaceRecognitionError, IOError):
    """Raised when an image cannot be read or decoded from disk."""
    pass


class NoFaceDetectedError(FaceRecognitionError):
    """Raised when no face is found in the target image."""
    pass


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class FaceMetadata:
    """Detailed face detection metadata."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    landmarks: list[Tuple[int, int]]  # 5 landmarks: right eye, left eye, nose, right mouth, left mouth
    score: float  # Detection confidence score [0.0 - 1.0]
    area: int  # Bounding box area (width * height)


@dataclass
class FaceFeatureResult:
    """Extraction output for a single face."""
    image_path: Path
    feature_vector: np.ndarray  # 128-dimensional embedding vector (float32)
    aligned_face: np.ndarray  # Aligned and cropped face image (112x112 BGR)
    metadata: FaceMetadata


@dataclass
class MatchResult:
    """Comparison result between two face feature vectors."""
    is_match: bool
    similarity: float
    threshold: float

    def __str__(self) -> str:
        status = "MATCH" if self.is_match else "NO MATCH"
        return f"[{status}] Similarity: {self.similarity:.4f} (Threshold: {self.threshold:.2f})"


# ==============================================================================
# Core Functions
# ==============================================================================

def init_face_models(
    yunet_path: Union[str, Path],
    sface_path: Union[str, Path],
    score_threshold: float = 0.8,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV,
    target_id: int = cv2.dnn.DNN_TARGET_CPU,
) -> Tuple[cv2.FaceDetectorYN, cv2.FaceRecognizerSF]:
    """
    Initialize OpenCV YuNet (FaceDetectorYN) and SFace (FaceRecognizerSF) models.

    Args:
        yunet_path: Path to the YuNet ONNX model (e.g. face_detection_yunet_2023mar.onnx).
        sface_path: Path to the SFace ONNX model (e.g. face_recognition_sface_2021dec.onnx).
        score_threshold: Filter out bounding boxes with score lower than this.
        nms_threshold: Non-maximum suppression threshold.
        top_k: Keep top K bounding boxes before NMS.
        backend_id: OpenCV DNN backend identifier (default: DNN_BACKEND_OPENCV).
        target_id: OpenCV DNN target device identifier (default: DNN_TARGET_CPU).

    Returns:
        A tuple of (FaceDetectorYN instance, FaceRecognizerSF instance).

    Raises:
        ModelNotFoundError: If either ONNX model file does not exist on disk.
    """
    y_path = Path(yunet_path).resolve()
    s_path = Path(sface_path).resolve()

    if not y_path.is_file():
        raise ModelNotFoundError(
            f"YuNet model not found at '{y_path}'. Please verify the path or download it."
        )
    if not s_path.is_file():
        raise ModelNotFoundError(
            f"SFace model not found at '{s_path}'. Please verify the path or download it."
        )

    # Initial input size is a placeholder (320, 320); it will be updated dynamically per image
    detector = cv2.FaceDetectorYN.create(
        model=str(y_path),
        config="",
        input_size=(320, 320),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
        backend_id=backend_id,
        target_id=target_id,
    )

    recognizer = cv2.FaceRecognizerSF.create(
        model=str(s_path),
        config="",
        backend_id=backend_id,
        target_id=target_id,
    )

    return detector, recognizer


def extract_face_features(
    image_path: Union[str, Path],
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
    select_primary_by: str = "area",
) -> FaceFeatureResult:
    """
    Robust pipeline to detect faces, align/crop the primary face, and extract its 128D embedding.

    Pipeline steps:
    1. Validates existence and loads the image from disk.
    2. Resizes YuNet's input dynamically to match image width and height.
    3. Runs face detection.
    4. Selects the primary face (by largest bounding box area or highest score).
    5. Aligns and crops the face using 5 facial landmarks.
    6. Extracts the 128-dimensional L2-normalized feature vector.

    Args:
        image_path: Path to the image file.
        detector: An initialized cv2.FaceDetectorYN instance.
        recognizer: An initialized cv2.FaceRecognizerSF instance.
        select_primary_by: Strategy for choosing primary face: 'area' (default) or 'score'.

    Returns:
        FaceFeatureResult containing the 128D vector, aligned face array, and metadata.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ImageReadError: If OpenCV cannot decode or open the image.
        NoFaceDetectedError: If no faces were detected in the image.
    """
    img_path = Path(image_path).resolve()
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: '{img_path}'")

    # Read image using OpenCV (supports JPEG, PNG, WEBP, TIFF, JFIF, etc.)
    image = cv2.imread(str(img_path))
    if image is None:
        raise ImageReadError(
            f"Failed to read image from '{img_path}'. The file may be corrupt or an unsupported format."
        )

    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ImageReadError(f"Image has invalid dimensions: width={width}, height={height}")

    # Dynamically update YuNet detector input size to match current image dimensions
    detector.setInputSize((width, height))

    # Detect faces: retval, faces (faces shape: [N, 15])
    _, raw_faces = detector.detect(image)

    if raw_faces is None or len(raw_faces) == 0:
        raise NoFaceDetectedError(f"No face detected in '{img_path.name}'.")

    # Each row in raw_faces contains 15 values:
    # 0-1: x, y (top-left bounding box)
    # 2-3: width, height
    # 4-13: 5 facial landmarks (x, y for right eye, left eye, nose tip, right mouth, left mouth)
    # 14: detection score

    # Select primary face
    if select_primary_by == "score":
        primary_idx = int(np.argmax(raw_faces[:, 14]))
    else:
        # Default: Largest bounding box area
        areas = raw_faces[:, 2] * raw_faces[:, 3]
        primary_idx = int(np.argmax(areas))

    primary_face = raw_faces[primary_idx]

    # Parse metadata
    x, y, w, h = int(primary_face[0]), int(primary_face[1]), int(primary_face[2]), int(primary_face[3])
    score = float(primary_face[14])
    landmarks = [
        (int(primary_face[4 + 2 * i]), int(primary_face[5 + 2 * i]))
        for i in range(5)
    ]
    metadata = FaceMetadata(
        bbox=(x, y, w, h),
        landmarks=landmarks,
        score=score,
        area=w * h,
    )

    # Align and crop the face using facial landmarks
    aligned_face = recognizer.alignCrop(image, primary_face)

    # Extract 128-dimensional embedding vector
    feature = recognizer.feature(aligned_face)
    feature_vector = feature.flatten().astype(np.float32)

    return FaceFeatureResult(
        image_path=img_path,
        feature_vector=feature_vector,
        aligned_face=aligned_face,
        metadata=metadata,
    )


def compare_features(
    feature1: np.ndarray,
    feature2: np.ndarray,
    threshold: float = DEFAULT_SFACE_COSINE_THRESHOLD,
    recognizer: Optional[cv2.FaceRecognizerSF] = None,
) -> MatchResult:
    """
    Compare two face feature vectors using Cosine Similarity.

    Args:
        feature1: First 128D feature vector.
        feature2: Second 128D feature vector.
        threshold: Cosine similarity cutoff (default: 0.36 for SFace).
        recognizer: Optional FaceRecognizerSF instance to use OpenCV's native match().

    Returns:
        MatchResult indicating whether the vectors match, the similarity score, and threshold.
    """
    f1 = np.asarray(feature1, dtype=np.float32)
    f2 = np.asarray(feature2, dtype=np.float32)

    if recognizer is not None:
        # Use OpenCV's native FaceRecognizerSF.match with FR_COSINE
        # Input features for recognizer.match must be 2D arrays of shape (1, 128)
        f1_mat = f1.reshape(1, -1)
        f2_mat = f2.reshape(1, -1)
        similarity = float(recognizer.match(f1_mat, f2_mat, cv2.FaceRecognizerSF_FR_COSINE))
    else:
        # Standard cosine similarity calculation
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        if norm1 == 0.0 or norm2 == 0.0:
            similarity = 0.0
        else:
            similarity = float(np.dot(f1, f2) / (norm1 * norm2))

    is_match = similarity >= threshold

    return MatchResult(
        is_match=is_match,
        similarity=similarity,
        threshold=threshold,
    )


# ==============================================================================
# High-Level Helper Class (digiKam-Style Pipeline)
# ==============================================================================

class DigiKamFaceEngine:
    """
    High-level facial recognition engine wrapping YuNet and SFace.
    Provides automated model management and clean scan-and-match interfaces.
    """

    def __init__(
        self,
        yunet_path: Union[str, Path],
        sface_path: Union[str, Path],
        score_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        cosine_threshold: float = DEFAULT_SFACE_COSINE_THRESHOLD,
    ):
        self.yunet_path = Path(yunet_path)
        self.sface_path = Path(sface_path)
        self.cosine_threshold = cosine_threshold

        self.detector, self.recognizer = init_face_models(
            yunet_path=self.yunet_path,
            sface_path=self.sface_path,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )

    @classmethod
    def from_models_dir(
        cls,
        models_dir: Union[str, Path] = "models",
        auto_download: bool = True,
        **kwargs,
    ) -> DigiKamFaceEngine:
        """Initialize engine from a directory, optionally downloading missing models."""
        m_dir = Path(models_dir)
        m_dir.mkdir(parents=True, exist_ok=True)

        yunet_file = m_dir / "face_detection_yunet_2023mar.onnx"
        sface_file = m_dir / "face_recognition_sface_2021dec.onnx"

        if auto_download:
            if not yunet_file.is_file():
                print(f"[FaceEngine] Downloading YuNet model to {yunet_file}...")
                urllib.request.urlretrieve(DEFAULT_YUNET_URL, str(yunet_file))
            if not sface_file.is_file():
                print(f"[FaceEngine] Downloading SFace model to {sface_file}...")
                urllib.request.urlretrieve(DEFAULT_SFACE_URL, str(sface_file))

        return cls(yunet_path=yunet_file, sface_path=sface_file, **kwargs)

    def process_image(self, image_path: Union[str, Path]) -> FaceFeatureResult:
        """Scan a photo, detect primary face, align, and extract 128D vector."""
        return extract_face_features(
            image_path=image_path,
            detector=self.detector,
            recognizer=self.recognizer,
        )

    def match(
        self,
        feature1: np.ndarray,
        feature2: np.ndarray,
        threshold: Optional[float] = None,
    ) -> MatchResult:
        """Compare two feature vectors."""
        th = threshold if threshold is not None else self.cosine_threshold
        return compare_features(feature1, feature2, threshold=th, recognizer=self.recognizer)

    def compare_images(
        self,
        image_path_1: Union[str, Path],
        image_path_2: Union[str, Path],
        threshold: Optional[float] = None,
    ) -> MatchResult:
        """Convenience method: compare primary faces in two image files directly."""
        res1 = self.process_image(image_path_1)
        res2 = self.process_image(image_path_2)
        return self.match(res1.feature_vector, res2.feature_vector, threshold=threshold)


# ==============================================================================
# Demonstration / Example Execution
# ==============================================================================

def main():
    """Simple demonstration matching a source photo against a target photo."""
    print("=" * 70)
    print("OpenCV YuNet + SFace Facial Recognition Engine (digiKam Architecture)")
    print("=" * 70)

    # 1. Define model paths
    models_directory = Path("models")
    yunet_model = models_directory / "face_detection_yunet_2023mar.onnx"
    sface_model = models_directory / "face_recognition_sface_2021dec.onnx"

    # 2. Functional API demonstration: Initializing models dynamically
    try:
        detector, recognizer = init_face_models(
            yunet_path=yunet_model,
            sface_path=sface_model,
        )
        print("[OK] Models initialized via init_face_models():")
        print(f"     YuNet: {yunet_model.name}")
        print(f"     SFace: {sface_model.name}\n")
    except Exception as exc:
        print(f"[ERROR] Model initialization failed: {exc}")
        return

    # 3. Define sample test pairs
    source_photo = Path("C:/Users/vasiy/Documents/Az/13.jfif")
    target_photo = Path("C:/Users/vasiy/Documents/Az/14.jfif")
    different_photo = Path("C:/Users/vasiy/Documents/Human Faces and Objects Mix Image Dataset/Human faces and object dataset/Images/male_faces/male_1.jpg")

    print("--- [Example 1: Matching Photos (Functional API)] ---")
    print(f"  Source: {source_photo.name}")
    print(f"  Target: {target_photo.name}")
    try:
        # Step A: Scan & extract features from source image
        res_source = extract_face_features(source_photo, detector, recognizer)
        print(f"  Source face detected at bbox {res_source.metadata.bbox} (score: {res_source.metadata.score:.2%})")

        # Step B: Scan & extract features from target image
        res_target = extract_face_features(target_photo, detector, recognizer)
        print(f"  Target face detected at bbox {res_target.metadata.bbox} (score: {res_target.metadata.score:.2%})")

        # Step C: Compare features using Cosine Similarity (threshold: 0.36)
        match_result = compare_features(
            res_source.feature_vector,
            res_target.feature_vector,
            threshold=0.36,
            recognizer=recognizer,
        )
        print(f"  Comparison Result: {match_result}\n")
    except FaceRecognitionError as err:
        print(f"  [ERROR] {err}\n")

    print("--- [Example 2: Comparing Different People] ---")
    print(f"  Source: {source_photo.name}")
    print(f"  Target: {different_photo.name}")
    try:
        res_diff = extract_face_features(different_photo, detector, recognizer)
        match_diff = compare_features(res_source.feature_vector, res_diff.feature_vector, threshold=0.36)
        print(f"  Comparison Result: {match_diff}\n")
    except FaceRecognitionError as err:
        print(f"  [ERROR] {err}\n")

    print("--- [Example 3: Error Handling Demonstration] ---")
    # A: Missing file
    try:
        print("  Attempting to process non-existent image...")
        extract_face_features("non_existent_image.jpg", detector, recognizer)
    except FileNotFoundError as err:
        print(f"  [OK Handled] Caught FileNotFoundError: {err}")

    # B: Image with no face (create a synthetic blank image)
    blank_test_path = Path("scratch_blank_test.jpg")
    try:
        cv2.imwrite(str(blank_test_path), np.zeros((300, 300, 3), dtype=np.uint8))
        print("  Attempting to process image with no faces...")
        extract_face_features(blank_test_path, detector, recognizer)
    except NoFaceDetectedError as err:
        print(f"  [OK Handled] Caught NoFaceDetectedError: {err}")
    finally:
        if blank_test_path.exists():
            blank_test_path.unlink()

    print("\n" + "=" * 70)
    print("Execution completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
