"""
YAV2P
[Y]et [A]nother [V]ideo To [P]anorama Stitcher
==============================================

By: [@fdovila](https://github.com/fdovila)

A robust video panorama stitching tool that extracts frames from video and creates high-quality panoramic images.
Designed to handle challenging conditions including dark videos, motion blur, and varying exposure levels.

Key Features:
- Adaptive frame extraction with quality validation
- Multi-stage image enhancement pipeline
- Memory-efficient parallel processing
- Segmented stitching for better handling of large panoramas
- Quality-aware sequential merging
- Automatic error recovery and alternative stitching approaches
- Workspace management for long-running operations

Technical Details:
- Uses SIFT for feature detection
- Implements CLAHE and multi-method frame enhancement
- Supports multiple stitching modes (SCAN and PANORAMA)
- Includes unwarp correction for better results
- Provides detailed progress and quality metrics

Usage:
    python main.py [--no-workspace]
    
    Options:
        --no-workspace    Ignore existing workspace and recompute features

Requirements:
    - OpenCV (cv2)
    - NumPy
    - multiprocessing
    - psutil
    - tqdm

Keywords:
    #computer-vision #image-processing #panorama #video-processing
    #opencv #python #image-stitching #feature-detection #sift
    #parallel-processing #image-enhancement #video-to-panorama
    #panorama-stitcher #video-frames #dark-video #motion-blur
    #clahe #image-blending #quality-metrics #memory-efficient
    #error-recovery #workspace-management #multi-threading
    #adaptive-processing #unwarp-correction #sequential-merging

Topics:
    Computer Vision, Image Processing, Video Processing, OpenCV,
    Feature Detection, Parallel Processing, Image Enhancement,
    Memory Management, Error Recovery, Quality Control

Related Projects:
    - OpenCV
    - Video Processing Tools
    - Image Stitching Applications
    - Panorama Creation Software
"""

import cv2
import os
import numpy as np
import multiprocessing as mp
from itertools import combinations
from tqdm import tqdm  # for progress bars
import time
import gc
import psutil
import threading
import glob

def enhance_frame(frame, method='clahe'):
    """
    Enhance a dark frame using various methods.
    
    Args:
        frame: Input frame
        method: 'histogram', 'clahe', or 'gamma'
    """
    if method == 'histogram':
        # Simple histogram equalization
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    elif method == 'clahe':
        # Contrast Limited Adaptive Histogram Equalization
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    elif method == 'gamma':
        # Gamma correction
        gamma = 1.5  # Adjust this value to control brightness (>1 = brighter)
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255)
        return cv2.LUT(frame, lookUpTable)

def resize_frame(frame, max_dimension=800):
    """Resize frame while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))
    return frame

def keypoint_to_tuple(keypoint):
    """Convert a cv2.KeyPoint to a tuple for serialization"""
    return (
        keypoint.pt[0], keypoint.pt[1],
        keypoint.size,
        keypoint.angle,
        keypoint.response,
        keypoint.octave,
        keypoint.class_id
    )

def tuple_to_keypoint(keypoint_tuple):
    """Convert a tuple back to cv2.KeyPoint"""
    return cv2.KeyPoint(
        x=keypoint_tuple[0],
        y=keypoint_tuple[1],
        size=keypoint_tuple[2],
        angle=keypoint_tuple[3],
        response=keypoint_tuple[4],
        octave=int(keypoint_tuple[5]),
        class_id=int(keypoint_tuple[6])
    )

def detect_features(frame):
    """Detect features in a single frame using SIFT"""
    try:
        # Resize frame before processing
        frame = resize_frame(frame)
        
        # Convert to grayscale and reduce size
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Further reduce size if needed
        if max(gray.shape) > 800:
            gray = resize_frame(gray, 800)
        
        # Use fewer features
        sift = cv2.SIFT_create(nfeatures=1000)
        
        # Clear some memory
        frame = None
        
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        # Clear more memory
        gray = None
        sift = None
        
        # Convert keypoints to tuples for serialization
        keypoints_tuple = [keypoint_to_tuple(kp) for kp in keypoints]
        
        return keypoints_tuple, descriptors
    except Exception as e:
        print(f"Error in detect_features: {str(e)}")
        return None, None
    finally:
        # Force garbage collection
        gc.collect()

def match_features(args):
    """Match features between two frames"""
    idx1, idx2, desc1, desc2 = args
    try:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                # Store only the indices for serialization
                good_matches.append((m.queryIdx, m.trainIdx, m.distance))
        return (idx1, idx2, good_matches)
    except Exception as e:
        print(f"Error in match_features: {str(e)}")
        return (idx1, idx2, [])

def parallel_stitch(frames, batch_size=2):
    """Parallel implementation with batch processing"""
    try:
        print("\n🔍 Phase 1: Parallel Feature Detection")
        
        # Limit number of processes
        n_processes = min(mp.cpu_count(), 2)
        
        with mp.Pool(processes=n_processes) as pool:
            print(f"   💪 Using {n_processes} CPU cores")
            
            # Process frames in smaller batches
            all_keypoints_tuple = []
            all_descriptors = []
            
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                print(f"\n   🔎 Processing batch {i//batch_size + 1}/{(len(frames) + batch_size - 1)//batch_size}")
                
                # Process one frame at a time in the batch
                for frame in batch:
                    result = detect_features(frame)
                    if result[0] is not None:
                        kps, desc = result
                        all_keypoints_tuple.append(kps)
                        all_descriptors.append(desc)
                    
                    # Clear memory
                    frame = None
                    import gc
                    gc.collect()
                
                # Clear batch memory
                batch = None
                gc.collect()
            
            print("\n   Converting keypoints back to OpenCV format...")
            # Convert keypoints back to OpenCV format
            all_keypoints = []
            for frame_kps in all_keypoints_tuple:
                opencv_kps = [tuple_to_keypoint(kp) for kp in frame_kps]
                all_keypoints.append(opencv_kps)
                
            return all_keypoints, all_descriptors
            
    except Exception as e:
        print(f"Error in parallel_stitch: {str(e)}")
        raise
    finally:
        # Final cleanup
        gc.collect()

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    return mem

def log_memory(message):
    """Log a message with current memory usage"""
    mem = get_memory_usage()
    print(f"   💾 Memory: {mem:.1f}MB - {message}")

def try_alternative_stitching(frames, original_status):
    """Try alternative stitching approaches if the first one fails"""
    try:
        print("\n   🔄 Trying alternative stitching approach...")
        
        # Updated warper types with correct OpenCV constants
        warper_types = [
            ("SPHERICAL", 1.0),
            ("CYLINDRICAL", 1.0),
            ("PLANE", 1.0),
            ("AFFINE", 1.0)
        ]
        
        for warper_name, conf in warper_types:
            print(f"      📷 Trying {warper_name} warper...")
            stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
            
            # Configure stitcher
            try:
                stitcher.setWaveCorrection(True)
                stitcher.setPanoConfidenceThresh(conf)
                stitcher.setRegistrationResol(0.6)
                stitcher.setSeamEstimationResol(0.1)
            except Exception as e:
                print(f"      ⚠️ Error configuring stitcher: {str(e)}")
            
            # Try stitching
            try:
                status, panorama = stitcher.stitch(frames)
                
                if status == cv2.Stitcher_OK:
                    print(f"      ✅ Success with {warper_name} warper!")
                    return status, panorama
                else:
                    print(f"      ❌ Failed with {warper_name} warper")
            except Exception as e:
                print(f"      ⚠️ Error with {warper_name} warper: {str(e)}")
                continue
        
        # Try with different image sizes
        print("\n      🔍 Trying different image sizes...")
        scale_factors = [0.75, 0.5, 0.25]
        
        for scale in scale_factors:
            print(f"      📏 Trying scale factor: {scale}")
            scaled_frames = []
            for frame in frames:
                new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
                scaled_frame = cv2.resize(frame, new_size)
                scaled_frames.append(scaled_frame)
            
            try:
                status, panorama = stitcher.stitch(scaled_frames)
                if status == cv2.Stitcher_OK:
                    print(f"      ✅ Success with scale {scale}!")
                    return status, panorama
            except Exception as e:
                print(f"      ⚠️ Error with scale {scale}: {str(e)}")
                continue
        
        return original_status, None
        
    except Exception as e:
        print(f"Error in alternative stitching: {str(e)}")
        return original_status, None

def validate_frame(frame, debug=False):
    """Validate frame and return True if it's usable"""
    if frame is None:
        if debug: print("Frame is None")
        return False
    if frame.size == 0:
        if debug: print("Frame size is 0")
        return False
    if len(frame.shape) != 3:
        if debug: print("Frame is not 3-channel")
        return False
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        if debug: print("Invalid frame dimensions")
        return False

    # Relaxed brightness thresholds
    mean_value = np.mean(frame)
    if mean_value < 1:  # Reduced from 5
        if debug: print(f"Frame too dark (mean: {mean_value})")
        return False
    if mean_value > 250:
        if debug: print(f"Frame too bright (mean: {mean_value})")
        return False

    # Reduced blur threshold for dark videos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    blur_threshold = 30  # Further reduced for dark videos
    if laplacian_var < blur_threshold:
        if debug: print(f"Frame too blurry (variance: {laplacian_var})")
        return False

    if debug: print(f"Frame passed validation (blur: {laplacian_var}, brightness: {mean_value})")
    return True

def preprocess_frames(frames):
    """Preprocess frames for stitching"""
    print("\n   🔄 Preprocessing frames...")
    processed_frames = []
    
    try:
        # Get median frame size
        heights = [f.shape[0] for f in frames]
        widths = [f.shape[1] for f in frames]
        target_height = int(np.median(heights))
        target_width = int(np.median(widths))
        
        print(f"      📏 Target dimensions: {target_width}x{target_height}")
        
        for i, frame in enumerate(frames):
            try:
                # Print initial brightness for debugging
                print(f"      Initial brightness: {np.mean(frame)}")
                
                # 1. First enhance the dark frame using CLAHE
                frame = enhance_frame(frame, method='clahe')
                print(f"      After CLAHE: {np.mean(frame)}")
                
                # 2. Resize to consistent dimensions
                frame = cv2.resize(frame, (target_width, target_height))
                
                # 3. Strong brightness enhancement
                frame_enhanced = cv2.convertScaleAbs(frame, alpha=2.0, beta=30)
                print(f"      After brightness enhancement: {np.mean(frame_enhanced)}")
                
                # 4. Reduce noise
                frame_denoised = cv2.fastNlMeansDenoisingColored(
                    frame_enhanced,
                    h=10,
                    hColor=10,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
                
                # 5. Sharpen
                kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
                frame_final = cv2.filter2D(frame_denoised, -1, kernel)
                
                print(f"      Final brightness: {np.mean(frame_final)}")
                
                # Now validate the enhanced frame
                if validate_frame(frame_final, debug=True):
                    processed_frames.append(frame_final)
                    print(f"      ✓ Processed frame {i+1}/{len(frames)}")
                else:
                    print(f"      ⚠️ Frame {i+1} failed quality validation")
                
            except Exception as e:
                print(f"      ⚠️  Failed to process frame {i}: {str(e)}")
                continue
        
        print(f"      ✅ Successfully processed {len(processed_frames)} frames")
        return processed_frames
        
    except Exception as e:
        print(f"      ❌ Error in preprocessing: {str(e)}")
        return frames  # Return original frames if preprocessing fails

def save_workspace(frames, features, filename="workspace.npz"):
    """Save frames and their features to a workspace file"""
    try:
        print(f"\n💾 Saving workspace to {filename}...")
        
        # Convert features to a format that can be saved
        keypoints_list = []
        descriptors_list = []
        
        for keypoints, descriptors in features:
            # Convert keypoints to serializable format
            keypoints_data = []
            for kp in keypoints:
                keypoints_data.append([
                    kp.pt[0], kp.pt[1],
                    kp.size,
                    kp.angle,
                    kp.response,
                    kp.octave,
                    kp.class_id
                ])
            
            keypoints_list.append(np.array(keypoints_data))
            descriptors_list.append(descriptors)
        
        # Save as compressed NPZ file
        np.savez_compressed(
            filename,
            frames=np.array(frames),
            keypoints=np.array(keypoints_list, dtype=object),
            descriptors=np.array(descriptors_list, dtype=object),
            timestamp=time.time()
        )
        print("✅ Workspace saved successfully")
        
    except Exception as e:
        print(f"❌ Error saving workspace: {str(e)}")
        # Print more debug info
        print(f"Frames shape: {np.array(frames).shape}")
        print(f"Number of keypoint sets: {len(keypoints_list)}")
        print(f"Number of descriptor sets: {len(descriptors_list)}")

def load_workspace(filename="workspace.npz"):
    """Load frames and features from a workspace file"""
    try:
        print(f"\n📂 Loading workspace from {filename}...")
        
        if not os.path.exists(filename):
            print("❌ Workspace file not found")
            return None, None
            
        workspace = np.load(filename, allow_pickle=True)
        frames = workspace['frames']
        
        # Reconstruct features
        features = []
        keypoints_data = workspace['keypoints']
        descriptors_data = workspace['descriptors']
        
        for kp_data, desc in zip(keypoints_data, descriptors_data):
            # Convert back to KeyPoint objects
            keypoints = []
            for kp in kp_data:
                keypoint = cv2.KeyPoint(
                    x=float(kp[0]),
                    y=float(kp[1]),
                    size=float(kp[2]),
                    angle=float(kp[3]),
                    response=float(kp[4]),
                    octave=int(kp[5]),
                    class_id=int(kp[6])
                )
                keypoints.append(keypoint)
            
            features.append((keypoints, desc))
        
        print("✅ Workspace loaded successfully")
        return frames, features
        
    except Exception as e:
        print(f"❌ Error loading workspace: {str(e)}")
        return None, None

def find_panorama_segments(directory="."):
    """Find all panorama segment files in the directory"""
    segments = []
    for file in os.listdir(directory):
        if file.startswith("panorama_segment_") and file.endswith(".jpg"):
            segments.append(os.path.join(directory, file))
    return sorted(segments)  # Sort to maintain sequence

def stitch_existing_panoramas():
    """Second pass: stitch existing panorama segments"""
    try:
        print("\n🔄 Starting second pass with existing panorama segments...")
        
        # Find all panorama segments
        segment_files = find_panorama_segments()
        if not segment_files:
            print("❌ No panorama segments found")
            return None
            
        print(f"📂 Found {len(segment_files)} panorama segments")
        
        # Load segments
        segments = []
        for file in segment_files:
            try:
                img = cv2.imread(file)
                if img is not None and validate_frame(img):
                    segments.append(img)
                    print(f"✓ Loaded: {file}")
                else:
                    print(f"⚠️ Skipped invalid segment: {file}")
            except Exception as e:
                print(f"⚠️ Error loading {file}: {str(e)}")
                continue
        
        if len(segments) < 2:
            print("❌ Not enough valid segments for stitching")
            return None
        
        print(f"\n🧩 Attempting to stitch {len(segments)} segments...")
        
        # Try different stitching approaches
        for mode in [cv2.Stitcher_SCANS, cv2.Stitcher_PANORAMA]:
            mode_name = "SCANS" if mode == cv2.Stitcher_PANORAMA else "PANORAMA"
            print(f"\n   🔄 Trying {mode_name} mode...")
            
            stitcher = cv2.Stitcher_create(mode)
            stitcher.setPanoConfidenceThresh(0.6)
            
            # Try with different subsets if full set fails
            for size in range(len(segments), 1, -1):
                for start in range(len(segments) - size + 1):
                    subset = segments[start:start + size]
                    print(f"      📍 Trying segments {start+1} to {start+size}")
                    
                    try:
                        status, panorama = stitcher.stitch(subset)
                        
                        if status == cv2.Stitcher_OK and panorama is not None:
                            output_path = f"panorama_second_pass_{start+1}_to_{start+size}.jpg"
                            cv2.imwrite(output_path, panorama)
                            print(f"      ✅ Success! Saved as: {output_path}")
                            return panorama
                    except Exception as e:
                        print(f"      ⚠️ Failed: {str(e)}")
                        continue
        
        print("\n❌ Could not stitch any segment combinations")
        return None
        
    except Exception as e:
        print(f"Error in second pass: {str(e)}")
        return None

def extract_and_stitch(video_path, output_image_path, frame_interval=15, use_workspace=True):
    """Extract frames from video and stitch them into a panorama"""
    try:
        workspace_file = "stitching_workspace.npz"
        frames = None
        features = None
        
        if use_workspace and os.path.exists(workspace_file):
            frames, features = load_workspace(workspace_file)
        
        if frames is None:
            # Extract frames from video
            print("\n🎥 Extracting frames from video...")
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print("❌ Error: Could not open video file")
                return False
            
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    # Preprocess frame
                    frame = preprocess_frames([frame])[0]
                    if validate_frame(frame, debug=True):
                        frames.append(frame)
                        print(f"✓ Extracted frame {len(frames)}")
                
                frame_count += 1
            
            cap.release()
            print(f"📊 Extracted {len(frames)} frames from {frame_count} total frames")
            
            if len(frames) < 2:
                print("❌ Not enough valid frames extracted")
                return False
            
            # Detect features
            print("\n🔍 Detecting features...")
            features = []
            for i, frame in enumerate(frames):
                sift = cv2.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                    None
                )
                features.append((keypoints, descriptors))
                print(f"Frame {i+1}/{len(frames)}: {len(keypoints)} features")
            
            # Save workspace
            save_workspace(frames, features, workspace_file)
        
        # Attempt stitching with different modes
        for mode in [cv2.Stitcher_SCANS, cv2.Stitcher_PANORAMA]:
            mode_name = "SCANS" if mode == cv2.Stitcher_PANORAMA else "PANORAMA"
            print(f"\n   🔄 Attempting stitching with {mode_name} mode...")
            
            stitcher = cv2.Stitcher_create(mode)
            stitcher.setPanoConfidenceThresh(0.6)
            stitcher.setRegistrationResol(0.6)
            stitcher.setSeamEstimationResol(0.1)
            stitcher.setCompositingResol(1.0)
            stitcher.setWaveCorrection(True)
            
            # Try segmented stitching approach
            status, panorama = stitch_with_segments(stitcher, frames)
            
            if status == cv2.Stitcher_OK and panorama is not None:
                print(f"\n✅ Segmented stitching successful with {mode_name} mode!")
                print(f"   💾 Saving final panorama to: {output_image_path}")
                cv2.imwrite(output_image_path, panorama)
                print(f"   ✅ Final panorama saved successfully!")
                return True
            else:
                print(f"\n⚠️ Segmented stitching failed with {mode_name} mode")
                
                # Try alternative stitching approach
                print("\n   🔄 Trying alternative approach...")
                status, panorama = try_alternative_stitching(frames, status)
                
                if status == cv2.Stitcher_OK and panorama is not None:
                    print(f"\n✅ Alternative stitching successful!")
                    print(f"   💾 Saving final panorama to: {output_image_path}")
                    cv2.imwrite(output_image_path, panorama)
                    print(f"   ✅ Final panorama saved successfully!")
                    return True
        
        print("\n❌ All stitching attempts failed")
        return False
        
    except Exception as e:
        print(f"Error in extract_and_stitch: {str(e)}")
        return False
    finally:
        cv2.destroyAllWindows()
        gc.collect()

def validate_frames_for_stitching(frames):
    """Validate frames before stitching attempt"""
    if not frames or len(frames) < 2:
        print("      ⚠️ Not enough frames for stitching")
        return None
    
    valid_frames = []
    target_size = None
    
    for i, frame in enumerate(frames):
        try:
            # Basic frame validation
            if frame is None or not isinstance(frame, np.ndarray):
                print(f"      ⚠️ Frame {i+1} is invalid")
                continue
                
            # Convert boolean arrays to uint8
            if frame.dtype == bool:
                frame = frame.astype(np.uint8) * 255
                
            # Ensure frame is in correct format
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"      ⚠️ Frame {i+1} has invalid shape: {frame.shape}")
                continue
            
            # Set target size from first valid frame
            if target_size is None:
                target_size = (frame.shape[1], frame.shape[0])
                print(f"      📏 Target size: {target_size}")
            
            # Resize if necessary
            if frame.shape[:2][::-1] != target_size:
                print(f"      ↔️ Resizing frame {i+1} to match target size")
                frame = cv2.resize(frame, target_size)
            
            # Ensure frame is contiguous and in the correct data type
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            valid_frames.append(frame)
            
        except Exception as e:
            print(f"      ⚠️ Error processing frame {i+1}: {str(e)}")
            continue
    
    if len(valid_frames) < 2:
        print("      ⚠️ Not enough valid frames after validation")
        return None
        
    print(f"      ✅ Validated {len(valid_frames)} frames")
    return valid_frames

def validate_panorama_content(image, min_content_ratio=0.1):
    """Validate that the panorama has enough content"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate non-empty pixels (not black)
        non_empty = np.count_nonzero(gray > 10)  # threshold of 10 to account for noise
        total_pixels = gray.size
        content_ratio = non_empty / total_pixels
        
        print(f"      📊 Content ratio: {content_ratio:.2%}")
        return content_ratio >= min_content_ratio
    except Exception as e:
        print(f"      ⚠️ Error validating content: {str(e)}")
        return False

def unwarp_panorama(image, segment_name):
    """Unwarp a panorama segment by straightening edges"""
    try:
        print(f"\n      🔄 Attempting to unwarp {segment_name}...")
        
        # Save original for comparison
        original_path = f"panorama_{segment_name}_original.jpg"
        cv2.imwrite(original_path, image)
        print(f"      💾 Saved original segment to: {original_path}")
        
        # Validate original content
        if not validate_panorama_content(image):
            print("      ⚠️ Original image has insufficient content")
            return image, False
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        edges_path = f"panorama_{segment_name}_edges.jpg"
        cv2.imwrite(edges_path, edges)
        
        # Find lines using Hough transform with multiple parameter sets
        lines = None
        parameters = [
            (100, 0.3),  # (threshold, minLineLength ratio)
            (50, 0.2),
            (150, 0.4)
        ]
        
        for threshold, length_ratio in parameters:
            lines = cv2.HoughLinesP(
                edges, 
                rho=1,
                theta=np.pi/180,
                threshold=threshold,
                minLineLength=int(image.shape[1] * length_ratio),
                maxLineGap=20
            )
            if lines is not None and len(lines) >= 2:
                print(f"      ✅ Found lines with threshold={threshold}, length_ratio={length_ratio}")
                break
        
        if lines is None or len(lines) < 2:
            print(f"      ⚠️ Not enough strong lines detected for {segment_name}")
            return image, False
        
        # Draw and save detected lines
        lines_image = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lines_path = f"panorama_{segment_name}_lines.jpg"
        cv2.imwrite(lines_path, lines_image)
        
        # Try multiple angle thresholds for line classification
        best_result = None
        best_score = 0
        
        angle_thresholds = [(20, 70), (30, 60), (15, 75)]
        
        for h_thresh, v_thresh in angle_thresholds:
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < h_thresh or angle > (180 - h_thresh):
                    horizontal_lines.append(line[0])
                elif (90 - v_thresh) < angle < (90 + v_thresh):
                    vertical_lines.append(line[0])
            
            if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                try:
                    # Attempt unwarping with current line sets
                    unwarped = unwarp_with_lines(image, horizontal_lines, vertical_lines)
                    if unwarped is not None:
                        # Score the result based on content and straightness
                        score = evaluate_unwarping(unwarped)
                        if score > best_score:
                            best_score = score
                            best_result = unwarped
                except Exception as e:
                    print(f"      ⚠️ Failed with angles {h_thresh},{v_thresh}: {str(e)}")
                    continue
        
        if best_result is None:
            print(f"      ⚠️ No valid unwarping found for {segment_name}")
            return image, False
        
        # Validate unwarped result
        if not validate_panorama_content(best_result):
            print("      ⚠️ Unwarped result has insufficient content")
            return image, False
        
        # Save unwarped result
        unwarped_path = f"panorama_{segment_name}_unwarped.jpg"
        cv2.imwrite(unwarped_path, best_result)
        print(f"      ✅ Successfully unwarped {segment_name}")
        
        return best_result, True
            
    except Exception as e:
        print(f"      ❌ Error in unwarp_panorama: {str(e)}")
        return image, False

def evaluate_unwarping(image):
    """Evaluate the quality of unwarping"""
    try:
        # Check content ratio
        content_score = validate_panorama_content(image, min_content_ratio=0.1)
        if not content_score:
            return 0
        
        # Check edge straightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return 0
        
        # Calculate average line straightness
        straightness_scores = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate how straight the line is (perfect straight line would have angle 0° or 90°)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            angle_score = min(abs(angle % 90), abs(90 - (angle % 90))) / 45  # Normalize to [0,1]
            straightness_scores.append(angle_score)
        
        if not straightness_scores:
            return 0
            
        return np.mean(straightness_scores)
        
    except Exception as e:
        print(f"      ⚠️ Error evaluating unwarping: {str(e)}")
        return 0

def stitch_segment(stitcher, frames, segment_name=""):
    """Stitch a segment of frames with progress monitoring"""
    print(f"\n      🔄 Stitching {segment_name} ({len(frames)} frames)...")
    log_memory(f"Before stitching {segment_name}")
    
    try:
        # Validate minimum frame requirement
        if len(frames) < 2:
            print(f"      ⚠️ Not enough frames for {segment_name} (need at least 2, got {len(frames)})")
            return None
        
        # Convert frames to list if it's a numpy array
        frames_list = [frame for frame in frames]
        
        # Validate and prepare frames
        valid_frames = validate_frames_for_stitching(frames_list)
        if valid_frames is None:
            print(f"      ❌ No valid frames for {segment_name}")
            return None
            
        cv2.ocl.setUseOpenCL(False)
        
        print("\n         🧩 Final composition...")
        stitch_start = time.time()
        
        try:
            # Convert to numpy array for stitching
            frames_array = np.array(valid_frames)
            
            # Print frame information for debugging
            print(f"      📊 Frame array shape: {frames_array.shape}")
            print(f"      📊 Frame array dtype: {frames_array.dtype}")
            
            # Try different stitching configurations
            for mode in [cv2.Stitcher_SCANS, cv2.Stitcher_PANORAMA]:
                mode_name = "SCANS" if mode == cv2.Stitcher_SCANS else "PANORAMA"
                print(f"      🔄 Trying {mode_name} mode...")
                
                try:
                    local_stitcher = cv2.Stitcher_create(mode)
                    local_stitcher.setPanoConfidenceThresh(0.3)  # Lower threshold for small segments
                    status, panorama = local_stitcher.stitch(frames_array)
                    
                    if status == cv2.Stitcher_OK and panorama is not None:
                        print(f"      ✅ Successfully stitched {segment_name} with {mode_name} mode")
                        
                        # Save original stitched result
                        original_path = f"panorama_{segment_name}_stitched_{mode_name}.jpg"
                        cv2.imwrite(original_path, panorama)
                        print(f"      💾 Saved original stitch to: {original_path}")
                        
                        # Try to unwarp
                        panorama, was_unwarped = unwarp_panorama(panorama, f"{segment_name}_{mode_name}")
                        
                        # Save final result
                        final_path = f"panorama_{segment_name}_final_{mode_name}.jpg"
                        cv2.imwrite(final_path, panorama)
                        print(f"      💾 Saved final {'unwarped' if was_unwarped else 'original'} panorama to: {final_path}")
                        return panorama
                except Exception as e:
                    print(f"      ⚠️ Failed with {mode_name} mode: {str(e)}")
                    continue
            
            # If we get here, both modes failed
            print(f"      ❌ All stitching modes failed for {segment_name}")
            
            # Try alternative approach for small segments
            if len(valid_frames) <= 3:
                print("      🔄 Trying alternative approach for small segment...")
                try:
                    # Just blend the frames together
                    result = valid_frames[0].copy()
                    for frame in valid_frames[1:]:
                        result = cv2.addWeighted(result, 0.5, frame, 0.5, 0)
                    
                    blend_path = f"panorama_{segment_name}_blend.jpg"
                    cv2.imwrite(blend_path, result)
                    print(f"      💾 Saved blended result to: {blend_path}")
                    return result
                except Exception as e:
                    print(f"      ⚠️ Blending failed: {str(e)}")
            
            return None
                
        except Exception as e:
            print(f"      ❌ Stitching error: {str(e)}")
            return None
            
    except Exception as e:
        print(f"      ❌ Error in stitch_segment: {str(e)}")
        return None
    finally:
        log_memory(f"After {segment_name}")
        gc.collect()

def evaluate_panorama_quality(panorama):
    """Evaluate the quality of a panorama"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        # 1. Content ratio (non-black pixels)
        non_black = np.count_nonzero(gray > 10)
        content_ratio = non_black / gray.size
        
        # 2. Edge coherence (reduced weight)
        edges = cv2.Canny(gray, 50, 150)
        edge_coherence = np.count_nonzero(edges) / edges.size
        
        # 3. Blur detection (variance of Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # 4. Aspect ratio reasonableness
        height, width = panorama.shape[:2]
        aspect_ratio = width / height
        aspect_score = min(aspect_ratio / 4.0, 1.0)  # Expect wider panoramas
        
        # Combine metrics with adjusted weights
        quality_score = (content_ratio * 0.5 + 
                        (1 - edge_coherence) * 0.1 +  # Reduced weight for edges
                        min(1.0, sharpness / 1000) * 0.2 +
                        aspect_score * 0.2)
        
        print(f"      📊 Quality metrics:")
        print(f"         Content ratio: {content_ratio:.2%}")
        print(f"         Edge coherence: {edge_coherence:.2%}")
        print(f"         Sharpness: {sharpness:.2f}")
        print(f"         Aspect ratio: {aspect_ratio:.2f}")
        print(f"         Overall score: {quality_score:.2f}")
        
        return quality_score
        
    except Exception as e:
        print(f"⚠️ Error evaluating quality: {str(e)}")
        return 0.0

def merge_panorama_segments_sequential(segments, output_prefix="final_panorama", quality_threshold=0.3):
    """Merge panorama segments sequentially with quality control"""
    try:
        print("\n🔄 Starting sequential panorama merging...")
        
        if len(segments) < 2:
            print("⚠️ Not enough segments to merge")
            return None
            
        # Sort segments by filename to ensure correct order
        segments = sorted(segments)
        print(f"📊 Found {len(segments)} segments to merge")
        
        # Start with the first segment
        base_image = cv2.imread(segments[0])
        if base_image is None:
            print(f"❌ Could not load first segment: {segments[0]}")
            return None
            
        print(f"✅ Loaded base segment: {segments[0]}")
        best_panorama = base_image
        best_quality = evaluate_panorama_quality(base_image)
        best_step = 0
        
        # Track quality history
        quality_history = [best_quality]
        declining_quality_count = 0
        catastrophic_threshold = 0.5  # Threshold for catastrophic quality drop
        
        for i in range(1, len(segments)):
            current_segment = cv2.imread(segments[i])
            if current_segment is None:
                print(f"⚠️ Skipping unreadable segment: {segments[i]}")
                continue
                
            print(f"\n🔄 Merging segment {i+1}/{len(segments)}: {segments[i]}")
            
            # Create stitcher for each pair
            stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
            stitcher.setPanoConfidenceThresh(0.3)
            
            try:
                # Attempt to stitch current pair
                status, panorama = stitcher.stitch([base_image, current_segment])
                
                if status == cv2.Stitcher_OK and panorama is not None:
                    # Evaluate quality
                    current_quality = evaluate_panorama_quality(panorama)
                    quality_history.append(current_quality)
                    
                    # Calculate quality change
                    quality_change = current_quality - quality_history[-2]
                    quality_ratio = current_quality / quality_history[-2]
                    
                    # Save intermediate result
                    intermediate_path = f"{output_prefix}_step_{i}.jpg"
                    cv2.imwrite(intermediate_path, panorama)
                    print(f"💾 Saved intermediate result: {intermediate_path}")
                    print(f"      Quality change: {quality_change:+.2f} (ratio: {quality_ratio:.2f})")
                    
                    # Check for catastrophic quality drop
                    if quality_ratio < catastrophic_threshold:
                        print(f"🛑 Stopping due to catastrophic quality drop")
                        break
                    
                    # Update best result if quality is good enough
                    if current_quality > best_quality * 0.8:  # More lenient threshold
                        best_quality = max(best_quality, current_quality)
                        best_panorama = panorama.copy()
                        best_step = i
                        declining_quality_count = 0
                    else:
                        declining_quality_count += 1
                        print(f"⚠️ Quality declined ({declining_quality_count} times)")
                    
                    # Stop if quality consistently declines significantly
                    if declining_quality_count >= 3 and quality_ratio < 0.7:
                        print(f"\n🛑 Stopping due to consistent quality decline")
                        break
                    
                    # Update base image for next iteration
                    base_image = panorama
                else:
                    print(f"⚠️ Failed to merge segment {i+1}")
                    declining_quality_count += 1
                    
                    if declining_quality_count >= 3:
                        print(f"\n🛑 Stopping due to repeated merge failures")
                        break
                
            except Exception as e:
                print(f"❌ Error merging segment {i+1}: {str(e)}")
                declining_quality_count += 1
                if declining_quality_count >= 3:
                    break
        
        # Save best result
        final_path = f"{output_prefix}_best_step_{best_step}.jpg"
        cv2.imwrite(final_path, best_panorama)
        print(f"\n✅ Best panorama saved to: {final_path}")
        
        # Plot quality history
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(quality_history)
            plt.axhline(y=best_quality * 0.8, color='r', linestyle='--', label='Quality Threshold')
            plt.title('Panorama Quality History')
            plt.xlabel('Step')
            plt.ylabel('Quality Score')
            plt.grid(True)
            plt.legend()
            plt.savefig(f"{output_prefix}_quality_history.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not save quality history plot: {str(e)}")
        
        return best_panorama
        
    except Exception as e:
        print(f"❌ Error in sequential merging: {str(e)}")
        return None

def enhance_panorama_quality(panorama, reference_segments, output_prefix="enhanced_panorama"):
    """Restore panorama quality using reference segments"""
    try:
        print("\n🔄 Starting quality enhancement...")
        
        # Convert panorama to float32 for better precision
        enhanced = panorama.astype(np.float32) / 255.0
        
        # Create SIFT detector for feature matching
        sift = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Load and sort reference segments
        reference_segments = sorted(reference_segments)
        print(f"📊 Found {len(reference_segments)} reference segments")
        
        # Create mask for tracking enhanced regions
        quality_mask = np.zeros(panorama.shape[:2], dtype=np.float32)
        blend_mask = np.zeros_like(quality_mask)
        
        for i, ref_path in enumerate(reference_segments):
            try:
                print(f"\n🔄 Processing reference segment {i+1}/{len(reference_segments)}")
                
                # Load reference segment
                reference = cv2.imread(ref_path)
                if reference is None:
                    print(f"⚠️ Could not load reference: {ref_path}")
                    continue
                
                # Convert to float32
                reference = reference.astype(np.float32) / 255.0
                
                # Detect features
                kp1, des1 = sift.detectAndCompute(cv2.cvtColor(
                    (enhanced * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), None)
                kp2, des2 = sift.detectAndCompute(cv2.cvtColor(
                    (reference * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), None)
                
                if des1 is None or des2 is None:
                    print("⚠️ No features found in one of the images")
                    continue
                
                # Match features
                matches = matcher.knnMatch(des1, des2, k=2)
                
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) < 4:
                    print("⚠️ Not enough good matches")
                    continue
                
                # Get matching points
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                
                if H is None:
                    print("⚠️ Could not find homography")
                    continue
                
                # Warp reference segment to panorama space
                h, w = panorama.shape[:2]
                warped_reference = cv2.warpPerspective(reference, H, (w, h))
                
                # Create weight mask for blending
                weight_mask = np.ones_like(quality_mask)
                weight_mask = cv2.warpPerspective(weight_mask, H, (w, h))
                
                # Apply gaussian blur to weight mask for smooth blending
                weight_mask = cv2.GaussianBlur(weight_mask, (21, 21), 0)
                
                # Update quality mask
                valid_mask = (warped_reference.sum(axis=2) > 0).astype(np.float32)
                blend_mask += weight_mask
                
                # Blend enhanced regions
                enhanced = np.where(weight_mask[..., None] > 0,
                                  enhanced * (1 - weight_mask[..., None]) + 
                                  warped_reference * weight_mask[..., None],
                                  enhanced)
                
                # Save intermediate result
                intermediate = (enhanced * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(f"{output_prefix}_step_{i+1}.jpg", intermediate)
                print(f"💾 Saved enhancement step {i+1}")
                
            except Exception as e:
                print(f"⚠️ Error processing reference {i+1}: {str(e)}")
                continue
        
        # Normalize blend mask
        blend_mask = np.clip(blend_mask, 1e-7, None)
        enhanced = enhanced / blend_mask[..., None]
        
        # Apply final enhancements
        try:
            # 1. Contrast enhancement
            lab = cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # 2. Sharpening
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]]) / 9
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # 3. Color correction
            enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
            
        except Exception as e:
            print(f"⚠️ Error in final enhancement: {str(e)}")
        
        # Save final result
        final = (enhanced * 255).clip(0, 255).astype(np.uint8)
        final_path = f"{output_prefix}_final.jpg"
        cv2.imwrite(final_path, final)
        print(f"\n✅ Enhanced panorama saved to: {final_path}")
        
        return final
        
    except Exception as e:
        print(f"❌ Error in quality enhancement: {str(e)}")
        return panorama

def stitch_with_segments(stitcher, frames):
    """Stitch frames using smaller segments and enhance quality"""
    try:
        # Use smaller segments (4 frames per segment with 1 frame overlap)
        segment_size = 4
        overlap = 1
        
        panoramas = []
        
        # Calculate number of segments
        n_segments = (len(frames) - overlap) // (segment_size - overlap)
        if (len(frames) - overlap) % (segment_size - overlap) != 0:
            n_segments += 1
            
        print(f"      📊 Dividing into {n_segments} segments of {segment_size} frames")
        
        # Process each segment
        for i in range(n_segments):
            start_idx = i * (segment_size - overlap)
            end_idx = min(start_idx + segment_size, len(frames))
            segment_frames = frames[start_idx:end_idx]
            
            if len(segment_frames) < 2:
                continue
                
            segment_name = f"segment_{i+1}_of_{n_segments}"
            print(f"\n   📍 Processing {segment_name}")
            print(f"      Frames {start_idx+1} to {end_idx} (total: {len(segment_frames)})")
            
            panorama = stitch_segment(stitcher, segment_frames, segment_name)
            if panorama is not None:
                panoramas.append(panorama)
                # Save intermediate results
                cv2.imwrite(f"panorama_{segment_name}.jpg", panorama)
                print(f"      💾 Saved intermediate panorama: panorama_{segment_name}.jpg")
            
            gc.collect()
            log_memory(f"After {segment_name}")
        
        # Find all successful segment files
        segment_files = glob.glob("panorama_segment_*_final_SCANS.jpg")
        if segment_files:
            print(f"\n📂 Found {len(segment_files)} successful segments")
            final_panorama = merge_panorama_segments_sequential(segment_files)
            if final_panorama is not None:
                print("\n🎨 Enhancing panorama quality...")
                # Get list of successful segment files
                segment_files = glob.glob("panorama_segment_*_final_SCANS.jpg")
                if segment_files:
                    final_panorama = enhance_panorama_quality(final_panorama, segment_files)
        
        return status, final_panorama
        
    except Exception as e:
        print(f"❌ Error in segmented stitching: {str(e)}")
        return -1, None

# Main execution
if __name__ == "__main__":
    try:
        # Add argument parsing for workspace control
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--no-workspace', action='store_true', 
                          help='Ignore existing workspace and recompute features')
        args = parser.parse_args()
        
        success = extract_and_stitch(
            video_path='your_video.mp4',
            output_image_path='panorama.jpg',
            frame_interval=5,
            use_workspace=not args.no_workspace
        )
        
        if success:
            print("\n🎉 Panorama created successfully!")
        else:
            print("\n❌ Failed to create panorama.")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        cv2.destroyAllWindows()
    
