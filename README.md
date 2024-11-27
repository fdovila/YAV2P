# YAV2P: Yet Another Video To Panorama Stitcher

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.5+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![GitHub issues](https://img.shields.io/github/issues/fdovila/YAV2P.svg)](https://github.com/fdovila/YAV2P/issues)
[![GitHub stars](https://img.shields.io/github/stars/fdovila/YAV2P.svg)](https://github.com/fdovila/YAV2P/stargazers)
![Luck](https://img.shields.io/badge/luck-needed-yellow)
![Quality](https://img.shields.io/badge/quality-ish-blueviolet)
![Humour](https://img.shields.io/badge/humour-dry-red)
![Docs](https://img.shields.io/badge/docs-sarcastic-lightgrey)

A *slightly ambitious* attempt at yet another video panorama stitching tool that extracts frames from video and (with varying degrees of success) tries to create high-quality panoramic images. Designed to handle challenging conditions including dark videos, motion blur, and varying exposure levels - though "handle" might be a rather optimistic term. Specially designed for the British weather.

Because the world definitely needed another panorama stitcher, this one includes expectations management while still doing its best to deliver something that might, under favorable circumstances and with a bit of luck, resemble a proper panorama.

*Note: Results may vary. Significantly.*

## 🌟 Features

- **Adaptive Frame Extraction**
  - Quality-aware frame selection
  - Motion blur detection
  - Exposure validation

- **Advanced Image Enhancement**
  - Multi-stage enhancement pipeline (CLAHE, histogram equalization, gamma correction)
  - Dark video optimization

- **Robust Stitching**
  - Memory-efficient parallel processing
  - Segmented stitching for large panoramas
  - Quality-aware sequential merging
  - Automatic error recovery and alternative stitching approaches

- **Performance Optimizations**
  - Workspace management for long-running operations
  - Memory usage monitoring
  - Multi-threaded processing
  - Detailed progress tracking

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

### Required Python Packages
- OpenCV (cv2)
- NumPy
- tqdm (for progress bars)
- psutil (for memory management)

### System Requirements
- Operating System: Windows 10+, macOS 10.14+, or Linux
- Minimum RAM: 4GB (more recommended for large videos)
- Storage: 500MB free space (more needed for workspace and output)

### Development Environment Setup

1. Clone the repository
```bash
git clone https://github.com/fdovila/YAV2P.git
cd YAV2P
```

2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install opencv-python numpy tqdm psutil
```

4. Verify installation
```bash
python --version
python -c "import cv2, numpy, tqdm, psutil; print('All packages installed successfully')"
```

## 📁 Working with Video Files

### How to Use

1. Place your video file in the project root as `your_video.mp4`, or modify the `video_path` in main.py:
```python
success = extract_and_stitch(
    video_path='path/to/your/video.mp4',
    output_image_path='panorama.jpg',
    frame_interval=5,
    use_workspace=True
)
```

2. Run the script:
```bash
# Use existing workspace (if available)
python main.py

# Ignore workspace and recompute features
python main.py --no-workspace
```

The script will:
1. Extract frames from your video
2. Process them using SIFT feature detection
3. Attempt stitching using multiple approaches
4. Save the final panorama as 'panorama.jpg'

### Supported Video Formats
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)

### Common Video Path Issues
- ❌ Don't use: `C:\My Videos\video.mp4` or `/home/user/videos/video.mp4`
- ✅ Do use: Relative paths from the project root, e.g., `your_video.mp4`
- ✅ Make sure your video file exists and has the correct permissions
- ✅ Check that the video format is supported

### Troubleshooting Video Loading
If your video isn't being recognized:
1. Confirm the file exists in the correct location
2. Verify the file permissions
3. Check the file format is supported
4. Try using the absolute path (if all else fails)

## 📧 Advanced Usage

### Workspace Management
The script uses a workspace to save intermediate results. This can speed up subsequent runs on the same video. Use the `--no-workspace` flag to ignore existing workspace and recompute all features.

### Memory Optimization
The script includes memory management features to handle large videos. It uses batch processing and garbage collection to minimize memory usage.

### Alternative Stitching Methods
If the initial stitching fails, the script will attempt alternative methods, including different warpers (SPHERICAL, CYLINDRICAL, PLANE, AFFINE) and image scaling.

### Quality Enhancement
The final panorama undergoes a quality enhancement process, which includes contrast enhancement, sharpening, and color correction.

## 📊 Output

The script generates several output files:
- `panorama.jpg`: The final stitched panorama
- Intermediate panoramas for each processing step
- Quality history plot (if matplotlib is installed)

## 📧 Contact

- GitHub: [@fdovila](https://github.com/fdovila)
- Email: favila[at]gmail[dot]com

## 🔑 Keywords
Computer Vision, Image Processing, Panorama Stitching, OpenCV, SIFT, 
Feature Detection, Parallel Processing, Image Enhancement, 
Memory Management, Error Recovery, Quality Control
