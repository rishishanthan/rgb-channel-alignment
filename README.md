# RGB Channel Alignment

This project reconstructs full-color RGB images from grayscale photographs where the Red, Green, and Blue channels are stacked vertically.  
The implementation uses **ORB (Oriented FAST and Rotated BRIEF)** feature matching and **homography transformation** in OpenCV to align channels and correct misalignments, producing high-quality reconstructed images.

---

## 🎯 Objectives
- Split grayscale input (with 3 stacked channels) into individual **B**, **G**, and **R** channels.
- Use **feature-based alignment (ORB + Homography with RANSAC)** to correct channel misalignments.
- Warp and reconstruct aligned channels into a single RGB image.
- Save and visualize the reconstructed image.

---

## 🛠️ Implementation Details

### 1. **Splitting Channels**
Each raw input consists of three stacked grayscale images:
- Top third → Blue channel
- Middle third → Green channel
- Bottom third → Red channel

### 1. **Splitting Channels**
Each raw input consists of three stacked grayscale images:
- Top third → Blue channel
- Middle third → Green channel
- Bottom third → Red channel

### 2. Channel Alignment
- ORB (Oriented FAST and Rotated BRIEF) is used to detect and describe keypoints in each channel.
- BFMatcher with Hamming distance finds correspondences between feature descriptors.
- A homography is estimated with RANSAC to robustly align channels.

### 3. Reconstruction
- Red and Blue are aligned relative to Green (chosen as reference).
- Channels are stacked to form an RGB image.

### 4. Command-Line Interface
- The program accepts --input (raw stacked image) and --output (path to save aligned result).
- It now automatically handles directories or missing extensions.

### 5. Final Script Usage
Run the script as:
python3 untitled.py \
  --input /tests/test_image_1.jpg \
  --output /results/test_1_aligned.jpg
  
## ✅ Elample Result
![Input (stacked grayscale plates)](/tests/test_image_1.jpg)
![Output (aligned RGB image)](/results/test_1_aligned.jpg)

## ⚙️ Dependencies
- Python 3.x
- OpenCV ≥ 4.5.4
- NumPy
- Matplotlib
### Install with:
    pip install opencv-python numpy matplotlib

## 🔑 Key Takeaways
- This project demonstrates how early color photography relied on careful channel alignment.
- ORB + Homography with RANSAC gives robust alignment even with perspective distortions.
- Results show good alignment, though extreme motion blur or damaged plates may need advanced techniques (e.g., pyramid search, cross-correlation).

## 📌 Future Improvements
- Implement pyramid alignment (multi-scale search).
- Add automatic cropping to remove misaligned borders.
- Explore mutual information alignment for robustness.

## 📖 References
- S. Prokudin-Gorskii Photo Collection, Library of Congress
- OpenCV ORB and Homography Documentation
