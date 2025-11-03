"""Video augmentation operations for sim2real"""

import numpy as np
import cv2
from typing import Optional, Tuple
from scipy import ndimage
from skimage.util import random_noise


class VideoAugmentor:
    """Apply photometric, geometric, and temporal augmentations to video frames"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    # ==================== PHOTOMETRIC AUGMENTATIONS ====================
    
    def add_gaussian_noise(
        self, 
        frame: np.ndarray, 
        sigma: float = 0.01
    ) -> np.ndarray:
        """Add Gaussian noise"""
        noise = self.rng.normal(0, sigma, frame.shape)
        noisy = np.clip(frame.astype(np.float32) / 255.0 + noise, 0, 1)
        return (noisy * 255).astype(np.uint8)
    
    def add_shot_noise(
        self, 
        frame: np.ndarray, 
        k: float = 0.01
    ) -> np.ndarray:
        """Add signal-dependent shot noise (Poisson-like)"""
        frame_float = frame.astype(np.float32) / 255.0
        # Shot noise variance proportional to signal intensity
        noise_std = np.sqrt(k * frame_float)
        noise = self.rng.normal(0, 1, frame.shape) * noise_std
        noisy = np.clip(frame_float + noise, 0, 1)
        return (noisy * 255).astype(np.uint8)
    
    def adjust_brightness(
        self, 
        frame: np.ndarray, 
        factor: float = 1.0
    ) -> np.ndarray:
        """Adjust brightness (factor: 0.5=darker, 2.0=brighter)"""
        adjusted = frame.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def adjust_contrast(
        self, 
        frame: np.ndarray, 
        factor: float = 1.0
    ) -> np.ndarray:
        """Adjust contrast (factor: 0.5=lower, 2.0=higher)"""
        mean = frame.mean()
        adjusted = (frame.astype(np.float32) - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def adjust_saturation(
        self, 
        frame: np.ndarray, 
        factor: float = 1.0
    ) -> np.ndarray:
        """Adjust saturation in HSV space"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    def adjust_hue(
        self, 
        frame: np.ndarray, 
        shift: float = 0.0
    ) -> np.ndarray:
        """Shift hue in HSV space (shift in degrees, -180 to 180)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift / 2) % 180  # OpenCV uses 0-179 for hue
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    def adjust_gamma(
        self, 
        frame: np.ndarray, 
        gamma: float = 1.0
    ) -> np.ndarray:
        """Apply gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in range(256)]).astype(np.uint8)
        return cv2.LUT(frame, table)
    
    def adjust_white_balance(
        self, 
        frame: np.ndarray, 
        r_gain: float = 1.0, 
        g_gain: float = 1.0, 
        b_gain: float = 1.0
    ) -> np.ndarray:
        """Adjust white balance by scaling RGB channels"""
        adjusted = frame.astype(np.float32)
        adjusted[:, :, 0] *= r_gain
        adjusted[:, :, 1] *= g_gain
        adjusted[:, :, 2] *= b_gain
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def apply_color_jitter(
        self,
        frame: np.ndarray,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0
    ) -> np.ndarray:
        """Combined color jitter (all params as delta ranges)"""
        result = frame.copy()
        
        if brightness != 0:
            b_factor = 1.0 + self.rng.uniform(-brightness, brightness)
            result = self.adjust_brightness(result, b_factor)
        
        if contrast != 0:
            c_factor = 1.0 + self.rng.uniform(-contrast, contrast)
            result = self.adjust_contrast(result, c_factor)
        
        if saturation != 0:
            s_factor = 1.0 + self.rng.uniform(-saturation, saturation)
            result = self.adjust_saturation(result, s_factor)
        
        if hue != 0:
            h_shift = self.rng.uniform(-hue, hue)
            result = self.adjust_hue(result, h_shift)
        
        return result
    
    # ==================== BLUR & OPTICS ====================
    
    def apply_motion_blur(
        self, 
        frame: np.ndarray, 
        kernel_size: int = 9,
        angle: Optional[float] = None
    ) -> np.ndarray:
        """Apply motion blur with optional angle (degrees)"""
        if angle is None:
            angle = self.rng.uniform(0, 360)
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0
        kernel = kernel / kernel_size
        
        # Rotate kernel
        center = (kernel_size // 2, kernel_size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        
        return cv2.filter2D(frame, -1, kernel)
    
    def apply_defocus_blur(
        self, 
        frame: np.ndarray, 
        radius: int = 5
    ) -> np.ndarray:
        """Apply defocus (disk) blur"""
        kernel_size = radius * 2 + 1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        kernel = mask.astype(np.float32)
        kernel = kernel / kernel.sum()
        
        return cv2.filter2D(frame, -1, kernel)
    
    def apply_gaussian_blur(
        self, 
        frame: np.ndarray, 
        sigma: float = 1.0
    ) -> np.ndarray:
        """Apply Gaussian blur"""
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    
    def apply_lens_distortion(
        self, 
        frame: np.ndarray, 
        k1: float = 0.0, 
        k2: float = 0.0
    ) -> np.ndarray:
        """Apply radial lens distortion (barrel/pincushion)"""
        h, w = frame.shape[:2]
        
        # Camera matrix (simplified, centered)
        fx = fy = max(h, w)
        cx, cy = w / 2, h / 2
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        dist_coeffs = np.array([k1, k2, 0, 0, 0])
        
        return cv2.undistort(frame, camera_matrix, dist_coeffs)
    
    def apply_chromatic_aberration(
        self, 
        frame: np.ndarray, 
        shift_r: int = 2, 
        shift_b: int = -2
    ) -> np.ndarray:
        """Apply chromatic aberration (shift red/blue channels)"""
        result = frame.copy()
        
        if shift_r != 0:
            result[:, :, 0] = np.roll(frame[:, :, 0], shift_r, axis=1)
        
        if shift_b != 0:
            result[:, :, 2] = np.roll(frame[:, :, 2], shift_b, axis=1)
        
        return result
    
    def apply_vignetting(
        self, 
        frame: np.ndarray, 
        strength: float = 0.5
    ) -> np.ndarray:
        """Apply vignetting effect"""
        h, w = frame.shape[:2]
        
        # Create radial mask
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_max = np.sqrt(cx**2 + cy**2)
        
        mask = 1 - (r / r_max) ** 2 * strength
        mask = np.clip(mask, 0, 1)
        
        # Apply to all channels
        result = frame.astype(np.float32) * mask[:, :, np.newaxis]
        return np.clip(result, 0, 255).astype(np.uint8)
    
    # ==================== COMPRESSION & ARTIFACTS ====================
    
    def apply_jpeg_compression(
        self, 
        frame: np.ndarray, 
        quality: int = 75
    ) -> np.ndarray:
        """Simulate JPEG compression artifacts"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    
    def add_fixed_pattern_noise(
        self, 
        frame: np.ndarray, 
        strength: float = 0.02
    ) -> np.ndarray:
        """Add fixed pattern noise (sensor artifacts)"""
        h, w = frame.shape[:2]
        
        # Generate fixed pattern (same for all frames if seed is fixed)
        pattern = self.rng.normal(0, strength, (h, w, 3))
        
        noisy = frame.astype(np.float32) / 255.0 + pattern
        return np.clip(noisy * 255, 0, 255).astype(np.uint8)
    
    # ==================== TEMPORAL EFFECTS ====================
    
    def add_temporal_flicker(
        self, 
        frame: np.ndarray, 
        intensity: float = 0.1, 
        frame_idx: int = 0
    ) -> np.ndarray:
        """Add temporal brightness flicker"""
        # Sinusoidal flicker pattern
        flicker = 1.0 + intensity * np.sin(frame_idx * 0.3)
        return self.adjust_brightness(frame, flicker)
    
    # ==================== COMBINED PIPELINES ====================
    
    def apply_all(
        self,
        frame: np.ndarray,
        params: dict
    ) -> np.ndarray:
        """Apply all augmentations based on params dict"""
        result = frame.copy()
        
        # Photometric
        if params.get("gaussian_noise", 0) > 0:
            result = self.add_gaussian_noise(result, params["gaussian_noise"])
        
        if params.get("shot_noise", 0) > 0:
            result = self.add_shot_noise(result, params["shot_noise"])
        
        # Color jitter
        if any(params.get(k, 0) != 0 for k in ["brightness", "contrast", "saturation", "hue"]):
            result = self.apply_color_jitter(
                result,
                brightness=params.get("brightness", 0),
                contrast=params.get("contrast", 0),
                saturation=params.get("saturation", 0),
                hue=params.get("hue", 0)
            )
        
        # Gamma
        if params.get("gamma", 1.0) != 1.0:
            result = self.adjust_gamma(result, params["gamma"])
        
        # White balance
        if params.get("wb_r", 1.0) != 1.0 or params.get("wb_g", 1.0) != 1.0 or params.get("wb_b", 1.0) != 1.0:
            result = self.adjust_white_balance(
                result,
                params.get("wb_r", 1.0),
                params.get("wb_g", 1.0),
                params.get("wb_b", 1.0)
            )
        
        # Blur
        if params.get("motion_blur", 0) > 0:
            result = self.apply_motion_blur(result, int(params["motion_blur"]))
        
        if params.get("defocus_blur", 0) > 0:
            result = self.apply_defocus_blur(result, int(params["defocus_blur"]))
        
        if params.get("gaussian_blur", 0) > 0:
            result = self.apply_gaussian_blur(result, params["gaussian_blur"])
        
        # Optics
        if params.get("lens_k1", 0) != 0 or params.get("lens_k2", 0) != 0:
            result = self.apply_lens_distortion(
                result,
                params.get("lens_k1", 0),
                params.get("lens_k2", 0)
            )
        
        if params.get("chromatic_aberration", 0) != 0:
            shift = int(params["chromatic_aberration"])
            result = self.apply_chromatic_aberration(result, shift, -shift)
        
        if params.get("vignetting", 0) > 0:
            result = self.apply_vignetting(result, params["vignetting"])
        
        # Compression
        if params.get("jpeg_quality", 100) < 100:
            result = self.apply_jpeg_compression(result, int(params["jpeg_quality"]))
        
        # Fixed pattern noise
        if params.get("fixed_pattern_noise", 0) > 0:
            result = self.add_fixed_pattern_noise(result, params["fixed_pattern_noise"])
        
        # Temporal flicker
        if params.get("flicker", 0) > 0:
            frame_idx = params.get("frame_idx", 0)
            result = self.add_temporal_flicker(result, params["flicker"], frame_idx)
        
        return result

