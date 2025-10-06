import eerfile
import mrcfile
import numpy as np
from tifffile import TiffFile

#eer reader
eer_file = "xenon_131_000_0.0.eer"
total_fluence = 54.75
dose_per_frame = 1.0

gain_file = "20240627_210834_EER_GainReference.gain"
flip_gain = 0
rot_gain = 0

output_file = "xenon_131_000_0.0_gain_corrected_hot_pixel_removed.mrc"

def load_gain_reference(gain_file: str) -> np.ndarray:
    """
    Load gain reference from either .mrc or .gain (TIFF) file.
    
    Args:
        gain_file: Path to gain reference file (.mrc or .gain)
        
    Returns:
        2D numpy array containing the gain reference
    """
    if gain_file.lower().endswith('.gain'):
        print(f"Loading gain reference from TIFF file: {gain_file}")
        # .gain files are TIFF format
        with TiffFile(gain_file) as tif:
            gain_map = tif.asarray().astype(np.float32)
    elif gain_file.lower().endswith('.mrc'):
        print(f"Loading gain reference from MRC file: {gain_file}")
        with mrcfile.open(gain_file) as f:
            gain_map = f.data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported gain file format: {gain_file}. Only .mrc and .gain files are supported.")
    
    return gain_map

def gain_correct(
        movie : np.ndarray, 
        gain_file : str, 
        flip_gain : int, 
        rot_gain : int,
        multiply_gain: bool = True
) -> np.ndarray:
    """
    Apply gain correction to movie frames.
    
    Args:
        movie: Movie array with shape (n_frames, height, width)
        gain_file: Path to gain reference file (.mrc or .gain)
        flip_gain: Flip gain map (0=no flip, 1=flipY, 2=flipX)
        rot_gain: Rotate gain map (number of 90-degree rotations)
        multiply_gain: Whether to multiply the gain map by the movie
    Returns:
        Gain-corrected movie array
    """
    # Load gain map (handles both .mrc and .gain files)
    gain_map = load_gain_reference(gain_file)
    
    # Apply transformations to gain map
    if flip_gain == 1:
        gain_map = np.flip(gain_map, axis=0)  # flipY
    elif flip_gain == 2:
        gain_map = np.flip(gain_map, axis=1)  # flipX
    
    if rot_gain != 0:
        gain_map = np.rot90(gain_map, k=-rot_gain)

    return movie * gain_map if multiply_gain else movie / gain_map

def remove_hot_pixels(movie: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    """
    Remove hot pixels from movie frames by replacing pixels that are more than 
    threshold standard deviations above OR below the mean with a random adjacent pixel value.
    
    Args:
        movie: Movie array with shape (n_frames, height, width)
        threshold: Number of standard deviations above/below mean to consider as hot pixel
        
    Returns:
        Movie array with hot pixels replaced
    """
    print(f"Removing hot pixels with threshold {threshold} standard deviations...")
    
    movie_corrected = movie.copy()
    n_frames, height, width = movie.shape
    
    for frame_idx in range(n_frames):
        frame = movie_corrected[frame_idx]
        
        # Calculate mean and std for this frame
        # Take mean and std from the middle half of the image
        h, w = frame.shape
        y_start = h // 4
        y_end = y_start + h // 2
        x_start = w // 4
        x_end = x_start + w // 2
        frame_center = frame[y_start:y_end, x_start:x_end]
        frame_mean = np.mean(frame_center)
        frame_std = np.std(frame_center)
        
        # Find hot pixels (pixels above OR below threshold * std from mean)
        hot_pixel_mask = (frame > (frame_mean + threshold * frame_std)) | (frame < (frame_mean - threshold * frame_std))
        hot_pixel_coords = np.where(hot_pixel_mask)
        
        if len(hot_pixel_coords[0]) > 0:
            print(f"  Frame {frame_idx}: Found {len(hot_pixel_coords[0])} hot pixels")
            
            # Replace each hot pixel with a random adjacent pixel
            for y, x in zip(hot_pixel_coords[0], hot_pixel_coords[1]):
                # Define the 8-connected neighborhood bounds
                y_min = max(0, y - 1)
                y_max = min(height - 1, y + 1)
                x_min = max(0, x - 1)
                x_max = min(width - 1, x + 1)
                
                # Get adjacent pixels (excluding the hot pixel itself)
                adjacent_pixels = []
                for adj_y in range(y_min, y_max + 1):
                    for adj_x in range(x_min, x_max + 1):
                        if adj_y != y or adj_x != x:  # Exclude the hot pixel itself
                            adjacent_pixels.append(frame[adj_y, adj_x])
                
                # Replace with random adjacent pixel value
                if adjacent_pixels:
                    replacement_value = np.random.choice(adjacent_pixels)
                    movie_corrected[frame_idx, y, x] = replacement_value
    
    return movie_corrected

def main():
    #render movie
    print(f"Rendering movie {eer_file}...")
    movie = eerfile.render(eer_file, dose_per_output_frame=dose_per_frame, total_fluence=total_fluence)
    #gain correct
    print(f"Gain correcting movie {eer_file}...")
    movie = gain_correct(movie, gain_file, flip_gain, rot_gain)
    
    #remove hot pixels
    print(f"Removing hot pixels from movie {eer_file}...")
    movie = remove_hot_pixels(movie, threshold=10.0)

    #save movie
    print(f"Saving movie to {output_file}")
    with mrcfile.new(output_file, overwrite=True) as f:
        f.set_data(movie.astype(np.float32))


if __name__ == "__main__":
    main()