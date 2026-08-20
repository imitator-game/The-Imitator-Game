"""
Frame samplers for video arrays (numpy arrays) instead of file paths.
"""

import random
import numpy as np
from xskill.dataset.frame_samplers import FrameSampler


class VideoArrayFrameSampler(FrameSampler):
    """
    Base frame sampler for video arrays (numpy arrays).
    Unlike the original FrameSampler which loads frames from disk,
    this sampler works directly with video arrays in memory.
    """
    
    def _load_frames(self, video_array):
        """
        For video arrays, we don't need to load frames from disk.
        Just return the video array itself.
        
        Args:
            video_array: Numpy array of shape (T, H, W, C)
            
        Returns:
            The video array
        """
        return video_array
    
    def sample(self, video_array):
        """
        Sample frames from a video array.
        
        Args:
            video_array: Numpy array of shape (T, H, W, C)
            
        Returns:
            A dict containing:
                - frames: The video array
                - frame_idxs: Indices of sampled frames
                - vid_len: Total number of frames in video
                - ctx_idxs: Same as frame_idxs (no context frames)
        """
        # For video arrays, frames is just the array itself
        frames = video_array
        frame_idxs = self._sample(frames)
        return {
            "frames": frames,
            "frame_idxs": frame_idxs,
            "vid_len": len(frames),
            "ctx_idxs": frame_idxs,  # No context frames, just use the same indices
        }


class UniformVideoArraySampler(VideoArrayFrameSampler):
    """
    Uniformly sample video frames from a video array, starting from an optional offset.
    """
    
    def __init__(self, offset, *args, **kwargs):
        """
        Args:
            offset: An offset from which to start the uniform random sampling.
            *args: Additional positional arguments for FrameSampler
            **kwargs: Additional keyword arguments for FrameSampler
        """
        super().__init__(*args, **kwargs)
        assert isinstance(offset, int), "`offset` must be an integer."
        self._offset = offset
    
    def _sample(self, frames):
        """
        Sample frame indices uniformly from the video.
        
        Args:
            frames: Video array of shape (T, H, W, C)
            
        Returns:
            List of sampled frame indices
        """
        vid_len = len(frames)
        cond1 = vid_len >= self._offset
        cond2 = self._num_frames < (vid_len - self._offset)
        
        if cond1 and cond2:
            cc_idxs = list(range(self._offset, vid_len))
            random.shuffle(cc_idxs)
            cc_idxs = cc_idxs[:self._num_frames]
            return sorted(cc_idxs)
        return list(range(0, min(self._num_frames, vid_len)))


class StridedVideoArraySampler(VideoArrayFrameSampler):
    """
    Sample every n'th frame from a video array.
    """
    
    def __init__(self, stride, offset=True, *args, **kwargs):
        """
        Args:
            stride: The spacing between consecutively sampled frames.
            offset: If True, a random starting point is chosen. Else, starts at frame 0.
            *args: Additional positional arguments for FrameSampler
            **kwargs: Additional keyword arguments for FrameSampler
        """
        super().__init__(*args, **kwargs)
        assert stride >= 1, "stride must be >= 1."
        assert isinstance(stride, int), "stride must be an integer."
        
        self._offset = offset
        self._stride = stride
    
    def _sample(self, frames):
        """
        Sample frame indices with a fixed stride.
        
        Args:
            frames: Video array of shape (T, H, W, C)
            
        Returns:
            List of sampled frame indices
        """
        vid_len = len(frames)
        
        if self._offset:
            # Random offset between 0 and max location for full coverage
            offset = random.randint(
                0, max(1, vid_len - self._stride * self._num_frames)
            )
        else:
            offset = 0
        
        cc_idxs = list(
            range(
                offset,
                offset + self._num_frames * self._stride + 1,
                self._stride,
            )
        )
        cc_idxs = np.clip(cc_idxs, a_min=0, a_max=vid_len - 1)
        return cc_idxs[:self._num_frames]


class VariableStridedVideoArraySampler(VideoArrayFrameSampler):
    """
    Strided sampling based on a video's number of frames.
    The stride is automatically calculated to evenly sample across the video.
    """
    
    def _sample(self, frames):
        """
        Sample frame indices with variable stride based on video length.
        
        Args:
            frames: Video array of shape (T, H, W, C)
            
        Returns:
            List of sampled frame indices
        """
        vid_len = len(frames)
        stride = vid_len / self._num_frames
        cc_idxs = np.arange(0.0, vid_len, stride).round().astype(int)
        cc_idxs = np.clip(cc_idxs, a_min=0, a_max=vid_len - 1)
        cc_idxs = cc_idxs[:self._num_frames]
        return cc_idxs


class WindowVideoArraySampler(VideoArrayFrameSampler):
    """
    Sample a contiguous window of frames from a video array.
    """
    
    def _sample(self, frames):
        """
        Sample a contiguous window of frames.
        
        Args:
            frames: Video array of shape (T, H, W, C)
            
        Returns:
            List of sampled frame indices
        """
        vid_len = len(frames)
        
        if vid_len > self._num_frames:
            range_min = random.randrange(vid_len - self._num_frames)
            range_max = range_min + self._num_frames
            return list(range(range_min, range_max))
        return list(range(0, min(self._num_frames, vid_len)))


class AllVideoArraySampler(VideoArrayFrameSampler):
    """
    Sample all frames from a video array (with optional stride).
    Useful for evaluation when you want to process the entire video.
    """
    
    def __init__(self, stride=1, *args, **kwargs):
        """
        Args:
            stride: The spacing between consecutively sampled frames.
                    A stride of 1 samples all frames.
            *args: Additional positional arguments for FrameSampler
            **kwargs: Additional keyword arguments for FrameSampler
        """
        kwargs["num_frames"] = 1  # Will be overridden in _sample
        super().__init__(*args, **kwargs)
        self._stride = stride
    
    def _sample(self, frames):
        """
        Sample all frames (or every stride'th frame).
        
        Args:
            frames: Video array of shape (T, H, W, C)
            
        Returns:
            List of sampled frame indices
        """
        vid_len = len(frames)
        self._num_frames = int(np.ceil(vid_len / self._stride))
        return list(range(0, vid_len, self._stride))


class UniformDownSampleVideoArraySampler(VideoArrayFrameSampler):
    """
    Uniformly sample video frames from a downsampled video array, starting from an optional offset.
    This is the video array version of UniformDownSampleSampler from frame_samplers.py.
    """
    
    def __init__(self, downsample_ratio, offset, *args, **kwargs):
        """
        Args:
            downsample_ratio: Ratio for downsampling the video before uniform sampling.
            offset: An offset from which to start the uniform random sampling.
            *args: Additional positional arguments for FrameSampler
            **kwargs: Additional keyword arguments for FrameSampler
        """
        super().__init__(*args, **kwargs)
        assert isinstance(offset, int), "`offset` must be an integer."
        self._offset = offset
        self.downsample_ratio = downsample_ratio
    
    def _sample(self, frames):
        """
        Sample frame indices uniformly from a downsampled video.
        
        Args:
            frames: Video array of shape (T, H, W, C)
            
        Returns:
            List of sampled frame indices from the original video
        """
        # Create downsampled frame indices
        downsample_frames = (np.arange(int(len(frames) * self.downsample_ratio)) / self.downsample_ratio).astype(np.int32)
        vid_len = len(downsample_frames)
        
        cond1 = vid_len >= self._offset
        cond2 = self._num_frames < (vid_len - self._offset)
        
        if cond1 and cond2:
            cc_idxs = list(range(self._offset, vid_len))
            random.shuffle(cc_idxs)
            cc_idxs = cc_idxs[:self._num_frames]
            downsample_cc_idxs = sorted(cc_idxs)
        else:
            downsample_cc_idxs = list(range(0, self._num_frames))
            downsample_cc_idxs = np.clip(downsample_cc_idxs, a_min=0, a_max=vid_len - 1)

        # Return the actual frame indices from the original video
        return [int(downsample_frames[i]) for i in downsample_cc_idxs]