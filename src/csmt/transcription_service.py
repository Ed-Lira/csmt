"""
Service for handling audio transcription using Silero VAD and Faster-Whisper.
"""
import torch
import torchaudio
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Tuple, List
import asyncio
from dataclasses import dataclass
import queue

@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    text: str
    is_final: bool
    start_time: float
    end_time: float

class TranscriptionService:
    def __init__(self, 
                 whisper_model_size: str = "base",
                 device: str = "cpu",
                 language: str = "en",
                 vad_threshold: float = 0.5,
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 100):
        """Initialize the transcription service.
        
        Args:
            whisper_model_size: Size of the Whisper model to use (tiny, base, small, medium, large)
            device: Device to run the models on (cpu, cuda)
            language: Language code for transcription
            vad_threshold: Threshold for voice activity detection
            min_speech_duration_ms: Minimum duration of speech segments in milliseconds
            min_silence_duration_ms: Minimum duration of silence segments in milliseconds
        """
        self.device = device
        self.language = language
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        
        # Initialize Silero VAD
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True
        )
        self.vad_model = self.vad_model.to(device)
        self.get_speech_timestamps = utils[0]
        self.save_audio = utils[2]
        self.read_audio = utils[3]
        
        # Initialize Whisper
        self.whisper_model = WhisperModel(
            whisper_model_size,
            device=device,
            compute_type="int8" if device == "cpu" else "float16"
        )
        
        # Buffer for accumulating audio
        self.audio_buffer = []
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        self.buffer_duration = 30  # Buffer up to 30 seconds of audio
        
        # Queue for transcription results
        self.result_queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        
        # Start the processing task
        self.processing_task = None
        self.is_running = False

    async def start(self):
        """Start the transcription service."""
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_audio_loop())

    async def stop(self):
        """Stop the transcription service."""
        self.is_running = False
        if self.processing_task:
            await self.processing_task
            self.processing_task = None
        self.audio_buffer = []

    def add_audio(self, audio_data: np.ndarray, sample_rate: int):
        """Add audio data to the buffer.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio data
        """
        # Ensure audio is mono (single channel)
        if len(audio_data.shape) > 1:
            # If stereo, take the mean of both channels
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if necessary
        if sample_rate != self.sample_rate:
            audio_data = self._resample_audio(audio_data, sample_rate, self.sample_rate)
        
        # Add to buffer
        self.audio_buffer.extend(audio_data.tolist())
        
        # Trim buffer if it's too long
        max_samples = int(self.buffer_duration * self.sample_rate)
        if len(self.audio_buffer) > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]
        
        print(f"Added audio data to buffer. Buffer size: {len(self.audio_buffer)} samples")

    async def get_transcription(self) -> Optional[TranscriptionResult]:
        """Get the next transcription result."""
        try:
            return await asyncio.wait_for(self.result_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        audio_tensor = torch.from_numpy(audio).float()
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        return resampler(audio_tensor).numpy()

    async def _process_audio_loop(self):
        """Main processing loop for audio transcription."""
        while self.is_running:
            if not self.audio_buffer:
                await asyncio.sleep(0.1)
                continue

            # Convert buffer to numpy array
            audio_data = np.array(self.audio_buffer)
            print(f"Processing audio buffer of size: {len(audio_data)} samples")
            
            # Get speech timestamps using Silero VAD
            speech_timestamps = self._detect_speech(audio_data)
            print(f"Detected {len(speech_timestamps)} speech segments")
            
            if speech_timestamps:
                # Process each speech segment
                for start, end in speech_timestamps:
                    segment = audio_data[start:end]
                    print(f"Processing speech segment of size: {len(segment)} samples")
                    
                    # Transcribe the segment
                    result = self._transcribe_audio(segment)
                    if result:
                        print(f"Got transcription: {result.text}")
                        await self.result_queue.put(result)
                    else:
                        print("No transcription result for segment")
                
                # Clear processed audio from buffer
                if speech_timestamps:
                    last_end = speech_timestamps[-1][1]
                    self.audio_buffer = self.audio_buffer[last_end:]
                    print(f"Cleared buffer up to {last_end} samples. Remaining: {len(self.audio_buffer)}")
            
            await asyncio.sleep(0.1)

    def _detect_speech(self, audio_data: np.ndarray) -> List[Tuple[int, int]]:
        """Detect speech segments in audio using Silero VAD."""
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_data).float()
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.vad_model,
            threshold=self.vad_threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms
        )
        
        timestamps = [(ts['start'], ts['end']) for ts in speech_timestamps]
        if timestamps:
            print(f"Speech detected at timestamps: {timestamps}")
        return timestamps

    def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[TranscriptionResult]:
        """Transcribe audio using Faster-Whisper."""
        try:
            # Run inference
            segments, _ = self.whisper_model.transcribe(
                audio_data,
                language=self.language,
                beam_size=5,
                vad_filter=False  # We're using Silero VAD instead
            )
            
            # Get the first segment (we're processing short segments)
            segment = next(segments, None)
            if segment:
                return TranscriptionResult(
                    text=segment.text,
                    is_final=True,
                    start_time=segment.start,
                    end_time=segment.end
                )
        except Exception as e:
            print(f"Error in transcription: {e}")
            import traceback
            traceback.print_exc()
        
        return None 