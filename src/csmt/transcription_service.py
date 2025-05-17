"""
Service for handling audio transcription using Silero VAD and Faster-Whisper.
Optimized for conversational turn detection and transcription.
"""
import torch
import torchaudio
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Tuple, List, Deque
import asyncio
from dataclasses import dataclass
import queue
from enum import Enum
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TurnState(Enum):
    """State of the current conversation turn."""
    SILENCE = 0
    SPEECH_START = 1
    SPEAKING = 2
    TURN_END = 3

@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    text: str
    is_final: bool
    start_time: float
    end_time: float
    turn_id: int

class EOUModel:
    """End of Utterance (EOU) model for turn detection."""
    
    def __init__(self, model_name: str = "livekit/eou-model", device: str = "cpu"):
        """Initialize the EOU model.
        
        Args:
            model_name: Name of the model on HuggingFace
            device: Device to run the model on (cpu, cuda)
        """
        logger.info(f"Loading EOU model from {model_name}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            load_in_8bit=True if device == "cpu" else False
        )
        self.context_window = 4  # Number of turns to keep in context
        self.turn_history: Deque[str] = deque(maxlen=self.context_window)
        logger.info("EOU model loaded successfully")
    
    def add_turn(self, text: str):
        """Add a turn to the history."""
        self.turn_history.append(text)
    
    def predict_turn_end(self, current_text: str) -> Tuple[bool, float]:
        """Predict if the current turn is likely to end.
        
        Args:
            current_text: The current transcription text
            
        Returns:
            Tuple of (is_turn_end, confidence)
        """
        try:
            # Prepare the context
            context = "\n".join(list(self.turn_history) + [current_text])
            
            # Tokenize input
            inputs = self.tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                
                # Get probability of end token
                end_token_id = self.tokenizer.encode("</s>")[0]
                end_prob = probs[0, end_token_id].item()
                
                # Convert to turn end prediction
                is_turn_end = end_prob > 0.5
                confidence = end_prob if is_turn_end else 1 - end_prob
                
                logger.debug(f"EOU prediction: end={is_turn_end}, confidence={confidence:.2f}")
                return is_turn_end, confidence
                
        except Exception as e:
            logger.error(f"Error in EOU prediction: {e}")
            # Fallback to basic heuristics if model fails
            return self._fallback_prediction(current_text)
    
    def _fallback_prediction(self, text: str) -> Tuple[bool, float]:
        """Fallback prediction using basic heuristics."""
        # Check for very short turns
        if len(text.split()) <= 3:
            return True, 0.9
        
        # Check for sentence endings
        if any(text.strip().endswith(p) for p in ['.', '!', '?']):
            return True, 0.7
        
        return False, 0.6

class TranscriptionService:
    def __init__(self, 
                 whisper_model_size: str = "large",
                 device: str = "cpu",
                 language: str = "en",
                 vad_threshold: float = 0.6,  # Keep this higher for noise rejection
                 min_speech_duration_ms: int = 150,  # Reduced from 250ms to catch short utterances
                 min_silence_duration_ms: int = 200,  # Reduced from 300ms for faster turn detection
                 turn_silence_threshold_ms: int = 400):  # Reduced from 500ms for more responsive turns
        """Initialize the transcription service.
        
        Args:
            whisper_model_size: Size of the Whisper model to use (tiny, base, small, medium, large)
            device: Device to run the models on (cpu, cuda)
            language: Language code for transcription
            vad_threshold: Threshold for voice activity detection
            min_speech_duration_ms: Minimum duration of speech segments in milliseconds
            min_silence_duration_ms: Minimum duration of silence segments in milliseconds
            turn_silence_threshold_ms: Duration of silence to consider a turn ended
        """
        self.device = device
        self.language = language
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.turn_silence_threshold_ms = turn_silence_threshold_ms
        
        # Initialize Silero VAD with optimized settings
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True
        )
        self.vad_model = self.vad_model.to(device)
        self.get_speech_timestamps = utils[0]
        self.save_audio = utils[2]
        self.read_audio = utils[3]
        
        # Initialize Whisper with faster settings
        self.whisper_model = WhisperModel(
            whisper_model_size,
            device=device,
            compute_type="int8" if device == "cpu" else "float16"
        )
        
        # Audio processing
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        self.audio_buffer = []
        self.turn_buffer = []  # Buffer for the current turn
        self.current_turn_state = TurnState.SILENCE
        self.turn_start_time = 0
        self.last_speech_time = 0
        self.current_turn_id = 0
        self.max_buffer_duration = 10  # Reduced buffer duration for lower latency
        
        # Queue for transcription results
        self.result_queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        
        # Start the processing task
        self.processing_task = None
        self.is_running = False
        
        # Replace semantic detector with EOU model
        self.eou_model = EOUModel(device=device)
        self.current_turn_text = ""
        self.dynamic_silence_threshold = turn_silence_threshold_ms
        self.base_silence_threshold = turn_silence_threshold_ms

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
        max_samples = int(self.max_buffer_duration * self.sample_rate)
        if len(self.audio_buffer) > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]

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
        """Main processing loop for audio transcription with turn detection."""
        print("Starting audio processing loop...")
        last_speech_time = 0
        is_speaking = False
        turn_buffer = []
        turn_start_time = 0
        current_turn_id = 0
        buffer_start_time = 0  # Track when the buffer started
        
        while self.is_running:
            if not self.audio_buffer:
                await asyncio.sleep(0.1)
                continue

            # Convert buffer to numpy array
            audio_data = np.array(self.audio_buffer)
            current_time = buffer_start_time + (len(self.audio_buffer) / self.sample_rate)
            
            # Get speech timestamps using Silero VAD
            speech_timestamps = self._detect_speech(audio_data)
            
            if speech_timestamps:
                # Get the latest speech segment
                start, end = speech_timestamps[-1]
                segment_time = buffer_start_time + (start / self.sample_rate)
                last_speech_time = buffer_start_time + (end / self.sample_rate)
                
                if not is_speaking:
                    # Start of a new turn
                    print(f"Speech detected, starting new turn at {segment_time:.2f}s")
                    is_speaking = True
                    turn_start_time = segment_time
                    turn_buffer = audio_data[start:end].tolist()
                    current_turn_id += 1
                    self.current_turn_text = ""  # Reset current turn text
                else:
                    # Continuing current turn
                    print(f"Continuing turn {current_turn_id}, adding {end-start} samples")
                    turn_buffer.extend(audio_data[start:end].tolist())
                
                # Clear processed audio
                self.audio_buffer = self.audio_buffer[end:]
                buffer_start_time = last_speech_time  # Update buffer start time after trimming
            
            # Check for turn end with EOU model
            elif is_speaking:
                silence_duration = (current_time - last_speech_time) * 1000  # in ms
                
                # Get EOU prediction if we have some text
                if self.current_turn_text:
                    is_turn_end, confidence = self.eou_model.predict_turn_end(self.current_turn_text)
                    # Adjust silence threshold based on EOU prediction
                    if not is_turn_end:
                        self.dynamic_silence_threshold = int(self.base_silence_threshold * (1 + confidence))
                    else:
                        self.dynamic_silence_threshold = int(self.base_silence_threshold * (1 - confidence * 0.5))
                else:
                    self.dynamic_silence_threshold = self.base_silence_threshold
                
                logger.info(f"Silence duration: {silence_duration:.0f}ms, Dynamic threshold: {self.dynamic_silence_threshold}ms")
                
                if silence_duration >= self.dynamic_silence_threshold:
                    print(f"Turn {current_turn_id} ended after {silence_duration:.0f}ms of silence")
                    # Process the turn
                    turn_audio = np.array(turn_buffer)
                    
                    if len(turn_audio) >= self.min_speech_duration_ms * self.sample_rate / 1000:
                        print(f"Processing turn {current_turn_id} of size: {len(turn_audio)} samples")
                        
                        # Transcribe the complete turn
                        result = self._transcribe_audio(turn_audio)
                        if result:
                            result.start_time = turn_start_time
                            result.end_time = last_speech_time
                            result.turn_id = current_turn_id
                            print(f"Got transcription for turn {current_turn_id}: {result.text}")
                            
                            # Update EOU model with the new turn
                            self.eou_model.add_turn(result.text)
                            self.current_turn_text = result.text
                            
                            await self.result_queue.put(result)
                        else:
                            print("No transcription result returned")
                    
                    # Reset for next turn
                    is_speaking = False
                    turn_buffer = []
                    self.current_turn_text = ""
                    self.dynamic_silence_threshold = self.base_silence_threshold
            
            # Trim the audio buffer if it gets too long
            max_samples = int(self.max_buffer_duration * self.sample_rate)
            if len(self.audio_buffer) > max_samples:
                trim_amount = len(self.audio_buffer) - max_samples
                self.audio_buffer = self.audio_buffer[trim_amount:]
                # Update buffer start time to maintain correct timing
                buffer_start_time += trim_amount / self.sample_rate
            
            await asyncio.sleep(0.1)

    def _detect_speech(self, audio_data: np.ndarray) -> List[Tuple[int, int]]:
        """Detect speech segments in audio using Silero VAD."""
        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_data).float()
            
            # Get speech timestamps with balanced settings
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                threshold=self.vad_threshold,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                window_size_samples=768,  # Balanced between 512 and 1024
                speech_pad_ms=40  # Balanced between 30ms and 50ms
            )
            
            timestamps = [(ts['start'], ts['end']) for ts in speech_timestamps]
            if timestamps:
                print(f"Speech detected: {len(timestamps)} segments, latest: {timestamps[-1]}")
            return timestamps
        except Exception as e:
            print(f"Error in speech detection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[TranscriptionResult]:
        """Transcribe audio using Faster-Whisper."""
        try:
            print(f"Starting transcription of {len(audio_data)} samples...")
            # Run inference with balanced settings
            segments, info = self.whisper_model.transcribe(
                audio_data,
                language=self.language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=True,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.65,  # Slightly reduced from 0.7 for better short utterance detection
                vad_filter=False,
                word_timestamps=True
            )
            
            # Get the first segment (we're processing short segments)
            segment = next(segments, None)
            if segment:
                # Log confidence scores for debugging
                print(f"Transcription successful: {segment.text}")
                print(f"Confidence: {info.language_probability:.2f}")
                if hasattr(segment, 'avg_logprob'):
                    print(f"Average log probability: {segment.avg_logprob:.2f}")
                
                # Adjusted confidence thresholds for better short utterance handling
                if (info.language_probability > 0.6 and  # Reduced from 0.7
                    (not hasattr(segment, 'avg_logprob') or segment.avg_logprob > -0.9)):  # Adjusted from -0.8
                    return TranscriptionResult(
                        text=segment.text,
                        is_final=True,
                        start_time=segment.start,
                        end_time=segment.end,
                        turn_id=self.current_turn_id
                    )
                else:
                    print("Low confidence transcription, skipping")
            else:
                print("No transcription segments returned")
        except Exception as e:
            print(f"Error in transcription: {e}")
            import traceback
            traceback.print_exc()
        
        return None 