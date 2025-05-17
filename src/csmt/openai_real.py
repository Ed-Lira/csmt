# realtime_text_only_client.py
"""A microphone‑driven client for OpenAI Realtime WebSocket API that
1. Forces **text‑only** assistant replies (no TTS)
2. Captures each spoken utterance as a WAV file
3. Prints the utterance transcript and the saved audio path

Dependencies
------------
$ pip install --upgrade openai sounddevice numpy websockets

Env vars required
-----------------
OPENAI_API_KEY            – your key
OPENAI_API_MODEL          – e.g. "gpt-4o-realtime-preview" (or Azure deployment)
OPENAI_API_BASE (optional)

Run
---
$ python realtime_text_only_client.py
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime as dt
import os
import signal
import sys
import wave
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI

# ───────────── Configuration ──────────────
MODEL: str = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini-realtime-preview")
SAMPLE_RATE = 24_000          # 24 kHz mono PCM16 as required
FRAME_MS = 20                 # 20‑ms frames
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
PRE_SPEECH_BUFFER_MS = 60     # Buffer 40ms of audio before speech detection
PRE_SPEECH_BUFFER_SIZE = int(SAMPLE_RATE * PRE_SPEECH_BUFFER_MS / 1000) * 2  # *2 for 16-bit samples
OUTPUT_DIR = Path("utterances")
OUTPUT_DIR.mkdir(exist_ok=True)

# Queue for console output to prevent thread interference
console_q: asyncio.Queue[str] = asyncio.Queue()
# Queue for audio data to prevent thread interference
audio_q: asyncio.Queue[bytes] = asyncio.Queue()

# State variables for interaction tracking
current_audio_bytes: Optional[bytes] = None
current_transcript: Optional[str] = None

# Timing variables for profiling
utterance_end_time: Optional[float] = None
transcription_start_time: Optional[float] = None
transcription_end_time: Optional[float] = None
assistant_start_time: Optional[float] = None
assistant_end_time: Optional[float] = None

async def log_interaction(audio: bytes, transcript: str, reply: str):
    """Log a complete interaction immediately."""
    await console_q.put("\n=== Complete Interaction ===\n")
    await console_q.put(f"User Audio Size: {len(audio)} bytes\n")
    await console_q.put(f"User Transcript: {transcript}\n")
    await console_q.put(f"Assistant Reply: {reply}\n")
    await console_q.put("========================\n\n")

async def console_writer():
    """Write console output from queue to prevent thread interference."""
    while True:
        msg = await console_q.get()
        print(msg, end="", flush=True)

async def log_timing_stats():
    """Log timing statistics for the current interaction."""
    if utterance_end_time is None:
        return
        
    now = time.time()
    stats = []
    
    if transcription_start_time:
        stats.append(f"Transcription start: {(transcription_start_time - utterance_end_time)*1000:.1f}ms")
    if transcription_end_time:
        stats.append(f"Transcription total: {(transcription_end_time - utterance_end_time)*1000:.1f}ms")
    if assistant_start_time:
        stats.append(f"Assistant start: {(assistant_start_time - utterance_end_time)*1000:.1f}ms")
    if assistant_end_time:
        stats.append(f"Assistant total: {(assistant_end_time - utterance_end_time)*1000:.1f}ms")
    
    if stats:
        await console_q.put("\n=== Timing Stats ===\n")
        await console_q.put("\n".join(stats) + "\n")
        await console_q.put("==================\n\n")

# ──────────── Microphone streaming ─────────
async def stream_microphone(conn):
    """Capture mic audio, stream to OpenAI, and keep local copy per utterance."""
    global current_audio_bytes, current_transcript
    
    loop = asyncio.get_running_loop()
    push_q: asyncio.Queue[str] = asyncio.Queue()
    current_audio: bytearray = bytearray()
    is_speaking: bool = False  # Track if we're currently in a speaking turn
    audio_chunks_collected: int = 0  # Debug counter
    
    # Initialize the assistant reply buffer in the outer scope
    assistant_reply_buffer: list[str] = []
    pending_assistant_reply: Optional[str] = None  # Store complete assistant reply if we get it before transcript
    
    # Circular buffer for pre-speech audio
    pre_speech_buffer = bytearray(PRE_SPEECH_BUFFER_SIZE)
    pre_speech_pos = 0

    def _callback(indata: np.ndarray, frames: int, t, status):  # noqa: N802
        nonlocal pre_speech_pos
        if status:
            loop.call_soon_threadsafe(console_q.put_nowait, f"Audio status: {status}\n")
        pcm_bytes = indata.tobytes()
        
        # Always update the pre-speech buffer
        chunk_size = len(pcm_bytes)
        if chunk_size > PRE_SPEECH_BUFFER_SIZE:
            # If chunk is larger than buffer, just keep the most recent part
            pcm_bytes = pcm_bytes[-PRE_SPEECH_BUFFER_SIZE:]
            chunk_size = PRE_SPEECH_BUFFER_SIZE
        
        # Update circular buffer
        space_left = PRE_SPEECH_BUFFER_SIZE - pre_speech_pos
        if chunk_size <= space_left:
            pre_speech_buffer[pre_speech_pos:pre_speech_pos + chunk_size] = pcm_bytes
        else:
            # Wrap around
            pre_speech_buffer[pre_speech_pos:] = pcm_bytes[:space_left]
            pre_speech_buffer[:chunk_size - space_left] = pcm_bytes[space_left:]
        pre_speech_pos = (pre_speech_pos + chunk_size) % PRE_SPEECH_BUFFER_SIZE
        
        # Only queue audio data if we're in a speaking turn
        if is_speaking:
            loop.call_soon_threadsafe(audio_q.put_nowait, pcm_bytes)
            nonlocal audio_chunks_collected
            audio_chunks_collected += 1
            if audio_chunks_collected % 50 == 0:  # Log every 50 chunks
                loop.call_soon_threadsafe(console_q.put_nowait, f"Collected {audio_chunks_collected} audio chunks\n")
        b64 = base64.b64encode(pcm_bytes).decode()
        loop.call_soon_threadsafe(push_q.put_nowait, b64)

    async def _writer():
        while True:
            chunk: str = await push_q.get()
            await conn.input_audio_buffer.append(audio=chunk)

    async def _audio_collector():
        """Collect audio data from the queue into the current buffer."""
        nonlocal audio_chunks_collected
        while True:
            chunk = await audio_q.get()
            if is_speaking:  # Double check we're still speaking
                current_audio.extend(chunk)
                if len(current_audio) % (FRAME_SAMPLES * 2 * 50) == 0:  # Log every ~1 second of audio
                    await console_q.put(f"Current audio buffer size: {len(current_audio)} bytes\n")

    # Coroutine that yields WAV when server finishes transcription
    async def _await_transcripts():
        nonlocal current_audio, is_speaking, audio_chunks_collected, assistant_reply_buffer, pending_assistant_reply
        global utterance_end_time, transcription_start_time, transcription_end_time, assistant_start_time, assistant_end_time
        
        async for event in conn:
            # Log all events for debugging
            await console_q.put(f"Event received: {event.type}\n")
            
            if event.type == "input_audio_buffer.speech_started":
                is_speaking = True
                current_audio = bytearray()  # Reset audio buffer at start of speech
                assistant_reply_buffer.clear()  # Reset assistant reply buffer
                pending_assistant_reply = None  # Reset pending reply
                current_audio_bytes = None
                current_transcript = None
                # Reset timing variables
                utterance_end_time = None
                transcription_start_time = None
                transcription_end_time = None
                assistant_start_time = None
                assistant_end_time = None
                await console_q.put("DEBUG: Reset all buffers for new interaction\n")
                
                # Add the pre-speech buffer to the start of current_audio
                if pre_speech_pos > 0:
                    current_audio.extend(pre_speech_buffer[pre_speech_pos:])
                    current_audio.extend(pre_speech_buffer[:pre_speech_pos])
                else:
                    current_audio.extend(pre_speech_buffer)
                
                audio_chunks_collected = 0
                await console_q.put("Speech started - starting audio collection\n")
                
            elif event.type == "input_audio_buffer.speech_stopped":
                utterance_end_time = time.time()
                await console_q.put(f"DEBUG: Utterance ended at {utterance_end_time}\n")
                
            elif event.type == "conversation.item.input_audio_transcription.started":
                transcription_start_time = time.time()
                await console_q.put(f"DEBUG: Transcription started at {transcription_start_time}\n")
                
            elif event.type == "conversation.item.input_audio_transcription.completed":
                transcription_end_time = time.time()
                transcript = event.transcript.strip()
                await console_q.put(f"Transcription completed: {transcript}\n")
                await console_q.put(f"DEBUG: Transcription ended at {transcription_end_time}\n")
                
                # Store the current audio and transcript
                current_audio_bytes = bytes(current_audio)
                current_transcript = transcript
                
                await console_q.put(f"You ➜ {transcript}\n  ↳ audio logged (size: {len(current_audio)} bytes)\n")
                await console_q.put(f"DEBUG: Got transcript, pending reply: {pending_assistant_reply is not None}\n")
                
                # If we have a pending assistant reply, log the interaction
                if pending_assistant_reply is not None:
                    await console_q.put("DEBUG: Logging interaction with pending assistant reply\n")
                    await log_interaction(current_audio_bytes, transcript, pending_assistant_reply)
                    await log_timing_stats()  # Log timing after complete interaction
                    pending_assistant_reply = None
                # If we have an in-progress reply, wait for it to complete
                elif assistant_reply_buffer:
                    await console_q.put("DEBUG: Waiting for assistant reply to complete\n")
                
                current_audio = bytearray()  # Start fresh for next utterance

            elif event.type == "response.created":
                assistant_start_time = time.time()
                await console_q.put(f"DEBUG: Assistant started at {assistant_start_time}\n")
                
            elif event.type == "response.text.delta":
                delta = event.delta
                assistant_reply_buffer.append(delta)
                await console_q.put(delta)
                
                # If we have both audio and transcript, log immediately
                if current_audio_bytes is not None and current_transcript is not None:
                    await console_q.put("DEBUG: Got assistant delta, have audio and transcript, logging interaction\n")
                    reply = "".join(assistant_reply_buffer)
                    await log_interaction(current_audio_bytes, current_transcript, reply)
                    await log_timing_stats()  # Log timing after complete interaction
                    assistant_reply_buffer.clear()
                    current_audio_bytes = None
                    current_transcript = None
                    
            elif event.type == "response.text.done":
                assistant_end_time = time.time()
                await console_q.put("\n")
                await console_q.put(f"DEBUG: Assistant ended at {assistant_end_time}\n")
                complete_reply = "".join(assistant_reply_buffer)
                
                # If we have both audio and transcript, log immediately
                if current_audio_bytes is not None and current_transcript is not None:
                    await console_q.put("DEBUG: Got response.done, logging final interaction\n")
                    await log_interaction(current_audio_bytes, current_transcript, complete_reply)
                    await log_timing_stats()  # Log timing after complete interaction
                # Otherwise store the complete reply for when we get the transcript
                else:
                    await console_q.put("DEBUG: Storing complete reply for when transcript arrives\n")
                    pending_assistant_reply = complete_reply
                
                assistant_reply_buffer.clear()
                current_audio_bytes = None
                current_transcript = None

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        callback=_callback,
    )

    # Start the stream synchronously
    stream.start()
    try:
        # Run the async tasks while the stream is active
        await asyncio.gather(_writer(), _await_transcripts(), console_writer(), _audio_collector())
    finally:
        # Ensure the stream is stopped when we're done
        stream.stop()
        stream.close()

# ─────────────────── Main ──────────────────
async def main() -> None:
    client = AsyncOpenAI()

    async with client.beta.realtime.connect(model=MODEL) as conn:
        # Disable assistant audio; enable transcription
        await conn.session.update(
            session={
                "modalities": ["text"],  # Text-only responses
                "input_audio_format": "pcm16",
                "instructions": "You are a friendly assistant. Talk only in english. Use super short conversationalsentences.",
                "input_audio_transcription": {"model": "whisper-1"},
                "max_response_output_tokens": 20,
                "turn_detection": {"type": "semantic_vad"},
            }
        )

        await console_q.put("🎙️  Speak after the prompt. Ctrl‑C to quit.\n\n")
        await stream_microphone(conn)

# ctrl‑C friendly
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye!", file=sys.stderr)
