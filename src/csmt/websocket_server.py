"""
WebSocket server for audio streaming and transcription.
Handles conversational turn detection and transcription using OpenAI.
"""
import asyncio
import json
import sounddevice as sd
import numpy as np
import base64
import time
from typing import Set, Optional, Dict, Any, List
import websockets
from websockets.server import WebSocketServerProtocol
from openai import AsyncOpenAI

class AudioStreamServer:
    def __init__(self, host: str = "localhost", port: int = None):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.stream: Optional[sd.InputStream] = None
        self.sample_rate = 24000  # 24kHz for OpenAI compatibility
        self.channels = 1
        self.dtype = np.int16  # int16 for OpenAI compatibility
        self.block_size = 1024
        self.audio_queue: asyncio.Queue = None
        self.text_queue: asyncio.Queue = None  # Queue for text messages (transcriptions and responses)
        self._broadcast_task: Optional[asyncio.Task] = None
        self._text_broadcast_task: Optional[asyncio.Task] = None
        self._openai_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # OpenAI related attributes
        self.openai_client: Optional[AsyncOpenAI] = None
        self.openai_conn = None
        self.openai_audio_queue = None
        self.current_turn_id = 0
        self.turn_start_time = 0
        self.assistant_reply_buffer: List[str] = []


    async def start(self):
        """Start the WebSocket server."""
        self._loop = asyncio.get_running_loop()
        self.audio_queue = asyncio.Queue()
        self.text_queue = asyncio.Queue()

        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI()

        # Start worker tasks
        self._broadcast_task = asyncio.create_task(self._broadcast_worker())
        self._text_broadcast_task = asyncio.create_task(self._text_broadcast_worker())

        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"WebSocket server started at ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever

    async def _initialize_openai(self):
        """Initialize the OpenAI connection."""
        if self.openai_conn:
            return

        # Create a queue for audio data to be sent to OpenAI
        self.openai_audio_queue = asyncio.Queue()

        # Create a task that will run for the lifetime of the connection
        self._openai_task = asyncio.create_task(self._openai_connection_task())

    async def _openai_connection_task(self):
        """Task that maintains the OpenAI connection and processes events."""
        try:
            # Create the connection manager
            print("Initializing OpenAI connection...")
            connection_manager = self.openai_client.beta.realtime.connect(
                model="gpt-4o-mini-realtime-preview",
            )

            # Use the connection manager as a context manager
            async with connection_manager as conn:
                print("OpenAI connection established")
                self.openai_conn = conn  # Store the connection for reference

                # Configure the session
                await conn.session.update(
                    session={
                        "modalities": ["text"],  # Text-only responses
                        "input_audio_format": "pcm16",
                        "instructions": "You are a friendly human. Talk only in english. Use super short conversational sentences.",
                        "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
                        "max_response_output_tokens": 20,
                        "turn_detection": {"type": "semantic_vad"},
                    }
                )

                # Start a task to process audio data from the queue
                audio_task = asyncio.create_task(self._process_audio_queue(conn))

                try:
                    # Process events from OpenAI
                    await self._process_openai_events(conn)
                finally:
                    # Cancel the audio processing task when we're done
                    audio_task.cancel()
                    try:
                        await audio_task
                    except asyncio.CancelledError:
                        pass
                    self.openai_conn = None
        except Exception as e:
            print(f"Error in OpenAI connection task: {e}")
            self.openai_conn = None
            raise

    async def _process_audio_queue(self, conn):
        """Process audio data from the queue and send it to OpenAI."""
        while True:
            # Get audio data from the queue
            b64_audio = await self.openai_audio_queue.get()

            # Send it to OpenAI
            await conn.input_audio_buffer.append(audio=b64_audio)

    async def _process_openai_events(self, conn):
        """Process events from OpenAI and send to clients."""
        try:
            async for event in conn:
                print(f"OpenAI event: {event.type}")
                if hasattr(event, 'error'):
                    print(f"OpenAI error details: {event.error}")  # Log error details
                if hasattr(event, 'data'):
                    print(f"OpenAI event data: {event.data}")  # Log event data

                if event.type == "error":
                    print(f"OpenAI error occurred: {event.error if hasattr(event, 'error') else 'Unknown error'}")
                    # Try to reconnect if it's a connection error
                    if hasattr(event, 'error') and 'connection' in str(event.error).lower():
                        print("Attempting to reconnect...")
                        await self._initialize_openai()
                    return  # Exit the event loop on error

                elif event.type == "input_audio_buffer.speech_started":
                    self.turn_start_time = time.time()
                    self.current_turn_id += 1
                    self.assistant_reply_buffer.clear()

                elif event.type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.transcript.strip()
                    turn_duration = time.time() - self.turn_start_time

                    # Send transcription to clients
                    message = {
                        "type": "transcription",
                        "text": transcript,
                        "is_final": True,
                        "start_time": self.turn_start_time,
                        "end_time": time.time(),
                        "turn_id": self.current_turn_id,
                        "turn_duration": turn_duration
                    }
                    await self.text_queue.put(json.dumps(message))

                elif event.type == "response.text.delta":
                    delta = event.delta
                    self.assistant_reply_buffer.append(delta)
                    print(f"Received response delta: {delta}")  # Debug log

                elif event.type == "response.text.done":
                    complete_reply = "".join(self.assistant_reply_buffer)
                    print(f"Received complete response: {complete_reply}")  # Debug log
                    # Get the transcript for this turn
                    user_transcript = ""
                    for msg in self.text_queue._queue:
                        try:
                            msg_data = json.loads(msg)
                            if msg_data.get("type") == "transcription" and msg_data.get("turn_id") == self.current_turn_id:
                                user_transcript = msg_data.get("text", "")
                                break
                        except:
                            continue

                    # Send assistant response to clients
                    message = {
                        "type": "assistant_response",
                        "turn_id": self.current_turn_id,
                        "user_transcript": user_transcript,
                        "ai_response": complete_reply
                    }
                    print(f"Sending assistant response message: {message}")  # Debug log
                    await self.text_queue.put(json.dumps(message))
                    self.assistant_reply_buffer.clear()

        except Exception as e:
            print(f"Error processing OpenAI events: {e}")

    async def _text_broadcast_worker(self):
        """Worker task that broadcasts text messages to connected clients."""
        while True:
            try:
                # Get text message from queue
                text_message = await self.text_queue.get()

                if self.clients:
                    # Create tasks to send to each client
                    tasks = []
                    for client in self.clients:
                        try:
                            tasks.append(asyncio.create_task(
                                client.send(text_message)
                            ))
                        except websockets.exceptions.ConnectionClosed:
                            pass

                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                print(f"Error in text broadcast worker: {e}")

            await asyncio.sleep(0.01)  # Small delay to prevent CPU overload

    async def _broadcast_worker(self):
        """Worker task that processes audio data but doesn't broadcast it to clients."""
        while True:
            try:
                # Get audio data from queue but don't broadcast it
                await self.audio_queue.get()
                # Audio data is still being processed for OpenAI in audio_callback
            except Exception as e:
                print(f"Error in broadcast worker: {e}")

            await asyncio.sleep(0.01)  # Small delay to prevent CPU overload

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input - puts data into the queue and sends to OpenAI."""
        if status:
            print(f"Audio callback status: {status}")
        if self._loop is not None:
            try:
                # Convert audio data to bytes and put in queue for streaming
                audio_data = indata.tobytes()
                self._loop.call_soon_threadsafe(
                    lambda: self.audio_queue.put_nowait(audio_data)
                )

                # Send to OpenAI if connection is established
                if hasattr(self, 'openai_audio_queue'):
                    # Convert to base64
                    b64_audio = base64.b64encode(audio_data).decode()
                    self._loop.call_soon_threadsafe(
                        lambda: self.openai_audio_queue.put_nowait(b64_audio)
                    )
            except Exception as e:
                print(f"Error in audio callback: {e}")

    async def stop(self):
        """Stop the server and cleanup resources."""
        # Stop OpenAI tasks
        if self._openai_task:
            self._openai_task.cancel()
            try:
                await self._openai_task
            except asyncio.CancelledError:
                pass

        # OpenAI connection is closed automatically when the task is cancelled

        # Stop text broadcast task
        if self._text_broadcast_task:
            self._text_broadcast_task.cancel()
            try:
                await self._text_broadcast_task
            except asyncio.CancelledError:
                pass

        # Stop audio broadcast task
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

        # Stop audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a new WebSocket client connection."""
        self.clients.add(websocket)
        try:
            print(f"New client connected. Total clients: {len(self.clients)}")
            if len(self.clients) == 1:  # Start audio stream when first client connects
                await self.start_audio_stream()

            # Keep the connection alive until client disconnects
            async for message in websocket:
                # Handle any messages from the client if needed
                pass

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print(f"Client disconnected. Remaining clients: {len(self.clients)}")
            if not self.clients:  # Stop audio stream when last client disconnects
                await self.stop_audio_stream()

    async def start_audio_stream(self):
        """Start capturing audio and streaming to clients and OpenAI."""
        # Initialize OpenAI connection
        await self._initialize_openai()

        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.block_size,
            callback=self.audio_callback
        )
        self.stream.start()
        print("Audio stream started")

    async def stop_audio_stream(self):
        """Stop the audio stream and OpenAI connection."""
        # Stop audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("Audio stream stopped")

        # Stop OpenAI tasks
        if self._openai_task:
            self._openai_task.cancel()
            try:
                await self._openai_task
            except asyncio.CancelledError:
                pass
            self._openai_task = None

        # OpenAI connection is closed automatically when the task is cancelled

        # Don't cancel the broadcast tasks as they're needed for the server lifetime
