"""
WebSocket server for audio streaming and transcription.
"""
import asyncio
import json
import sounddevice as sd
import numpy as np
from typing import Set, Optional
import websockets
from websockets.server import WebSocketServerProtocol
from .transcription_service import TranscriptionService

class AudioStreamServer:
    def __init__(self, host: str = "localhost", port: int = None):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.stream: Optional[sd.InputStream] = None
        self.sample_rate = 44100
        self.channels = 1
        self.dtype = np.float32
        self.block_size = 1024
        self.audio_queue: asyncio.Queue = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._transcription_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Initialize transcription service
        self.transcription_service = TranscriptionService(
            whisper_model_size="base",  # You can change this to "tiny" for faster but less accurate results
            device="cpu",  # Change to "cuda" if you have a GPU
            language="en"
        )

    async def start(self):
        """Start the WebSocket server."""
        self._loop = asyncio.get_running_loop()
        self.audio_queue = asyncio.Queue()
        self._broadcast_task = asyncio.create_task(self._broadcast_worker())
        self._transcription_task = asyncio.create_task(self._transcription_worker())
        
        # Start the transcription service
        await self.transcription_service.start()
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"WebSocket server started at ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever

    async def _transcription_worker(self):
        """Worker task that processes transcriptions and sends them to clients."""
        while True:
            try:
                # Get transcription result
                result = await self.transcription_service.get_transcription()
                if result and self.clients:
                    # Create message with transcription
                    message = {
                        "type": "transcription",
                        "text": result.text,
                        "is_final": result.is_final,
                        "start_time": result.start_time,
                        "end_time": result.end_time
                    }
                    
                    # Send to all clients
                    tasks = []
                    for client in self.clients:
                        try:
                            tasks.append(asyncio.create_task(
                                client.send(json.dumps(message))
                            ))
                        except websockets.exceptions.ConnectionClosed:
                            pass
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                print(f"Error in transcription worker: {e}")
            
            await asyncio.sleep(0.1)

    async def _broadcast_worker(self):
        """Worker task that broadcasts audio data to connected clients."""
        while True:
            try:
                # Get audio data from queue
                audio_data = await self.audio_queue.get()
                
                if self.clients:
                    # Create tasks to send to each client
                    tasks = []
                    for client in self.clients:
                        try:
                            tasks.append(asyncio.create_task(
                                client.send(audio_data)
                            ))
                        except websockets.exceptions.ConnectionClosed:
                            pass
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                print(f"Error in broadcast worker: {e}")
            
            await asyncio.sleep(0.01)  # Small delay to prevent CPU overload

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input - puts data into the queue and transcription service."""
        if status:
            print(f"Audio callback status: {status}")
        if self._loop is not None:
            try:
                # Convert audio data to bytes and put in queue for streaming
                audio_data = indata.tobytes()
                self._loop.call_soon_threadsafe(
                    lambda: self.audio_queue.put_nowait(audio_data)
                )
                
                # Add to transcription service
                self.transcription_service.add_audio(indata, self.sample_rate)
            except Exception as e:
                print(f"Error in audio callback: {e}")

    async def stop(self):
        """Stop the server and cleanup resources."""
        if self._transcription_task:
            self._transcription_task.cancel()
            try:
                await self._transcription_task
            except asyncio.CancelledError:
                pass
        
        await self.transcription_service.stop()
        
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        
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
        """Start capturing audio and streaming to clients."""
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
        """Stop the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("Audio stream stopped")
        
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None 