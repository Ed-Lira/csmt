"""
Main entry point for the CSMT application.
"""
import asyncio
import os
from aiohttp import web
from .websocket_server import AudioStreamServer

async def serve_static(request):
    """Serve the static HTML page."""
    return web.FileResponse(os.path.join(os.path.dirname(__file__), 'static', 'index.html'))

async def start_server(http_port: int = 9876, ws_port: int = 9877):
    """Start both the HTTP and WebSocket servers."""
    # Create the audio stream server
    audio_server = AudioStreamServer(host="localhost", port=ws_port)
    
    # Create the HTTP application
    app = web.Application()
    app.router.add_get('/', serve_static)
    
    # Start the HTTP server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", http_port)
    await site.start()
    
    print(f"HTTP server started at http://localhost:{http_port}")
    print(f"WebSocket server started at ws://localhost:{ws_port}")
    
    # Start the WebSocket server
    await audio_server.start()

def main() -> None:
    """Main entry point for the application."""
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 