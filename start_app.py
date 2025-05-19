# Simple HTTP server for serving the frontend files
import http.server
import socketserver
import webbrowser
import os
from threading import Thread
import time
import subprocess

# Configuration
FRONTEND_FOLDER = "frontend"
PORT = 3000
API_PORT = 8000

def start_frontend_server():
    """Start a simple HTTP server for the frontend"""
    os.chdir(FRONTEND_FOLDER)
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", PORT), handler)
    print(f"Frontend server started at http://localhost:{PORT}")
    httpd.serve_forever()

def start_api_server():
    """Start the FastAPI server using Uvicorn"""
    subprocess.Popen([
        "uvicorn", 
        "src.api:app", 
        "--host", "0.0.0.0", 
        "--port", str(API_PORT),
        "--reload"
    ])
    print(f"API server started at http://localhost:{API_PORT}")

def open_browser():
    """Open the browser with the frontend URL after a short delay"""
    time.sleep(2)  # Give the servers time to start
    webbrowser.open(f"http://localhost:{PORT}")

if __name__ == "__main__":
    # Start the API server
    start_api_server()
    
    # Start the frontend server in a separate thread
    frontend_thread = Thread(target=start_frontend_server)
    frontend_thread.daemon = True
    frontend_thread.start()
    
    # Open the browser
    open_browser()
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down servers...")
