# Gemini Workspace (`GEMINI.md`)

This file provides Gemini with instructional context about the `wheelson-explorer` project.

## Project Overview

This project, **Wheelson Explorer**, turns a CircuitMess Wheelson robot into an autonomous explorer. It uses a hybrid architecture:

*   **Firmware:** A C++/Arduino program runs on the Wheelson's ESP32. It handles hardware control (motors, camera, ultrasonic sensor) and exposes a minimal REST API over Wi-Fi.
*   **Middleware:** A Python application using FastAPI runs on a host computer. It fetches camera images from the robot, sends them to a Vision-Language Model (Gemini or Ollama), and uses the VLM's response to command the robot.
*   **Dashboard:** A live web UI served by the middleware visualizes the camera feed, the VLM's "thoughts," and sensor data in real-time using Server-Sent Events.

The core of the project is its use of "personalities," which are different system prompts that give the robot distinct characters, narration styles, and movement philosophies (e.g., a safety inspector, a nature documentarian).

## Key Files

*   `README.md`: The primary documentation. Contains detailed setup instructions for both firmware and middleware, hardware requirements, and API reference.
*   `firmware/wheelson_explorer/wheelson_explorer.ino`: The ESP32 firmware. It's written in C++/Arduino and is self-contained, requiring no external libraries beyond the ESP32 core.
*   `middleware/main.py`: The core of the project. This Python script runs the FastAPI server, manages the main exploration loop, communicates with the Wheelson and the VLM, and serves the dashboard.
*   `middleware/requirements.txt`: Python dependencies for the middleware (`fastapi`, `uvicorn`, `httpx`, `Pillow`, `google-genai`, `python-dotenv`).
*   `middleware/dashboard.html`: The single-page web interface for the live dashboard.
*   `middleware/.env.example`: Template for environment variables, including the Wheelson's IP address and the Gemini API key.

## Building and Running

### Firmware Setup

1.  **Prerequisites:** Arduino IDE 2.x with ESP32 board support.
2.  **Configuration:**
    *   Open `firmware/wheelson_explorer/wheelson_explorer.ino`.
    *   Edit the `WIFI_SSID` and `WIFI_PASSWORD` definitions.
3.  **Board Settings (in Arduino IDE):**
    *   **Board:** `ESP32 Dev Module`
    *   **Partition Scheme:** `Huge APP (3MB No OTA / 1MB SPIFFS)`
    *   **PSRAM:** `Enabled` (Critical for camera)
    *   **Upload Speed:** `115200` (recommended for stability)
4.  **Flash:** Connect the Wheelson via USB and click **Upload**.

### Middleware Setup

1.  **Prerequisites:** Python 3.10+ and a VLM provider (Gemini API Key or a local Ollama instance).
2.  **Installation:**
    ```bash
    cd middleware
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Configuration:**
    *   Copy `.env.example` to `.env`.
    *   Edit `.env` to set `WHEELSON_IP` (found from the firmware's serial output) and your `GEMINI_API_KEY`.
4.  **Run:**
    ```bash
    # Example using Gemini and the "Sir David" personality
    python main.py --provider gemini --personality sir_david
    ```
5.  **Dashboard:** Open `http://localhost:8000` (or the configured port) in a browser.

## Development Conventions

*   **Firmware:** The firmware is designed to be minimal and dependency-free (outside the standard ESP32 Arduino core). It handles low-level hardware interactions and provides a simple, stable API. Safety logic (a 10cm hard-stop) is built-in.
*   **Middleware:** The middleware contains the primary "business logic." It is responsible for the main loop, VLM interaction, and serving the user interface. It also implements safety features like a 25cm soft-stop and stuck detection.
*   **Communication:**
    *   Middleware to Firmware: Standard HTTP requests (`GET /camera`, `POST /move`).
    *   Middleware to Dashboard: Server-Sent Events (SSE) for pushing real-time data (logs, thoughts, status) to the browser.
*   **Personalities:** New robot behaviors can be added by defining a new entry in the `PERSONALITIES` dictionary in `middleware/main.py`. This involves creating a system prompt, a name, and associated metadata.
*   **Testing:** Manual testing is done by running the firmware and middleware and observing the robot's behavior and the dashboard output. The `README.md` includes `curl` commands to smoke-test the firmware API.
