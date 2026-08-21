import subprocess
import sys
import time
import webbrowser
from urllib.request import urlopen

FLASK_URL = "http://127.0.0.1:5000/health"
STREAMLIT_URL = "http://127.0.0.1:8501"


def wait_for_server(url, timeout=30):
    """Wait until a local web server becomes available."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with urlopen(url, timeout=1):
                return True
        except Exception:
            time.sleep(0.5)

    return False


def main():
    print("Starting Fraud Detection application...")
    print()

    # Start Flask API using module execution (-m) to properly resolve imports
    print("Starting Flask API...")
    flask_process = subprocess.Popen([sys.executable, "-m", "src.api"])

    # Wait for Flask to become available
    if not wait_for_server(FLASK_URL):
        print("ERROR: Flask API failed to start.")
        flask_process.terminate()
        return

    print("Flask API is running.")

    # Start Streamlit
    print("Starting Streamlit interface...")
    streamlit_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/streamlit_app.py",
            "--server.headless=true",
        ]
    )

    # Wait for Streamlit to become available
    if not wait_for_server(STREAMLIT_URL):
        print("ERROR: Streamlit failed to start.")
        flask_process.terminate()
        streamlit_process.terminate()
        return

    print("Streamlit interface is running.")
    print()

    # Open browser only after Streamlit is ready
    print("Opening application in browser...")
    webbrowser.open(STREAMLIT_URL)

    print()
    print("Application running:")
    print(f"  Streamlit: {STREAMLIT_URL}")
    print("  Flask API: http://127.0.0.1:5000")
    print()
    print("Press Ctrl+C to stop the application.")

    try:
        while True:

            if flask_process.poll() is not None:
                print("Flask API stopped.")
                break

            if streamlit_process.poll() is not None:
                print("Streamlit stopped.")
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print()
        print("Stopping application...")

    finally:

        print("Stopping Flask API...")
        flask_process.terminate()

        print("Stopping Streamlit...")
        streamlit_process.terminate()

        flask_process.wait()
        streamlit_process.wait()

        print("Application stopped.")


if __name__ == "__main__":
    main()
