import subprocess
import sys
import time
import webbrowser


def main():
    print("Starting Fraud Detection application...")
    print()

    # Start Flask API
    print("Starting Flask API on port 5000...")
    flask_process = subprocess.Popen([sys.executable, "src/api.py"])

    # Give Flask a moment to start
    time.sleep(3)

    # Start Streamlit
    print("Starting Streamlit interface...")
    streamlit_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/streamlit_app.py",
        ]
    )

    # Give Streamlit a moment to start
    time.sleep(4)

    # Open the application in the browser
    print("Opening Fraud Detection interface...")
    webbrowser.open("http://localhost:8501")

    print()
    print("Application running.")
    print("Streamlit: http://localhost:8501")
    print("Flask API: http://localhost:5000")
    print()
    print("Press Ctrl+C to stop the application.")

    try:
        # Keep the launcher alive while both services are running
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
        # Stop both processes
        flask_process.terminate()
        streamlit_process.terminate()

        flask_process.wait()
        streamlit_process.wait()

        print("Application stopped.")


if __name__ == "__main__":
    main()
