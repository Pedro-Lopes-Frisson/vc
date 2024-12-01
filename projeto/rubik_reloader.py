import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys

# Path to the file you want to watch
FILE_TO_WATCH = "./square_detection.py"

# Command to run the Python script
COMMAND = ["python", FILE_TO_WATCH]

# Process handler
process = None

def start_script():
    """Start the Python script."""
    global process
    if process:
        process.terminate()  # Terminate existing process
        process.wait()       # Ensure it's stopped
    process = subprocess.Popen(COMMAND)  # Start the script

class ChangeHandler(FileSystemEventHandler):
    """Handler to restart the script on file modification."""
    def on_modified(self, event):
        if event.src_path.endswith(FILE_TO_WATCH):
            print(f"Detected change in {FILE_TO_WATCH}. Restarting script...")
            start_script()

if __name__ == "__main__":
    # Start the script initially
    FILE_TO_WATCH = sys.argv[1]
    COMMAND = ["python", FILE_TO_WATCH]
    print(f"Starting {FILE_TO_WATCH}...")
    start_script()

    # Set up watchdog to monitor the file
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(sys.argv[0]) or ".", recursive=False)

    try:
        observer.start()
        print(f"Watching for changes in {FILE_TO_WATCH}...")
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        print("\nStopping...")
        observer.stop()
    observer.join()
    if process:
        process.terminate()

