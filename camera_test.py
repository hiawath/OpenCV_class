import cv2

def open_camera_feed():
    # 0 typically refers to the default camera connected to the system
    # You can change this number (e.g., 1, 2) if you have multiple cameras.
    cap = cv2.VideoCapture(0)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream or find camera.")
        return

    print("Camera opened successfully. Press 'q' on the video window to exit.")

    while True:
        # Capture frame-by-frame
        # 'ret' is a boolean indicating success, 'frame' is the actual image data
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('Live Camera Feed', frame)

        # Wait for a key press for 1 millisecond.
        # If the pressed key is 'q', break the loop.
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything is done, release the capture and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_camera_feed()