import cv2

def concatenate_videos_horizontally(video1_path, video2_path, output_path, label1=None, label2=None):
    # Open the two videos
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

    # Check if the videos are opened correctly
    if not video1.isOpened() or not video2.isOpened():
        print("Error opening videos")
        return

    # Read the first frames of the two videos
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    # Check if the frames are read correctly
    if not ret1 or not ret2:
        print("Error reading frames")
        return

    # Get information about the videos (size, number of frames, fps...)
    video_width = frame1.shape[1] + frame2.shape[1]
    video_height = max(frame1.shape[0], frame2.shape[0])
    fps = min(video1.get(cv2.CAP_PROP_FPS), video2.get(cv2.CAP_PROP_FPS))
    num_frames = max(int(video1.get(cv2.CAP_PROP_FRAME_COUNT)), int(video2.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Create the VideoWriter for the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))

    # Define the font and size of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    # Concatenate the frames of the two videos and write the resulting frame to the output video
    for i in range(num_frames):
        # Read the frames of the two videos
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        # Check if the frames are read correctly
        if not ret1 or not ret2:
            break

        # Resize the frames if necessary so that they have the same height
        if frame1.shape[0] != frame2.shape[0]:
            resize_factor = frame2.shape[0] / frame1.shape[0]
            frame1 = cv2.resize(frame1, (int(frame1.shape[1] * resize_factor), frame2.shape[0]))

        # Add the label texts to the frames if the labels are defined
        if label1 is not None:
            frame1 = cv2.putText(frame1, label1, (10, frame1.shape[0] - 10), font, font_scale, (0, 0, 255), font_thickness)
        if label2 is not None:
            frame2 = cv2.putText(frame2, label2, (10, frame2.shape[0] - 10), font, font_scale, (0, 0, 255), font_thickness)

        # Concatenate the frames horizontally
        frame = cv2.hconcat([frame1, frame2])

        # Write the resulting frame to the output video
        out.write(frame)

    # Release the resources
    video1.release()
    video2.release()
    out.release()
    cv2.destroyAllWindows()
