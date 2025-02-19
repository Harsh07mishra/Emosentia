import streamlit as st
import tempfile
import cv2
import os
from fer import FER  # Facial Emotion Recognition library
import whisper
from moviepy.editor import VideoFileClip, ImageSequenceClip
import time
from textblob import TextBlob

# Background and styles
image_url = "https://images.pexels.com/photos/8281874/pexels-photo-8281874.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    .title {{
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 25px;
        font-family: 'Arial', sans-serif;
    }}
    .sidebar-text {{
        writing-mode: vertical-rl;
        transform: rotate(180deg);
        font-size: 80px;
        font-family: 'Arial', sans-serif;
        color: orange;
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-left: 30px;
    }}
    .emotion-summary {{
        font-size: 18px;
        font-weight: bold;
        color: #333;
    }}
    .overall-sentiment {{
        font-size: 24px;
        font-weight: bold;
        color: #FF5733;
    }}
    </style>
    """, unsafe_allow_html=True
)

# Sidebar
st.sidebar.markdown('<div class="sidebar-text">EMOSENTIA</div>', unsafe_allow_html=True)

# Main title
st.markdown('<div class="title">Real-Time Emotion and Sentiment Detection with Speech Transcription</div>', unsafe_allow_html=True)

# Load FER and Whisper models
emotion_detector = FER(mtcnn=True)
whisper_model = whisper.load_model("base")

# Upload video file
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Temporary file to store the uploaded video
    st.subheader("Original Video")
    st.video(video_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        video_path = tfile.name


    # Load video using OpenCV
    vid = cv2.VideoCapture(video_path)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # Emotion to sentiment mapping
    emotion_sentiment = {
        "happy": 1, "surprise": 1,  # Positive
        "neutral": 0,               # Neutral
        "sad": -1, "angry": -1, "fear": -1, "disgust": -1  # Negative
    }
    emotion_counts = {}

    # Initialize a list to store processed frames
    processed_frames = []

    # Analyzing frames
    st.write("Analyzing video for emotion detection...")
    with st.spinner("Processing..."):
        frame_interval = 1  # Process every 10th frame
        for i in range(0, total_frames, frame_interval):
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vid.read()
            if not ret:
                st.warning("Could not read frame, stopping the analysis.")
                break

            # Resize for faster processing
            frame_resized = cv2.resize(frame, (320, 240))

            # Detect emotions and draw bounding box
            emotion_data = emotion_detector.detect_emotions(frame_resized)
            if emotion_data:
                for detection in emotion_data:
                    box = detection['box']
                    emotion, confidence = max(detection['emotions'].items(), key=lambda item: item[1])

                    # Draw green bounding box and add emotion label
                    cv2.rectangle(frame_resized, (int(box[0]), int(box[1])),
                                  (int(box[0] + box[2]), int(box[1] + box[3])), (0, 255, 0), 2)
                    cv2.putText(frame_resized, f"{emotion} ({confidence:.2f})",
                                (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Append the processed frame to the list
            processed_frames.append(frame_resized)

        # Convert frames to video using ImageSequenceClip
        output_video_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
        clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in processed_frames], fps=fps)
        video_clip = VideoFileClip(video_path)  # Load original video to access its audio
        if video_clip.audio is not None:
            clip = clip.set_audio(video_clip.audio)  # Set original audio to the processed video

        clip.write_videofile(output_video_path, codec="libx264")

        # Display the generated video with audio in Streamlit
        st.subheader("Emotion Detection Processed Video with Audio")
        st.video(output_video_path)

        # # Display the generated video in Streamlit
        # st.subheader("Emotion Detection Processed Video")
        # st.video(output_video_path)

         # Extract audio from video for transcription
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
        audio_path = audio_file.name
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)

    # Transcribe audio
    st.write("Extracting speech from the video...")
    transcription = whisper_model.transcribe(audio_path)
    os.remove(audio_path)  # Clean up temp audio file

    # Display transcription and analyze sentiment
    st.subheader("Video Speech Transcription")
    transcription_text = transcription['text']

    # Split the transcription into sentences and analyze sentiment
    sentences = transcription_text.split('.')
    sentiment_scores = []
    sentiment_score = 0  # Initialize sentiment score for overall sentiment analysis

    for sentence in sentences:
        if sentence.strip():  # Avoid empty sentences
            blob = TextBlob(sentence.strip())
            sentiment = blob.sentiment.polarity
            sentiment_scores.append((sentence.strip(), sentiment))  # Store sentence and its sentiment
            sentiment_score += sentiment  # Aggregate sentiment score

    # Determine overall sentiment from the aggregated score
    if sentiment_score > 0:
        overall_sentiment = "Positive"
    elif sentiment_score < 0:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    # Display overall sentiment
    st.subheader("Overall Video Sentiment")
    st.markdown(f'<div class="overall-sentiment">The overall sentiment of the video is **{overall_sentiment}**.</div>', unsafe_allow_html=True)

    # Display sentences that match the overall sentiment
    st.subheader(f"Sentences with **{overall_sentiment}** Sentiment:")
    matching_sentences = [sentence for sentence, score in sentiment_scores if (overall_sentiment == "Positive" and score > 0) or
                                                             (overall_sentiment == "Negative" and score < 0) or
                                                             (overall_sentiment == "Neutral" and score == 0)]

    if matching_sentences:
        for sentence in matching_sentences:
            st.write(f"- {sentence}")
    else:
        st.write("No sentences match the overall sentiment.")

    # Cleanup

    # Release the video
    vid.release()
    os.remove(video_path)    # Remove the original video file
    os.remove(output_video_path)  # Remove the processed video file
    