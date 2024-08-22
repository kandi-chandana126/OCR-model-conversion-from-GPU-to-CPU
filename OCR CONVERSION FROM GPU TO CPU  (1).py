#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python-headless easyocr torch')


# In[2]:


get_ipython().system('pip install editdistance')


# In[3]:


import torch
print(torch.cuda.is_available())  # Should return True if a GPU is available


# In[ ]:


#wWithout GPU


# In[27]:


import cv2
import easyocr
import time
import psutil
import editdistance
import matplotlib.pyplot as plt

# Initialize EasyOCR Reader (CPU-based)
reader = easyocr.Reader(['en'], gpu=False)

def extract_frames(video_path, target_fps=30, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps // target_fps))  # Ensure frame_interval is at least 1
    frames = []
    frame_count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def convert_to_grayscale(frames):
    grayscale_frames = []
    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale_frames.append(gray_frame)
    return grayscale_frames

def apply_ocr_on_frames(frames, reference_text, max_outputs=10):
    ocr_results = []
    total_time = 0
    frame_count = 0

    for i, frame in enumerate(frames):
        if i >= max_outputs:  # Stop after processing max_outputs frames
            break

        start_time = time.time()
        result = reader.readtext(frame)
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        frame_count += 1

        ocr_results.append(result)

        # Draw bounding boxes and text on the frames
        for (bbox, text, prob) in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple([int(coord) for coord in top_left])
            bottom_right = tuple([int(coord) for coord in bottom_right])

            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate FPS and CER (Character Error Rate)
        fps = frame_count / total_time if total_time > 0 else 0
        cer_value = calculate_cer(reference_text, result)
        accuracy_based_on_cer = 100 - cer_value

        # Add FPS and Accuracy to the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Accuracy: {accuracy_based_on_cer:.2f}%", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Convert frame to RGB for Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame using Matplotlib
        plt.imshow(frame_rgb)
        plt.axis('off')
        plt.show()

    return {
        "total_time": total_time,
        "fps": frame_count / total_time if total_time > 0 else 0,
        "ocr_results": ocr_results
    }

def get_resource_utilization():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    return {
        "cpu_usage": cpu_usage,
        "ram_usage": memory_info.percent
    }

def calculate_cer(reference_text, ocr_output):
    ocr_text = " ".join([text for _, text, _ in ocr_output])
    reference_text = ' '.join(reference_text.split())
    ocr_text = ' '.join(ocr_text.split())
    distance = editdistance.eval(reference_text, ocr_text)
    cer = distance / len(reference_text) if len(reference_text) > 0 else 0
    return cer * 100

def main(video_path, reference_path):
    # Extract a limited number of frames from the video
    frames = extract_frames(video_path, max_frames=10)

    if not frames:
        print("No frames extracted.")
        return

    # Show the first frame for verification
    plt.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Convert frames to grayscale
    grayscale_frames = convert_to_grayscale(frames)

    # Read reference text for comparison with explicit encoding (UTF-8)
    try:
        with open(reference_path, 'r', encoding='utf-8') as file:
            reference_text = file.read()
    except UnicodeDecodeError:
        print("Error reading file with UTF-8 encoding. Attempting 'latin-1' encoding.")
        with open(reference_path, 'r', encoding='latin-1') as file:
            reference_text = file.read()

    # Apply OCR to frames and calculate performance metrics
    performance_metrics = apply_ocr_on_frames(grayscale_frames, reference_text, max_outputs=10)

    # Get CPU and RAM usage
    resource_utilization = get_resource_utilization()

    # Print performance metrics
    print(f"Total Processing Time: {performance_metrics['total_time']} seconds")
    print(f"Frames Per Second (FPS): {performance_metrics['fps']}")
    print(f"CPU Usage: {resource_utilization['cpu_usage']}%")
    print(f"RAM Usage: {resource_utilization['ram_usage']}%")

# Specify video and reference text file paths
video_path = "C:/Users/k.chandana/Downloads/WhatsApp Video 2024-08-22 at 17.36.06.mp4"
reference_path = "C:/Users/k.chandana/Downloads/SOCIA; ALL DAY.txt"  # Path to a valid text file

# Run the main function
main(video_path, reference_path)
  # Path to a valid text file

# Run the main function
main(video_path, reference_path)


# In[ ]:


#with GPU


# In[8]:


get_ipython().system('pip install opencv-python-headless easyocr torch editdistance')


# In[9]:


import torch
print(torch.cuda.is_available())


# In[26]:


import cv2
import easyocr
import time
import psutil
import editdistance
import matplotlib.pyplot as plt

reader = easyocr.Reader(['en'], gpu=True)

def extract_frames(video_path, target_fps=30, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Ensure frame_interval is at least 1
    frame_interval = max(1, int(fps // target_fps))
    
    frames = []
    frame_count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def convert_to_grayscale(frames):
    grayscale_frames = []
    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale_frames.append(gray_frame)
    return grayscale_frames

def apply_ocr_on_frames(frames, reference_text):
    ocr_results = []
    total_time = 0
    frame_count = 0
    first_frame = None

    for i, frame in enumerate(frames):
        start_time = time.time()
        result = reader.readtext(frame)
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        frame_count += 1

        ocr_results.append(result)

        if i == 0:  # Process only the first frame for display
            first_frame = frame.copy()
            for (bbox, text, prob) in result:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple([int(coord) for coord in top_left])
                bottom_right = tuple([int(coord) for coord in bottom_right])
                cv2.rectangle(first_frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(first_frame, text, (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            fps = frame_count / total_time if total_time > 0 else 0
            cer_value = calculate_cer(reference_text, result)
            accuracy_based_on_cer = 100 - cer_value

            cv2.putText(first_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(first_frame, f"Accuracy: {accuracy_based_on_cer:.2f}%", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return {
        "total_time": total_time,
        "fps": frame_count / total_time if total_time > 0 else 0,
        "ocr_results": ocr_results,
        "first_frame": first_frame
    }

def get_resource_utilization():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    return {
        "cpu_usage": cpu_usage,
        "ram_usage": memory_info.percent
    }

def calculate_cer(reference_text, ocr_output):
    ocr_text = " ".join([text for _, text, _ in ocr_output])
    reference_text = ' '.join(reference_text.split())
    ocr_text = ' '.join(ocr_text.split())
    distance = editdistance.eval(reference_text, ocr_text)
    cer = distance / len(reference_text) if len(reference_text) > 0 else 0
    return cer * 100

def main(video_path, reference_path):
    # Limit to max 10 frames
    frames = extract_frames(video_path, max_frames=10)
    grayscale_frames = convert_to_grayscale(frames)

    try:
        with open(reference_path, 'r', encoding='utf-8') as file:
            reference_text = file.read()
    except UnicodeDecodeError:
        print("Error reading file with UTF-8 encoding. Attempting 'latin-1' encoding.")
        with open(reference_path, 'r', encoding='latin-1') as file:
            reference_text = file.read()

    performance_metrics = apply_ocr_on_frames(grayscale_frames, reference_text)
    resource_utilization = get_resource_utilization()

    if performance_metrics["first_frame"] is not None:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(performance_metrics["first_frame"], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    print(f"Total Processing Time: {performance_metrics['total_time']} seconds")
    print(f"Frames Per Second (FPS): {performance_metrics['fps']}")
    print(f"Character Error Rate (CER): {calculate_cer(reference_text, performance_metrics['ocr_results'][0]):.2f}%")
    print(f"OCR Accuracy Based on CER: {100 - calculate_cer(reference_text, performance_metrics['ocr_results'][0]):.2f}%")
    print(f"CPU Usage: {resource_utilization['cpu_usage']}%")
    print(f"RAM Usage: {resource_utilization['ram_usage']}%")

# Paths to video and reference file
video_path = "C:/Users/k.chandana/Downloads/WhatsApp Video 2024-08-22 at 17.36.06.mp4"
reference_path = "C:/Users/k.chandana/Downloads/SOCIA; ALL DAY.txt"
main(video_path, reference_path)


# In[ ]:


#ON CPU With
#Resolution Variation
#Multithreading
#skip similar frames


# In[29]:


get_ipython().system('pip install opencv-python-headless easyocr torch editdistance')


# In[30]:


import torch
print(torch.cuda.is_available())


# In[32]:


import cv2
import easyocr
import time
import psutil
import numpy as np
import editdistance
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

reader = easyocr.Reader(['en'], gpu=False)

def reduce_resolution(frames, scale_percent=90):
    processed_frames = []
    for frame in frames:
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        processed_frames.append(resized_frame)
    return processed_frames

def is_similar(frame1, frame2, threshold=3000):
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    mse = np.sum((gray_frame1 - gray_frame2) ** 2)
    mse /= float(gray_frame1.shape[0] * gray_frame1.shape[1])
    return mse < threshold

def process_frame_with_ocr(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray_frame)
    return results

def draw_boxes(frame, ocr_results):
    for bbox, text, _ in ocr_results:
        p1, p2, p3, p4 = bbox
        cv2.polylines(frame, [np.array([p1, p2, p3, p4], np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, text, (int(p1[0]), int(p1[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def apply_ocr_optimized(frames):
    processed_frames = reduce_resolution(frames)

    # Show and save a sample frame after reducing resolution
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(processed_frames[0], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    # cv2.imwrite('sample_reduced_resolution.jpg', processed_frames[0])

    ocr_results = []
    previous_frame = None

    with ThreadPoolExecutor(max_workers=4) as executor:  # Multi-threading
        for i in range(len(processed_frames)):
            if previous_frame is not None and is_similar(previous_frame, processed_frames[i]):
                continue  # Skip OCR if frames are similar
            result = executor.submit(process_frame_with_ocr, processed_frames[i]).result()
            ocr_results.append(result)
            previous_frame = processed_frames[i]

    return processed_frames, ocr_results

def calculate_editdistance(reference, ocr_output):
    ocr_texts = []
    for result in ocr_output:
        if len(result) == 3:  # Ensure there are three elements to unpack
            _, text, _ = result
            ocr_texts.append(text)
        elif len(result) == 1:  # Handle the case with only one element (unlikely but safe)
            ocr_texts.append(result[0])

    ocr_text = " ".join(ocr_texts)
    return editdistance.eval(reference, ocr_text), ocr_text

def main(video_path, reference_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    start_time = time.time()
    processed_frames, ocr_results = apply_ocr_optimized(frames)
    end_time = time.time()

    total_time = end_time - start_time
    fps = len(frames) / total_time if total_time > 0 else 0

    with open(reference_path, 'r', encoding='latin-1') as file:
        reference_text = file.read().strip()

    # Calculate Edit Distance and Accuracy Based on Edit Distance
    edit_distance_value, extracted_text = calculate_editdistance(reference_text, [item for sublist in ocr_results for item in sublist])
    accuracy_based_on_editdistance = (1 - (edit_distance_value / max(len(reference_text), len(extracted_text)))) * 100

    # Show sample frame with OCR bounding boxes
    sample_frame = processed_frames[0]
    draw_boxes(sample_frame, ocr_results[0])
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    # cv2.imwrite('sample_with_ocr_boxes.jpg', sample_frame)

    # Print results
    print(f"Total Processing Time: {total_time} seconds")
    print(f"Frames Per Second (FPS): {fps}")
    print(f"Edit Distance: {edit_distance_value}")
    print(f"Accuracy Based on Edit Distance: {accuracy_based_on_editdistance:.2f}%")
    print(f"Extracted Text: {extracted_text}")
    print(f"Reference Text: {reference_text}")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"RAM Usage: {psutil.virtual_memory().percent}%")

# Example Usage
video_path = "C:/Users/k.chandana/Downloads/WhatsApp Video 2024-08-22 at 17.36.06.mp4"
reference_path = "C:/Users/k.chandana/Downloads/SOCIA; ALL DAY.txt"
main(video_path, reference_path)


# In[ ]:




