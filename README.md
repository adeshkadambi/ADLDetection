# ADL Detection

This encompasses all the code required to run the ADL detection pipeline, which starts from raw videos and ends with the generation of the final results on 1-minute intervals.

## Pipeline

The pipeline consists of the following steps:

1. Chunking videos into 1-minute intervals
2. Sampling frames at 1 FPS on each chunk
3. Running object detection on sampled frames
4. Running hand-object detection on sampled frames
5. Determining active objects using the IoU > 0.8 between object detections and hand-object detections
6. Generating feature vectors from active and passive objects
7. Training a simple logistic regression model to classify the activity from the objects
