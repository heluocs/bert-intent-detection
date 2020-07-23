docker run -p 8501:8501 \
    --mount type=bind,source=/root/workspace/bert/SavedModel/,target=/models/uncased_L-12_H-768_A-12 \
    -e MODEL_NAME=uncased_L-12_H-768_A-12 -t tensorflow/serving
