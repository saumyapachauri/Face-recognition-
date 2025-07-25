# Java Console Face Recognition

## Description
A simple Java + OpenCV project to detect and recognize faces using webcam and LBPHFaceRecognizer.

## How to Use
1. Install OpenCV and add the `.jar` and native `.dll/.so` to your project.
2. Place face images in `data/user1/`, `data/user2/`, etc.
3. Compile and run the project.

## Run Command
Ensure OpenCV native library is loaded correctly.

```bash
java -Djava.library.path=path/to/opencv/native -cp . FaceRecognition
```
