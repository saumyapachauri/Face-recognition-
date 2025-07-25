
import org.opencv.core.*;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

public class FaceRecognition {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        CascadeClassifier faceDetector = new CascadeClassifier("classifier/haarcascade_frontalface_alt.xml");
        LBPHFaceRecognizer recognizer = LBPHFaceRecognizer.create();
        FaceTrainer.trainRecognizer(recognizer, "data/");

        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("Camera not detected!");
            return;
        }

        Mat frame = new Mat();
        while (true) {
            if (!camera.read(frame)) break;
            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            Rect[] facesArray = faceDetector.detectMultiScale(gray).toArray();

            for (Rect rect : facesArray) {
                Mat face = new Mat(gray, rect);
                Imgproc.resize(face, face, new Size(200, 200));
                int[] label = new int[1];
                double[] confidence = new double[1];
                recognizer.predict(face, label, confidence);
                Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
                Imgproc.putText(frame, "User: " + label[0], rect.tl(), Imgproc.FONT_HERSHEY_PLAIN, 1.5, new Scalar(255, 0, 0));
            }

            HighGui.imshow("Face Recognition", frame);
            if (HighGui.waitKey(1) == 27) break; // ESC to quit
        }

        camera.release();
        HighGui.destroyAllWindows();
    }
}
