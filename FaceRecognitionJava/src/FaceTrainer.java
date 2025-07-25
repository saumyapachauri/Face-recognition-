
import org.opencv.core.*;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.*;

public class FaceTrainer {
    public static void trainRecognizer(LBPHFaceRecognizer recognizer, String dataPath) {
        List<Mat> images = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        File root = new File(dataPath);
        int labelCount = 0;
        for (File dir : root.listFiles()) {
            if (dir.isDirectory()) {
                for (File img : dir.listFiles()) {
                    Mat image = Imgcodecs.imread(img.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
                    Imgproc.resize(image, image, new Size(200, 200));
                    images.add(image);
                    labels.add(labelCount);
                }
                labelCount++;
            }
        }
        recognizer.train(images, new MatOfInt(toIntArray(labels)));
    }

    private static int[] toIntArray(List<Integer> list) {
        return list.stream().mapToInt(i -> i).toArray();
    }
}
