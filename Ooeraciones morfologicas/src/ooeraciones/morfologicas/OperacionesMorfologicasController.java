package ooeraciones.morfologicas;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;

//import 
//import org.opencv.core.RotatedRect;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import static java.lang.Math.cos;
import static java.lang.Math.sin;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.ResourceBundle;
import java.util.Scanner;
import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javax.imageio.ImageIO;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import static org.opencv.core.CvType.CV_8UC3;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.Highgui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import ooeraciones.morfologicas.RotatedRect.*;
import org.jcp.xml.dsig.internal.dom.Utils;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.getRotationMatrix2D;
import static org.opencv.imgproc.Imgproc.line;

/**
 * FXML Controller class
 *
 * @author Alex_Salazar_M
 */
public class OperacionesMorfologicasController implements Initializable {

        @FXML ImageView uno;
        @FXML ImageView dos;
        @FXML ImageView tres;
        @FXML Label respuesta;
        @FXML Label puntos;
        @FXML Label  unodos;
        @FXML Label  title;
        @FXML Label  rayas;
        @FXML Label conect;
        
        
        public void iniciando(String archivo,String archivodos){
                Mat harris_scene_scaled=new Mat();
                Mat scene_norm=new Mat();
                int blockSize = 9;
                int apertureSize =5;
//                String archivo = "bookobject";
                Mat imagen = Imgcodecs.imread(archivo,Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);                
                scene_norm = Imgcodecs.imread(archivodos,Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);                
                        
                Mat imagenoriginal=imagen.clone();
                Mat imafenoriginal=scene_norm.clone();
                double trehs=Imgproc.threshold(imagen,imagen,158,255, Imgproc.THRESH_BINARY);
                double trehsdos=Imgproc.threshold(scene_norm,scene_norm,158,255, Imgproc.THRESH_BINARY);
                Harris(imagen,scene_norm,(int)trehs,(int)trehsdos,imagenoriginal,imafenoriginal);               
        }
        
        private void Harris(Mat Scene, Mat Object,int thresh,int threshdos,Mat imagen, Mat scene_norm) {

                Mat Harris_scene = new Mat();
                Mat Harris_object = new Mat();

                Mat harris_scene_norm = new Mat(), harris_object_norm = new Mat(), harris_scene_scaled = new Mat(), harris_object_scaled = new Mat();
                int blockSize = 9;
                int apertureSize = 5;
                double k = 0.1;
                int f=0;
        //        Imshow.show(Harris_scene,"antes");
                Imgproc.cornerHarris(Scene, Harris_scene,blockSize, apertureSize,k);
        //        Imshow.show(Harris_scene,"despues");
                Imgproc.cornerHarris(Object, Harris_object, blockSize,apertureSize,k);
        //        Imshow.show(Harris_object);
                Core.normalize(Harris_scene, harris_scene_norm, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());
                Core.normalize(Harris_object, harris_object_norm, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());

                Core.convertScaleAbs(harris_scene_norm, harris_scene_scaled);
                Core.convertScaleAbs(harris_object_norm,harris_object_norm);

                for( int j = 0; j < harris_scene_norm.rows() ; j++){
                    for( int i = 0; i < harris_scene_norm.cols(); i++){
                        if ((int) harris_scene_norm.get(j,i)[0] > thresh){
                            Imgproc.circle(Scene, new Point(i,j),1, new Scalar(0), 2 ,3 , 0);
        //                        System.out.println("posicion x: "+i+"posicion y: "+j);
                                f++;

                        }
                    }
                }
        //        System.out.println("puntos encontrados: "+f);
        //        Imshow.show(Scene);
        //harris_object_norm
                int cont=0;
                for( int j = 0; j < harris_object_norm.rows(); j++){
                    for( int i = 0; i < harris_object_norm.cols(); i++){
                        if ((int) harris_object_norm.get(j,i)[0] > thresh){
                            Imgproc.circle(scene_norm, new Point(i,j),2, new Scalar(0), 2 ,8 , 0);
                                boolean c = angleprediction(i,j,harris_object_norm,harris_object_norm,thresh);
                                if(c==true)cont++;
                                f++;

                        }
                    }
                }
                System.out.println("contador: "+cont);
                Imshow.show(Scene,"Cantidad de puntos: "+f+"ncontrados: "+cont);
         }        
        public boolean angleprediction(int x,int y,Mat magen,Mat harris_object_norm,int thresh){
        
        int centerX = magen.width()/2;
        int centerY = magen.height()/2;
        double theta = (20+0.0) * Math.PI / 180.0;
        int xT = (int) (centerX+(x-centerX)*cos(theta)+(y-centerY)*sin(theta));
        int yT = (int) (centerY-(x-centerX)*sin(theta)+(y-centerY)*cos(theta));
        
        for( int j = 0; j < harris_object_norm.rows() ; j++){
                for( int i = 0; i < harris_object_norm.cols(); i++){
                        if ((int) harris_object_norm.get(j,i)[0] > thresh){
                                Imgproc.circle(magen, new Point(i,j),2, new Scalar(0), 2 ,8 , 0);
                                if(j==xT&&i==yT){
                                        System.out.println("(Original: )"+"Col: "+xT+"row: "+yT+"(Original: )"+"Col: "+i+"row: "+j);
                                        return true;
                            }
                        }
                }
        }
    

        return false;
}
        /**
         * Initializes the controller class.
         */
        @Override
        public void initialize(URL url, ResourceBundle rb) {

                String bookObject = "imagenes//bookobject.jpg";
                Mat objectImage = Imgcodecs.imread(bookObject, Highgui.CV_LOAD_IMAGE_COLOR);
                for(int i=20;i<360;i+=20){
                        Mat salida=rotacionimagen(objectImage,(double)i);
                        Imgcodecs.imwrite("imagenes//bookobject"+i+".jpg", salida);
                }  
//                    MatOfKeyPoint keypoints_imagen_original;
                MatOfKeyPoint keypoints_imagen_original;
                System.out.println("What do u want\n\t\t1:-Harris\n\t\t2:-Sift");
                Scanner sc=new Scanner(System.in);
                int caso=sc.nextInt();
                System.out.println("Indique el angulo: ");
                double angles;
                angles=sc.nextDouble();
                keypoints_imagen_original=detection(objectImage);
                switch(caso){
                        case 1: System.out.println("What do u want\nexercise:\n1- Keypoints\n\t\t\t:->2-Rotacion\n\t\t\t:->3-Escalado");
                        int unmero=sc.nextInt();
                        if(unmero==1){
                                calculandorotacion();
                        }
                        if(unmero==2){
                                double angulo = angles;
                                double num_coincidencias = harris_features(objectImage, keypoints_imagen_original, angulo);
        //                        calculandorotacion();
        //                        harris_features(objectImage, keypoints_imagen_original, angulo);
                        }
                        if(unmero==3){

                                Scanner scan=new Scanner(System.in);
                                double de=scan.nextDouble();
                                double valor_escalado = 8;
                                double num_coincidencias = hescalado(objectImage, keypoints_imagen_original,valor_escalado);
                                System.out.println("Coincidencias: "+num_coincidencias);
                                System.out.println("Porcenytaje ingresar:"+num_coincidencias*100/de);


        //                        Escalado(angles, keypoint_imagen_prueba);       
                        }
                        break;
                        case 2:
                                System.out.println("Indica que usaras: ");
//                                Scanner sc= new Scanner(System.in);
                                int es=sc.nextInt();
                                System.out.println("1.- Rotación");
                                System.out.println("2.- Escalado");
                                String bookScene = "imagenes//bookobject.jpg";
                                keypoints_imagen_original=sift_detection(Imgcodecs.imread(bookScene, Highgui.CV_LOAD_IMAGE_COLOR));
                                Mat aqui=new Mat();
                                aqui=Imgcodecs.imread(bookScene, Highgui.CV_LOAD_IMAGE_COLOR);
                                if(es==1){
                                        for(double i=0;i<360;i+=20){
                                                String objetodos = "imagenes//bookobject"+i+".jpg";
                //                                objectImage
                                                Imgcodecs.imread(objetodos, Highgui.CV_LOAD_IMAGE_COLOR);
                                                double angulo = i;
                                                double num_coincidencias =sift_detection_rotacion(aqui, keypoints_imagen_original, angulo);            
                                                System.out.println("Coninciencias: "+num_coincidencias);
                                                System.out.println("Repetibilidad: "+(num_coincidencias/keypoints_imagen_original.toArray().length)*100);
                                        }     
                                }else{
                                        double valor_escalado = 7;
                                        double num_coincidencias = hescalado(aqui, keypoints_imagen_original, valor_escalado);
                                        System.out.println("Coincidencias: "+num_coincidencias);
                                        System.out.println("Repetibilidad: "+(num_coincidencias/keypoints_imagen_original.toArray().length));
            
                                }

                        break;
                        default:break;
                }              
        }

        public void SiFDetector(String bookObject,Mat sceneImage,int angle){
                System.out.println("Angule: "+angle);
                File lib = null;
                String os = System.getProperty("os.name");
                String bitness = System.getProperty("sun.arch.data.model");

        
        
//        System.out.println(lib.getAbsolutePath());
//        System.load(lib.getAbsolutePath());

//        String bookObject = "imagenes//bookobject.jpg";
//        String bookScene = "imagenes//bookobject.jpg";

                System.out.println("Iniciando....");
        //        System.out.println("Loading image..."); 
                Mat objectImage = Imgcodecs.imread(bookObject, Highgui.CV_LOAD_IMAGE_COLOR);
        //        Mat sceneImage = Imgcodecs.imread(bookScene, Highgui.CV_LOAD_IMAGE_COLOR);



                MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
                FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.ORB);
                System.out.println("Detecting key points...");
                featureDetector.detect(objectImage, objectKeyPoints);
                KeyPoint[] keypoints = objectKeyPoints.toArray();
                System.out.println(keypoints.length);
                respuesta.setText(": "+keypoints.length);
                MatOfKeyPoint objectDescriptors = new MatOfKeyPoint();
                DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
                System.out.println("Computing descriptors...");
                descriptorExtractor.compute(objectImage, objectKeyPoints, objectDescriptors);

                // Create the matrix for output image.
                Mat outputImage = new Mat(objectImage.rows(), objectImage.cols(), Highgui.CV_LOAD_IMAGE_COLOR);
                Scalar newKeypointColor = new Scalar(255, 0, 0);

                System.out.println("Drawing key points on object image...");
                Features2d.drawKeypoints(objectImage, objectKeyPoints, outputImage, newKeypointColor, 0);

        //se marca los puntos que coinciden con la imagen original
                MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
                MatOfKeyPoint sceneDescriptors = new MatOfKeyPoint();
        //        System.out.println("Deetectando puntos en la imagen de fondo...");
                featureDetector.detect(sceneImage, sceneKeyPoints);

        //        ArrayList<KeyPoint> pointre=new ArrayList<KeyPoint>((Collection<? extends KeyPoint>) sceneKeyPoints);
        //        System.out.println("Calculando descriptores en la imagen de fondo...");
                descriptorExtractor.compute(sceneImage, sceneKeyPoints, sceneDescriptors);
        //        List<KeyPoint> keypoints_almacenado = sceneDescriptors.toList();
        //        ArrayList<KeyPoint> key_points = new ArrayList (objectKeyPoints.toList());       
        //        PredicPosicionesdelmasasha((ArrayList<KeyPoint>) keypoints_almacenado,20,objectImage,sceneImage,sceneDescriptors);
                Mat matchoutput = new Mat(sceneImage.rows() * 2, sceneImage.cols() * 2, Highgui.CV_LOAD_IMAGE_COLOR);
                Scalar matchestColor = new Scalar(0, 255, 0);

                List<MatOfDMatch> matches = new LinkedList<>();
                DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_L1);
                System.out.println("Marcando los puntos en la imagen de fondo...");
                descriptorMatcher.knnMatch(objectDescriptors, sceneDescriptors, matches,2);

                System.out.println("Calculando los puntos...");
                LinkedList<DMatch> goodMatchesList = new LinkedList<>();
                float nndrRatio = 0.7f;
                int encontrados = 0;
                for (int i = 0; i < matches.size(); i++) {
                    MatOfDMatch matofDMatch = matches.get(i);
                    DMatch[] dmatcharray = matofDMatch.toArray();
                    DMatch m1 = dmatcharray[0];
                    DMatch m2 = dmatcharray[1];
                if (m1.distance <= m2.distance * nndrRatio) {
                        goodMatchesList.addLast(m1);
                    }
                        encontrados+=goodMatchesList.size();               
                }
                conect.setText(": "+encontrados);

                System.out.println(""+goodMatchesList.size());
                unodos.setText(": "+goodMatchesList.size());
                if (goodMatchesList.size() >= 7) {
                    System.out.println("Encontrado!!!");

                    List<KeyPoint> objKeypointlist = objectKeyPoints.toList();
                    List<KeyPoint> scnKeypointlist = sceneKeyPoints.toList();

                    LinkedList<Point> objectPoints = new LinkedList<>();
                    LinkedList<Point> scenePoints = new LinkedList<>();

                    for (int i = 0; i < goodMatchesList.size(); i++) {
                        objectPoints.addLast(objKeypointlist.get(goodMatchesList.get(i).queryIdx).pt);
                        scenePoints.addLast(scnKeypointlist.get(goodMatchesList.get(i).trainIdx).pt);
                    }
                    MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
                    objMatOfPoint2f.fromList(objectPoints);
                    MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
                    scnMatOfPoint2f.fromList(scenePoints);

                    Mat homography = Calib3d.findHomography(objMatOfPoint2f, scnMatOfPoint2f, Calib3d.RANSAC, 3);
        //              Mat homography=Imgproc.getPerspectiveTransform(objMatOfPoint2f,scnMatOfPoint2f);   
                    Mat obj_corners = new Mat(4, 1, CvType.CV_32FC2);
                    Mat scene_corners = new Mat(4, 1, CvType.CV_32FC2);

                    obj_corners.put(0, 0, new double[]{0, 0});
                    obj_corners.put(1, 0, new double[]{objectImage.cols(), 0});
                    obj_corners.put(2, 0, new double[]{objectImage.cols(), objectImage.rows()});
                    obj_corners.put(3, 0, new double[]{0, objectImage.rows()});
            
//            Mat presp=Imgproc.getPerspectiveTransform(objMatOfPoint2f,scnMatOfPoint2f);
            System.out.println("Transforming object corners to scene corners...");
//            Imgproc.warpPerspective(presp, presp, presp, dsize);
            Core.perspectiveTransform(obj_corners, scene_corners, homography);
//Vlfit
            Mat img = Imgcodecs.imread(bookObject, Highgui.CV_LOAD_IMAGE_COLOR);

//            line(img, new Point(scene_corners.get(0, 0)), new Point(scene_corners.get(1, 0)), new Scalar(0, 255, 0), 4);
//            line(img, new Point(scene_corners.get(1, 0)), new Point(scene_corners.get(2, 0)), new Scalar(0, 255, 0), 4);
//            line(img, new Point(scene_corners.get(2, 0)), new Point(scene_corners.get(3, 0)), new Scalar(0, 255, 0), 4);
//            line(img, new Point(scene_corners.get(3, 0)), new Point(scene_corners.get(0, 0)), new Scalar(0, 255, 0), 4);

            System.out.println("Dibujado puntos marcados...");
            MatOfDMatch goodMatches = new MatOfDMatch();
            goodMatches.fromList(goodMatchesList);
            Features2d.drawMatches(objectImage, objectKeyPoints, sceneImage, sceneKeyPoints, goodMatches, matchoutput, matchestColor, newKeypointColor, new MatOfByte(), 2);
//            double thresh = Imgproc.threshold(outputImage,outputImage,0,255, Imgproc.THRESH_BINARY_INV);           
            uno.setImage(convertir(outputImage));
            Imshow.show(matchoutput,"puntos similares: "+encontrados+"Totales: "+keypoints.length+"encontrados: "+goodMatchesList.size());
            dos.setImage(convertir(sceneImage));
            tres.setImage(convertir(img));
                System.out.println("Porcentaje: "+(encontrados*100/keypoints.length)/100+"%");            
        } else {
            System.out.println("Object Not Found");
        }

        System.out.println("Ended....");
    }

 

        public Image convertir(Mat imagen) {
                MatOfByte matOfByte = new MatOfByte();
                Imgcodecs.imencode(".jpg", imagen, matOfByte); 

                byte[] byteArray = matOfByte.toArray();
                BufferedImage bufImage = null;

                try {
                    InputStream in = new ByteArrayInputStream(byteArray);
                    bufImage = ImageIO.read(in);
                } catch (IOException e) {
                }    
                Image image = SwingFXUtils.toFXImage(bufImage, null);
                return image;
        }

        /** 
     * Image is first resized-to-fit the dst Mat and then rotated. 
     * mRgba is the source image, mIntermediateMat should have the same type.
     */
    private void rotationTutorial(Mat mRgba){
  
//         Mat mRgba=new Mat();
        double ratio =  mRgba.height() / (double) mRgba.width();

        int rotatedHeight = mRgba.height();     
        int rotatedWidth  = (int) Math.round(mRgba.height() * ratio);

        Mat mIntermediateMat=new Mat();

        Imgproc.resize(mRgba, mIntermediateMat, new Size(rotatedHeight, rotatedWidth));

        Core.flip(mIntermediateMat.t(), mIntermediateMat, 0);

        Mat ROI = mRgba.submat(0, mIntermediateMat.rows(), 0, mIntermediateMat.cols());

        mIntermediateMat.copyTo(ROI);       
    }


    /** 
     * Image is rotated - cropped-to-fit dst Mat.
     * 
     */
    public Mat rotationAffine(Mat mRgba, double angulo){

//            Mat mRgba=new Mat();
            // assuming source image's with and height are a pair value:
        int centerX = mRgba.width()/2;
        int centerY = mRgba.height()/2;
                Mat mIntermediateMat = new Mat(mRgba.cols(),mRgba.rows(),mRgba.type());
//        RotatedRect(,angle,scale);
        Point center = new Point(centerY,centerX);
        double angle = angulo;
        double scale = 1.0;
         double ratio= mRgba.height() / (double) mRgba.width();int rotatedHeight=0;int rotatedWidth=0;
        if(mRgba.cols()>mRgba.rows()){
                ratio =  mRgba.cols() /mRgba.rows();
                rotatedHeight = mRgba.width();       
                rotatedWidth  = (int)(mRgba.width()* ratio);
        }else{
                ratio =  mRgba.rows() /mRgba.cols();
                rotatedHeight = mRgba.height();       
                rotatedWidth  = (int)(mRgba.height()* ratio);
        }


        Mat mapMatrix = Imgproc.getRotationMatrix2D(center, angle, scale);
        Rect bbox =new RotatedRect(center,mRgba.size(),angle).boundingRect();
        mapMatrix.get(0,2)[0]+=(bbox.width/2.0)-center.x;
        mapMatrix.get(1,2)[0]+=(bbox.height/2.0)-center.y;
        
        Size rotatedSize = new Size(rotatedHeight,rotatedWidth);


        Imgproc.warpAffine(mRgba,mRgba, mapMatrix,bbox.size());

        Mat ROI =mRgba.submat(0,mRgba.rows(),0,mRgba.cols());
        mIntermediateMat.copyTo(ROI);
        line(mRgba, new Point(mIntermediateMat.get(0, 0)), new Point(mIntermediateMat.get(1, 0)), new Scalar(0, 255, 0), 4);



        return mIntermediateMat;

    }
    public void rotacionsincrop(String archivo){
               
                Mat imagen = Imgcodecs.imread("imagenes\\"+archivo+".jpg",Highgui.IMREAD_UNCHANGED);
                double ratio=imagen.height() / (double) imagen.width(); int rotatedHeight=0;int rotatedWidth=0;                
                rotatedHeight = imagen.height();
               
                double angle=20;
                int centerX = imagen.cols()/2;
                int centerY = imagen.rows()/2;
                Point center = new Point(centerY,centerX);
                Mat mapMatrix = Imgproc.getRotationMatrix2D(center, angle,1);
                Rect bbox =new RotatedRect(center,imagen.size(),angle).boundingRect();     
                mapMatrix.get(0,2)[0]+=bbox.width/2.0-center.x;
                mapMatrix.get(1,2)[0]+=bbox.height/2.0-center.y;
                Imgproc.warpAffine(imagen,imagen, mapMatrix,bbox.size()); 
                rotatedWidth  = (int)(imagen.width()* ratio);
                Size rotatedSize = new Size(rotatedHeight,rotatedWidth);
                Mat mIntermediateMat = new Mat(imagen.size(),imagen.type());
                Mat ROI =imagen.submat(0,mIntermediateMat.rows(),0,mIntermediateMat.cols());                
                mIntermediateMat.copyTo(ROI);
                line(mIntermediateMat, new Point(mIntermediateMat.get(0, 0)), new Point(mIntermediateMat.get(1, 0)), new Scalar(0, 255, 0), 4);
               Imshow.show(mIntermediateMat,"Prueba");
        }
    public void calculandorotacion(){
                Mat imagen=new Mat();
                Mat imagendos=new Mat();
                String archivo="imagenes//imagen_0.jpg";
                String archivodos="imagenes//imagen_20.jpg";
                Mat Scene=Imgcodecs.imread(archivo,Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
                Mat Object=Imgcodecs.imread(archivodos,Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
                double trehs=Imgproc.threshold(Scene.clone(),imagen,158,255, Imgproc.THRESH_BINARY);
                double treh=Imgproc.threshold(Object.clone(),imagendos,158,255, Imgproc.THRESH_BINARY);
                int thresh=(int)trehs; 
                int threshdos=0;Mat scene_norm=new Mat();

                Mat Harris_scene = new Mat();
                Mat Harris_object = new Mat();

                Mat harris_scene_norm = new Mat(), harris_object_norm = new Mat(), harris_scene_scaled = new Mat(), harris_object_scaled = new Mat();
                int blockSize = 9;
                int apertureSize = 5;
                double k = 0.1;
                double kd = 0.05;
                int f=0;
        //        Imshow.show(Harris_scene,"antes");
                Imgproc.cornerHarris(imagendos, Harris_scene,blockSize, apertureSize,kd);
        //        Imshow.show(Harriharriss_scene,"despues");
                Imgproc.cornerHarris(imagen, Harris_object, blockSize,apertureSize,k);
        //        Imshow.show(Harris_object);
                Core.normalize(Harris_scene, harris_scene_norm, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());
                Core.normalize(Harris_object, harris_object_norm, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());

                Core.convertScaleAbs(harris_scene_norm, harris_scene_scaled);
                Core.convertScaleAbs(harris_object_norm,harris_object_norm);

                for( int j = 0; j < harris_scene_norm.rows() ; j++){
                    for( int i = 0; i < harris_scene_norm.cols(); i++){
                        if ((int) harris_scene_norm.get(j,i)[0] > treh){
                            Imgproc.circle(Object, new Point(i,j),1, new Scalar(0), 2 ,3 , 0);
                            Imgproc.circle(Scene, new Point(i,j),1, new Scalar(0), 2 ,3 , 0);
                            f++;
                        }
                    }
                }
                System.out.println("puntos encontrados: "+f);
                Imshow.show(Object,"Imagen Rotada, Cantidad de puntos: "+f);
                Imshow.show(Scene,"Original con puntos rotados Cantidad de puntos: "+f);
                //harris_object_norm
                int cont=0;
                for( int j = 0; j < harris_object_norm.rows(); j++){
                    for( int i = 0; i < harris_object_norm.cols(); i++){
                        if ((int) harris_object_norm.get(j,i)[0] > thresh){
                            Imgproc.circle(Scene, new Point(i,j),2, new Scalar(0), 2 ,8 , 0);
//                                boolean c = angleprediction(i,j,harris_object_norm,harris_object_norm,thresh);
                                cont++;

                        }
                    }
                }
                System.out.println("porcentaje:"+f*100/cont+"%\n");
                System.out.println("puntos encontrados: "+cont);
                Imshow.show(Scene,"Original con puntos rotados y originales, Cantidad de puntos: "+f+"encontrados: "+prediccionesdeposiciondepixel(harris_object_norm,harris_scene_norm,thresh,(int)treh,Scene,cont));
//                prediccionesdeposiciondepixel(harris_object_norm,harris_scene_norm,thresh,(int)treh,Scene);
         }
    public int prediccionesdeposiciondepixel(Mat imagenconkeypoints,Mat imagenrotada,int thresh,int threshdo,Mat Sc,int c){
                
/*        double theta = (20+0.0) * Math.PI / 180.0;
        int xT = (int) (centerX+(x-centerX)*cos(theta)+(y-centerY)*sin(theta));
        int yT = (int) (centerY-(x-centerX)*sin(theta)+(y-centerY)*cos(theta));*/
                String archivo="imagenes//imagen_0.jpg";
                String archivos="imagenes//imagen_20.jpg";
                Mat Scene=Imgcodecs.imread(archivo,Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
                Mat Scenes=Imgcodecs.imread(archivos,Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
                int contador=0;
                double theta = Math.toRadians(-20);
                int centerX = imagenconkeypoints.width()/2;
                int centerY = imagenconkeypoints.height()/2;                
                Point center = new Point(centerY,centerX);
                int contadorinterno=0;
                int xoriginal=(imagenconkeypoints.width()/2);
                int yoriginal=(imagenconkeypoints.height()/2);
                for( int j = 0; j <imagenconkeypoints.rows(); j++){
                        for( int i = 0; i < imagenconkeypoints.cols(); i++){
                                if ((int) imagenconkeypoints.get(j,i)[0] > thresh){
//                                        float xprima = (float) (((i - xoriginal)*cos(theta))-((yoriginal- j)*sin(theta)));
                                        center = new Point(j,i);
//                                        int yc_predic=(int) ((int) ((int)(center.y-xoriginal)*(cos(theta)))-((center.x-xoriginal)*sin(theta))+);
                                        float xprima=(float) (xoriginal+(((i-xoriginal))*cos(theta))-((j-yoriginal)*sin(theta)));
//                                       float yprima=(float) (((yoriginal - j)*cos(theta))-((i - xoriginal) * sin(theta)));
                                        float yprima=(float) (xoriginal+(((i-xoriginal))*sin(theta))+((j-yoriginal)*cos(theta)));
                                        Imgproc.circle(Scene, new Point(xprima,yprima),2, new Scalar(0), 2 ,8 , 0);
                                        Imgproc.circle(Scenes, new Point(xprima,yprima),2, new Scalar(0), 2 ,8 , 0);
                                contadorinterno++;
                                }
                        }
                }
                Imshow.show(Scene,"Original y predichos");
//                Imshow.show(Scenes,"Original y predichos");
                System.out.println("Original: "+(contadorinterno-contador));
                System.out.println("Predicho: "+(c));
                System.out.println("Diferencia: "+(contadorinterno-contador));
                
        return contador;
        }
    public void scaladodekeypoints(Mat imagenconkeypoints){
                int y_center=imagenconkeypoints.height()/2;
                int x_center=imagenconkeypoints.width()/2;
                float m=(float) 1.2;
                float []scaleFactor ={0,1,2,3,4,5,6,7,8};
                

                
        }
        
    public Mat rotar_imagen(Mat imageOriginal,double angulo){        
        Point punto = new Point(imageOriginal.width()/2,imageOriginal.height()/2);          
        //System.out.println("Punto central prro2"+punto);        
        Mat r = getRotationMatrix2D(punto,angulo,1.0);
        double cos = Math.abs(r.get(0,0)[0]);
        double sin = Math.abs(r.get(0,1)[0]);        
        int nw = (int) ((imageOriginal.height() * sin) + (imageOriginal.width() * cos));
        int nh = (int) ((imageOriginal.height() * cos) + (imageOriginal.width() * sin));        
        double prro = r.get(0,2)[0] + nw/2.0 - punto.x;
        double prro2 = r.get(1,2)[0] + nh/2.0 - punto.y;        
        r.put(0, 2,prro);
        r.put(1, 2,prro2);        
        Mat salida = new Mat();          
        Imgproc.warpAffine(imageOriginal,salida,r,new Size(nw,nh));
        return salida;
    }

    public Mat escalarimag(Mat imagenOriginal,double n){
                Mat imagen_escalada = new Mat();        
                double m = 1.2;
                m = Math.pow(m, n);
                Size size = new Size(m*imagenOriginal.width(),m*imagenOriginal.height());
                Imgproc.resize(imagenOriginal,imagen_escalada,size,0,0,Imgproc.INTER_CUBIC);
                return imagen_escalada;
        }
        
    private MatOfKeyPoint PredicPosicionesdelmasasha(ArrayList<KeyPoint> OriginalImgfilter, double theta, Mat original, Mat rotada, MatOfKeyPoint keypoints_imagen) {
                
                Point pointercentreprueba = new Point(rotada.width() / 2, rotada.height() / 2);
                Point pointercentreoriginal = new Point(original.width() / 2, original.height() / 2);
                double errorx = pointercentreprueba.x - pointercentreoriginal.x;
                double errory = pointercentreprueba.y - pointercentreoriginal.y;
                theta = theta * -1;

                double XP = 0;
                double YP = 0;

                double cos = Math.cos(Math.toRadians(theta));
                double sin = Math.sin(Math.toRadians(theta));
                ArrayList<Point> keypoint = new ArrayList<Point>();
                for (int i = 0; i < OriginalImgfilter.size(); i++) {

                    double posicionX = OriginalImgfilter.get(i).pt.x + errorx;
                    double posiciony = OriginalImgfilter.get(i).pt.y + errory;
                    posicionX -= pointercentreprueba.x;
                    posiciony -= pointercentreprueba.y;
                    XP = (posicionX * cos) - (posiciony * sin) + pointercentreprueba.x;
                    YP = (posicionX * sin) + (posiciony * cos) + pointercentreprueba.y;
                    keypoint.add(new Point(XP, YP));
                }


        int rango_error = 2;
        List<KeyPoint> keypoints_almacenado = keypoints_imagen.toList();
        ArrayList<KeyPoint> keypoints_coincidencias = new ArrayList<>();

        for (int i = 0; i < keypoint.size(); i++) {
            Point point = keypoint.get(i);
            for (int j = 0; j < keypoints_almacenado.size(); j++) {
                if (point.x >= keypoints_almacenado.get(j).pt.x - rango_error && point.x <= keypoints_almacenado.get(j).pt.x + rango_error && point.y >= keypoints_almacenado.get(j).pt.y - rango_error && point.y <= keypoints_almacenado.get(j).pt.y + rango_error) {
                    keypoints_coincidencias.add(keypoints_almacenado.get(j));
                    break;
                }
            }
        }
        MatOfKeyPoint keypoints_coincidentes = new MatOfKeyPoint();
        keypoints_coincidentes.fromList(keypoints_coincidencias);
        return keypoints_coincidentes;
    }
        //--------------------------------------------------------------------------------------------------------------
    public double hescalado(Mat imagen_original,MatOfKeyPoint keypoints_imagen_original,double escalado){
        MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
        MatOfKeyPoint objectKeyPoi = new MatOfKeyPoint();
//        MatOfKeyPoint imagen_original= new MatOfKeyPoint();
        Mat imagen_original_copia = new Mat();
        Mat image_original_copia = new Mat();
        imagen_original_copia = imagen_original.clone();
//        imagen_original = escalarimag(imagen_original,escalado);
        image_original_copia = escalarimag(imagen_original,escalado);
        Mat imagen_escala_grises = EscalaGrises(imagen_original);
        double valor_umbral = Imgproc.threshold(imagen_escala_grises, new Mat(), 5, 255, Imgproc.THRESH_BINARY);
//        double valor_umbral = Imgproc.threshold(imagen_original, new Mat(), 5, 255, Imgproc.THRESH_BINARY);
        FeatureDetector harris = FeatureDetector.create(FeatureDetector.HARRIS);
        harris.detect(imagen_escala_grises,objectKeyPoints);
        MatOfKeyPoint copia = new MatOfKeyPoint();
        harris.detect(imagen_escala_grises,copia);
        System.out.println("Cantidad de keypoints: "+objectKeyPoints.toArray().length);
        if(objectKeyPoints.toArray().length > 600){
            KeyPoint[] keypoints = objectKeyPoints.toArray();
            ArrayList<KeyPoint> keypoints_filtrados = new ArrayList<>();
            for(int i = 0; i<400; i++){
                keypoints_filtrados.add(keypoints[i]);
            }
            objectKeyPoints = new MatOfKeyPoint();
            objectKeyPoints.fromList(keypoints_filtrados);
        }
        System.out.println("Keypoints con threshold: "+objectKeyPoints.toArray().length);
        double porcentaje=objectKeyPoints.toArray().length-0.1*100/objectKeyPoints.toArray().length;
//        System.out.println("POrcentaje: "+porcentaje);
        System.out.println("porcentaje: "+porcentaje+"%");
        Mat imagen_keypoints = new Mat();
        Scalar color = new Scalar(255,0,0);
        Features2d.drawKeypoints(imagen_original, objectKeyPoints, imagen_keypoints,color,0);
        ArrayList<Point> keypoints_predecidos = predecir_puntos_escalado(imagen_original, keypoints_imagen_original, escalado);
        objectKeyPoints = coincidencias(keypoints_predecidos,copia);
        color = new Scalar(0,0,255);
        Mat imagen_keypoints2 = new Mat();
        Features2d.drawKeypoints(imagen_keypoints, objectKeyPoints, imagen_keypoints2,color,0);
        Imshow.show(imagen_keypoints2);       
//        Highgui.imwrite("salida"+escalado+".jpg", imagen_keypoints2);
        return (objectKeyPoints.toArray().length);
    }
        
        public ArrayList<Point> predecir_puntos_escalado(Mat imagen_original,MatOfKeyPoint keypoints_imagen_original,double escalado){
        double m = 1.2;
        m = Math.pow(m, escalado);
        List<KeyPoint> keypoints = keypoints_imagen_original.toList();
        ArrayList<Point> keypoints_predecidos = new ArrayList<>();
        for(int i = 0; i<keypoints.size(); i++){
            double new_x = keypoints.get(i).pt.x*m;
            double new_y = keypoints.get(i).pt.y*m;
            keypoints_predecidos.add(new Point(new_x,new_y));
        }
        
        return keypoints_predecidos;
    } 

//--------------------------------------------------------------------------------------------------------------
        public MatOfKeyPoint Escalado(ArrayList<KeyPoint> filtradosIMG_Original, double escalado, MatOfKeyPoint keypoint_imagen_prueba){
                double XP = 0;
                double YP = 0;
                ArrayList<Point> keypoint = new ArrayList<>();
                for (int i = 0; i < filtradosIMG_Original.size(); i++) {
                    XP = filtradosIMG_Original.get(i).pt.x * escalado;
                    YP = filtradosIMG_Original.get(i).pt.y * escalado;
                    keypoint.add(new Point(XP, YP));
                }
                int rango_error = 2;
                List<KeyPoint> keypoints_almacenado = keypoint_imagen_prueba.toList();
                ArrayList<KeyPoint> keypoints_coincidencias = new ArrayList<>();

                for (int i = 0; i < keypoint.size(); i++) {
                    Point point = keypoint.get(i);
                    for (int j = 0; j < keypoints_almacenado.size(); j++) {
                        if (point.x >= keypoints_almacenado.get(j).pt.x - rango_error && point.x <= keypoints_almacenado.get(j).pt.x + rango_error && point.y >= keypoints_almacenado.get(j).pt.y - rango_error && point.y <= keypoints_almacenado.get(j).pt.y + rango_error) {
                            keypoints_coincidencias.add(keypoints_almacenado.get(j));
                            break;
                        }
                    }
                }
                MatOfKeyPoint keypoints_coincidentes = new MatOfKeyPoint();
                keypoints_coincidentes.fromList(keypoints_coincidencias);
                return keypoints_coincidentes;
            }
        public Mat rotacionimagen(Mat imagen,double angulo){        
                Point punto = new Point(imagen.width()/2,imagen.height()/2);          
                //System.out.println("Punto central prro2"+punto);        
                Mat r = getRotationMatrix2D(punto,angulo,1.0);
                double cos = Math.abs(r.get(0,0)[0]);
                double sin = Math.abs(r.get(0,1)[0]);        
                int nw = (int) ((imagen.height() * sin) + (imagen.width() * cos));
                int nh = (int) ((imagen.height() * cos) + (imagen.width() * sin));        
                double prro = r.get(0,2)[0] + nw/2.0 - punto.x;
                double prro2 = r.get(1,2)[0] + nh/2.0 - punto.y;        
                r.put(0, 2,prro);
                r.put(1, 2,prro2);        
                Mat salida = new Mat();          
                Imgproc.warpAffine(imagen,salida,r,new Size(nw,nh));
                return salida;
            }
        public double harris_features(Mat imagen_original,MatOfKeyPoint keypoints_imagen_original,double angulo){
                Mat obj=new Mat();
                MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
                MatOfKeyPoint objectKeyPonts = new MatOfKeyPoint();
                Mat imagen_original_copia = new Mat();
                imagen_original_copia = imagen_original.clone();
                imagen_original = rotar_imagen(imagen_original,angulo);
                Mat imagen_escala_grises =EscalaGrises(imagen_original);
                double valor_umbral = Imgproc.threshold(imagen_escala_grises, new Mat(),0, 250, Imgproc.THRESH_BINARY);
                FeatureDetector harris = FeatureDetector.create(FeatureDetector.HARRIS);
                harris.detect(imagen_escala_grises,objectKeyPoints);
                harris.detect(imagen_original,objectKeyPonts);
                MatOfKeyPoint copia = new MatOfKeyPoint();
                MatOfKeyPoint copiau = new MatOfKeyPoint();
                harris.detect(imagen_escala_grises,copia);
                harris.detect(imagen_original,copiau);
                System.out.println("la cantidad de keypoints detectado es: "+objectKeyPonts.toArray().length);
               /* if(objectKeyPoints.toArray().length > 600){
                    KeyPoint[] keypoints = objectKeyPoints.toArray();
                    ArrayList<KeyPoint> keypoints_filtrados = new ArrayList<>();
                    for(int i = 0; i<400; i++){
                        keypoints_filtrados.add(keypoints[i]);
                    }
                    objectKeyPoints = new MatOfKeyPoint();
                    objectKeyPoints.fromList(keypoints_filtrados);
                }*/
                System.out.println("la cantidad de keypoints detectado es: "+objectKeyPoints.toArray().length);
                if(objectKeyPoints.toArray().length > 600){
                    KeyPoint[] keypoints = objectKeyPoints.toArray();
                    ArrayList<KeyPoint> keypoints_filtrados = new ArrayList<>();
                    for(int i = 0; i<400; i++){
                        keypoints_filtrados.add(keypoints[i]);
                    }
                    objectKeyPoints = new MatOfKeyPoint();
                    objectKeyPoints.fromList(keypoints_filtrados);
                }
                System.out.println("la cantidad de keypoints detectados aplicando umbral es: "+objectKeyPoints.toArray().length);
                System.out.println("POrcentaje"+(objectKeyPoints.toArray().length)*100/objectKeyPonts.toArray().length+"%\n");
                Mat imagen_keypoints = new Mat();
                Scalar color = new Scalar(255,0,0);
                Features2d.drawKeypoints(imagen_original, objectKeyPoints, imagen_keypoints,color,0);
                ArrayList<Point> keypoints_predecidos = predecir_puntos(imagen_original, keypoints_imagen_original, angulo,imagen_original_copia);
                //MatOfKeyPoint prediccion_keypoints = predecir_puntos_rotacion2(imagen_original, keypoints_imagen_original, angulo,imagen_original_copia);

                Mat imagen_keypoints2 = new Mat();
                //color = new Scalar(0,255,0);
                //Features2d.drawKeypoints(imagen_keypoints, prediccion_keypoints, imagen_keypoints2,color,0);
                ///////////////////////////////////////////////////////////////////      
                objectKeyPoints =coincidencias(keypoints_predecidos,copia);
                color = new Scalar(0,0,255);        
                Features2d.drawKeypoints(imagen_keypoints, objectKeyPoints, imagen_keypoints2,color,0);
                //Core.circle(imagen_keypoints3, new Point(imagen_keypoints3.width()/2,imagen_keypoints3.height()/2), 3, new Scalar(0,0,255));
                Imshow.show(imagen_keypoints2,"Puntos: "+objectKeyPoints.toArray().length);
        //        Highgui.imwrite("salida imagenes rotadas/salida"+angulo+".jpg", imagen_keypoints2);
                return objectKeyPoints.toArray().length;  
            }

        private Mat EscalaGrises(Mat imagenOriginal) {
                Mat escalaGrises = new Mat();
                Imgproc.cvtColor(imagenOriginal,escalaGrises , Imgproc.COLOR_BGR2GRAY);        
                return escalaGrises;
        }

        private ArrayList<Point> predecir_puntos(Mat imgaevaluar, MatOfKeyPoint keypoints_imagen_original, double angulo, Mat clonacionoriginal) {
                Point punto_central = new Point(imgaevaluar.width()/2,imgaevaluar.height()/2);
                Point punto_central2 = new Point(clonacionoriginal.width()/2,clonacionoriginal.height()/2);
                double diferencia_x = punto_central.x - punto_central2.x;
                double diferencia_y = punto_central.y - punto_central2.y;
                angulo = angulo * -1;
                //Point punto_central2 = new Point(imagenOriginal.width()/2,imagenOriginal.height()/2);
                //System.out.println("punto central prro: "+punto_central);
                List<KeyPoint> keypoints = keypoints_imagen_original.toList();
                ArrayList<Point> keypoints_predecidos = new ArrayList<>();
                for(int i = 0; i<keypoints.size(); i++){
                    double _x = keypoints.get(i).pt.x + diferencia_x;
                    double _y = keypoints.get(i).pt.y + diferencia_y;
                    _x = _x - punto_central.x;
                    _y = _y - punto_central.y;
                    //double valor_x = keypoints.get(i).pt.x - punto_central.x;
                    //double valor_y = keypoints.get(i).pt.y - punto_central.y;
                    double x_prima = _x*Math.cos(Math.toRadians(angulo))-_y*Math.sin(Math.toRadians(angulo));
                    double y_prima = _x*Math.sin(Math.toRadians(angulo))+_y*Math.cos(Math.toRadians(angulo));                        
                    keypoints_predecidos.add(new Point((int)x_prima+punto_central.x,(int)y_prima+punto_central.y));
                    //System.out.println("Valores originales: "+keypoints.get(i).pt+"Nuevos valores"+new Point(x_prima+punto_central.x,y_prima+punto_central.y));
                }     
                return keypoints_predecidos;               
        }

        private MatOfKeyPoint coincidencias(ArrayList<Point> keypoints_predecidos,MatOfKeyPoint imgrotada) {
                int rango_error = 2;
                List<KeyPoint> keypoints_almacenado = imgrotada.toList();
                ArrayList<KeyPoint> coincidenciasdepuntos = new ArrayList<>();
                for(int i = 0; i<keypoints_predecidos.size(); i++){
                    Point point = keypoints_predecidos.get(i);
                    for(int j =0; j<keypoints_almacenado.size(); j++){
                        if(point.x >= keypoints_almacenado.get(j).pt.x-rango_error && point.x <= keypoints_almacenado.get(j).pt.x+rango_error && point.y >= keypoints_almacenado.get(j).pt.y-rango_error && point.y <= keypoints_almacenado.get(j).pt.y+rango_error){
                            //keypoints_almacenado.get(j).pt = point;
                            coincidenciasdepuntos.add(keypoints_almacenado.get(j));                    
                            break;
                        }
                    }            
                }         
                MatOfKeyPoint keypoints_coincidentes = new MatOfKeyPoint();
                keypoints_coincidentes.fromList(coincidenciasdepuntos);
                return keypoints_coincidentes;
        }
        
        public MatOfKeyPoint detection(Mat imagen_original){                              
        MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
        Mat imagen_escala_grises = EscalaGrises(imagen_original);
        double valor_umbral = Imgproc.threshold(imagen_escala_grises, new Mat(), 5, 255, Imgproc.THRESH_BINARY);        
        FeatureDetector harris = FeatureDetector.create(FeatureDetector.HARRIS);  
        harris.detect(imagen_escala_grises,objectKeyPoints);
        System.out.println("la cantidad de keypoints detectado es: "+objectKeyPoints.toArray().length);
        if(objectKeyPoints.toArray().length > 600){
            KeyPoint[] keypoints = objectKeyPoints.toArray();
            ArrayList<KeyPoint> keypoints_filtrados = new ArrayList<>();
            for(int i = 0; i<400; i++){
                keypoints_filtrados.add(keypoints[i]);
            }
            objectKeyPoints = new MatOfKeyPoint();
            objectKeyPoints.fromList(keypoints_filtrados);
        }        
        System.out.println("la cantidad de keypoints detectados aplicando umbral es: "+objectKeyPoints.toArray().length);
        Mat imagen_keypoints = new Mat();
        Scalar color = new Scalar(255,0,0);
        Features2d.drawKeypoints(imagen_original, objectKeyPoints, imagen_keypoints,color,0);
        Imgproc.circle(imagen_keypoints, new Point(imagen_keypoints.width()/2,imagen_keypoints.height()/2), 3, new Scalar(0,0,255));
        Imshow.show(imagen_keypoints);
//        Highgui.imwrite("salida imagenes rotadas/sinrotar.jpg", imagen_keypoints);
        return objectKeyPoints;
    }
//-----------------------------------------------------------------------------------------------------
        public MatOfKeyPoint sift_detection(Mat imagen_original){       
                MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
                Mat imagen_escala_grises = EscalaGrises(imagen_original);
                double valor_umbral = Imgproc.threshold(imagen_escala_grises, new Mat(), 6, 255, Imgproc.THRESH_BINARY);        
                System.out.println(valor_umbral);
                FeatureDetector sift = FeatureDetector.create(FeatureDetector.ORB);  
                sift.detect(imagen_escala_grises,objectKeyPoints);
                System.out.println("la cantidad de keypoints detectado es: "+objectKeyPoints.toArray().length);
                KeyPoint[] keypoints = objectKeyPoints.toArray();
                ArrayList<KeyPoint> keypoints_filtrados = new ArrayList<>();
                for(int i = 0; i<keypoints.length; i++){
                    if(keypoints[i].size > valor_umbral){
                        keypoints_filtrados.add(keypoints[i]);
                    }
                }
                objectKeyPoints.fromList(keypoints_filtrados);
                System.out.println("la cantidad de keypoints detectados aplicando umbral es: "+objectKeyPoints.toArray().length);
                Mat imagen_keypoints = new Mat();
                Scalar color = new Scalar(255,0,0);
                Features2d.drawKeypoints(imagen_original, objectKeyPoints, imagen_keypoints,color,0);
                Imshow.show(imagen_keypoints);           
                return objectKeyPoints;
        }
        public double sift_detection_rotacion(Mat imagen_original,MatOfKeyPoint keypoints_imagen_original,double angulo){
                MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
                Mat imagen_original_copia = new Mat();
                imagen_original_copia = imagen_original.clone();
                imagen_original = rotar_imagen(imagen_original,angulo);
                Mat imagen_escala_grises = EscalaGrises(imagen_original);
                double valor_umbral = Imgproc.threshold(imagen_escala_grises, new Mat(), 6, 255, Imgproc.THRESH_BINARY);
                FeatureDetector sift = FeatureDetector.create(FeatureDetector.ORB);
                sift.detect(imagen_escala_grises,objectKeyPoints);
                MatOfKeyPoint copia = new MatOfKeyPoint();
                sift.detect(imagen_escala_grises,copia);
                System.out.println("la cantidad de keypoints detectado es: "+objectKeyPoints.toArray().length);
                if(objectKeyPoints.toArray().length > 600){
                    KeyPoint[] keypoints = objectKeyPoints.toArray();
                    ArrayList<KeyPoint> keypoints_filtrados = new ArrayList<>();
                    for(int i = 0; i<keypoints.length; i++){
                        if(keypoints[i].size > valor_umbral){
                            keypoints_filtrados.add(keypoints[i]);
                        }                
                    }
                    objectKeyPoints = new MatOfKeyPoint();
                    objectKeyPoints.fromList(keypoints_filtrados);
                }
                System.out.println("la cantidad de keypoints detectados aplicando umbral es: "+objectKeyPoints.toArray().length);
                Mat imagen_keypoints = new Mat();
                Scalar color = new Scalar(255,0,0);
                Features2d.drawKeypoints(imagen_original, objectKeyPoints, imagen_keypoints,color,0);
                ArrayList<Point> keypoints_predecidos = predecirpuntos(imagen_original, keypoints_imagen_original, angulo,imagen_original_copia);
                objectKeyPoints =coincidencias(keypoints_predecidos,copia);
                color = new Scalar(0,0,255);
                Mat imagen_keypoints2 = new Mat();
                Features2d.drawKeypoints(imagen_keypoints, objectKeyPoints, imagen_keypoints2,color,0);
                Imshow.show(imagen_keypoints2);              
                return objectKeyPoints.toArray().length;
        }
        public ArrayList<Point> predecirpuntos(Mat imagen_alterada,MatOfKeyPoint keypoints_imagen_original,double angulo,Mat imagen_original_copia){
                Point punto_central = new Point(imagen_alterada.width()/2,imagen_alterada.height()/2);
                Point punto_central2 = new Point(imagen_original_copia.width()/2,imagen_original_copia.height()/2);
                double diferencia_x = punto_central.x - punto_central2.x;
                double diferencia_y = punto_central.y - punto_central2.y;
                angulo = angulo * -1;
                List<KeyPoint> keypoints = keypoints_imagen_original.toList();
                ArrayList<Point> keypoints_predecidos = new ArrayList<>();
                for(int i = 0; i<keypoints.size(); i++){
                    double valor_x = keypoints.get(i).pt.x + diferencia_x;
                    double valor_y = keypoints.get(i).pt.y + diferencia_y;
                    valor_x = valor_x - punto_central.x;
                    valor_y = valor_y - punto_central.y;
                    double x_prima = valor_x*Math.cos(Math.toRadians(angulo))-valor_y*Math.sin(Math.toRadians(angulo));
                    double y_prima = valor_x*Math.sin(Math.toRadians(angulo))+valor_y*Math.cos(Math.toRadians(angulo));                        
                    keypoints_predecidos.add(new Point((int)x_prima+punto_central.x,(int)y_prima+punto_central.y));
                }     
        return keypoints_predecidos;
        }    
        public MatOfKeyPoint coincidenciaR(ArrayList<Point> keypoints_predecidos,MatOfKeyPoint keypoints_imagen_alterada){
                int rango_error = 2;
                List<KeyPoint> keypoints_almacenado = keypoints_imagen_alterada.toList();
                ArrayList<KeyPoint> keypoints_coincidencias = new ArrayList<>();
                for(int i = 0; i<keypoints_predecidos.size(); i++){
                    Point point = keypoints_predecidos.get(i);
                    for(int j =0; j<keypoints_almacenado.size(); j++){
                        if(point.x >= keypoints_almacenado.get(j).pt.x-rango_error && point.x <= keypoints_almacenado.get(j).pt.x+rango_error && point.y >= keypoints_almacenado.get(j).pt.y-rango_error && point.y <= keypoints_almacenado.get(j).pt.y+rango_error){
                            keypoints_coincidencias.add(keypoints_almacenado.get(j));
                            break;
                        }
                    }
                }              
                MatOfKeyPoint keypoints_coincidentes = new MatOfKeyPoint();
                keypoints_coincidentes.fromList(keypoints_coincidencias);
                return keypoints_coincidentes;        
        }

}