/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ooeraciones.morfologicas;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.calib3d.Calib3d;

import java.io.File;
import java.util.LinkedList;
import java.util.List;
import java.io.File;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import org.opencv.core.Core;
import org.opencv.highgui.Highgui;
import org.opencv.imgcodecs.Imgcodecs;
import static org.opencv.imgproc.Imgproc.line;

/**
 *
 * @author Alex_Salazar_M
 */
public class OoeracionesMorfologicas extends Application{
        /**
         * @param args the command line arguments
         */
        static {
                System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        }
        /**
         *
         * @param stage
         * @throws Exception
         */
        @Override
        public void start(Stage stage) throws Exception {
                Parent root = FXMLLoader.load(getClass().getResource("OperacionesMorfologicas.fxml"));
                
                Scene scene = new Scene(root);
                
                stage.setScene(scene);

                stage.show();
        }
        /**
         * @param args the command line arguments
         */
        public static void main(String[] args) {
                // TODO code application logic here
                launch(args);
                File lib = null;
        String os = System.getProperty("os.name");
        String bitness = System.getProperty("sun.arch.data.model");


        //ROTACION CON RESPECTO AL ORIGEN


//        System.load(lib.getAbsolutePath());

        
        }
}
        

