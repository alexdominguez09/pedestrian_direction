/*

This program extract and saves images from a given video.
The images that extract are pedestrian moving to right, left or front.
It is based in HOG to detect a pedestrian, and optical flow to detect the pedestrian is moving.
It then saves all or some of the images in the process.
This code assumes it has opencv 2.4 installed.

*/

    #include <stdio.h>
    #include <iostream>
    #include <math.h>
    #include <opencv2/opencv.hpp>
    #include <opencv2/core/core.hpp>
    #include <opencv2/highgui/highgui.hpp>
    #include <opencv2/video/background_segm.hpp>
    #include <string>
    #include <stdlib.h>
    #include <sstream>
    #include <iomanip>




    using namespace cv;
    using namespace std;

    // Function to draw the optical flow.
    double drawOptFlowMap (const Mat& flow, Mat& cflowmap, int step, const Scalar& color)
    {
    double length = 0;
    int number_vectors = 0;

    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
            {
                const Point2f& fxy = flow.at< Point2f>(y, x);
                line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color,1);
                 // circle(cflowmap, Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, color, -1);
                length = length + sqrt ( pow( abs (x - cvRound(x+fxy.x)), 2) + pow( abs( y - cvRound(y+fxy.y)) , 2)  );
            }

           number_vectors = (cflowmap.rows/step )* (cflowmap.cols/step)  ;
           length = length / number_vectors;  // standarizar algo el valor de la suma de vectores

          return length;
          }  //  END of drawOptFlowMap


    int main()
    {
     double s=1;  // resizing factor
     int count_frame = 0;
     double optical_flow_total = 0;
     double optical_flow_threshold = 0.35 ;
     //global variables
     cv::Mat GetImg, GetImg2;
     cv::Mat prvs, next, next2; //current frame
     const char* video_location = "/home/alex/opencv/codeblocks/Test01/bin/Debug/MVI_biblio_432_30fps_sharp_mix.avi" ;
     const char* saving_location ="/home/alex/opencv/codeblocks/Test01/images/frame_A_";
     const char* saving_location_D ="/home/alex/opencv/codeblocks/Test01/images/frame_D_";
     const char* saving_location_E ="/home/alex/opencv/codeblocks/Test01/images/frame_E_";
     const char* saving_location_F ="/home/alex/opencv/codeblocks/Test01/images/frame_F_";

     //  Hog variables -----------------------.
     cv::Mat img;
     HOGDescriptor hog;
     hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
     cv::Mat originalImage;
	 cv::Rect faceRect;
	 cv::Mat croppedFaceImage;
     unsigned int n = 0;
     stringstream filename, filename_D, filename_E, filename_F;

     // Substract variables ------------
     cv::Mat img_D, img_E, img_F;

    // Angle variables ----------------------
    float MHI_DURATION = 1; // originally 0.05
    int DEFAULT_THRESHOLD = 32; // originally 32
    float MAX_TIME_DELTA = 0.5;  // originally 12500.0
    float MIN_TIME_DELTA = 0.05;  // originally 0.05
    vector<Rect> seg_bounds;
    cv::Mat silh_roi,orient_roi,mask_roi,mhi_roi;
    cv::Mat frame,ret,frame_diff,gray_diff,motion_mask;
    bool angle_out;

    // char fileName[250] ="/media/alex/Elements/HD300gb/tests_doctorado/entornos/entorno1/entorno_01_03.avi";
    //VideoCapture stream1(fileName);
    VideoCapture stream1(video_location);
    //    VideoCapture stream1(0);  //0 is the id of video device.0 if you have only one camera
    //    VideoCapture stream2(fileName);  // new -----------------


     if(!(stream1.read(GetImg))  ) //get one frame from video
      return 0;


     // Access video frames variables ------------
     double frnb ( stream1.get ( CV_CAP_PROP_FRAME_COUNT ) );
     cout << "Frame count total = " << frnb << endl;
     cout << "FPS = " << stream1.get ( CV_CAP_PROP_FPS ) << endl  ;
     cout << "Format = " << stream1.get ( CV_CAP_PROP_FORMAT ) << endl  ;
     cout << "FOURCC = " << stream1.get ( CV_CAP_PROP_FOURCC ) << endl  ;
     cout << "Optical flow threshold = " << optical_flow_threshold << endl  ;


     resize(GetImg, prvs, Size(GetImg.size().width/s, GetImg.size().height/s) );
     cvtColor(prvs, prvs, CV_BGR2GRAY);

        // Angle stuff ----------------------
        stream1.read(frame);
		Size frame_size = frame.size();
		int h = frame_size.height;
		int w = frame_size.width;

		cout << w << " x " << h << endl; // new -------------
		cout <<  "frame, angle, flow " << endl;
		cout << "test outttttttt "  << endl; //  00000000000000000000000000000000000000000000000
        cv::Mat prev_frame = frame.clone();
        cv::Mat motion_history(h,w, CV_32FC1,Scalar(0,0,0));
        cv::Mat mg_mask(h,w, CV_8UC1,Scalar(0,0,0));
        cv::Mat mg_orient(h,w, CV_32FC1,Scalar(0,0,0));
        cv::Mat seg_mask(h,w, CV_32FC1,Scalar(0,0,0));



     // unconditional loop
     while (true) {

      if(!(stream1.read(GetImg))  ) //get one frame from video
       break;
        stream1.read(GetImg2);  // new -----------------
        resize(GetImg2, next2, Size(GetImg2.size().width/s, GetImg2.size().height/s) ); // new -----------------
        //  cvtColor(next2, next2, CV_BGR2GRAY);  // new -----------------


       double count = stream1.get(CV_CAP_PROP_POS_FRAMES);
       cout << std::fixed << std::setprecision(0) << count << ","  ;



        // Angle stuff ----------------------
        //	double timestamp = 1000.0*clock()/CLOCKS_PER_SEC;
       	double timestamp = (double)clock()/CLOCKS_PER_SEC;
       	ret = frame.clone();

       	absdiff(frame, prev_frame, frame_diff);
        cvtColor(frame_diff,gray_diff, CV_BGR2GRAY );
        threshold(gray_diff,ret,DEFAULT_THRESHOLD,255,0);
       	motion_mask = ret.clone();
        //  	imshow("prev_frame", prev_frame);  // is static.. dosent move


        // Motion history for Optical flow.
        // motion_history = cv::Scalar(255,255,255);
       	updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION);
		calcMotionGradient(motion_history, mg_mask, mg_orient, MIN_TIME_DELTA, MAX_TIME_DELTA, 3);
		segmentMotion(motion_history, seg_mask, seg_bounds, timestamp, 32);


        angle_out = false;
        for(unsigned int h = 0; h < seg_bounds.size(); h++)
        {
            Rect rec = seg_bounds[h];

            // To show the seg_bounds, area of motion
             rectangle(prev_frame, rec, cv::Scalar(0,255,0), 1);
             imshow("prev_frame", prev_frame);
             cout << "test outttttttt 3333"  << endl; //  00000000000000000000000000000000000000000000000

            //	 cout << "h -- " << h << endl;  //new -------------------
            //  if(rec.area() > 5000 && rec.area() < 70000)
            if(rec.area() > 5000 && rec.area() < 80000)
			{
				silh_roi = motion_mask(rec);
				orient_roi = mg_orient(rec);
				mask_roi = mg_mask(rec);
				mhi_roi = motion_history(rec);
			    //	if(norm(silh_roi, NORM_L2, noArray()) > rec.area()*0.5)
			    cout << "test outttttttt 2222"  << endl; //  00000000000000000000000000000000000000000000000
				if(angle_out == false)
				{
					double angle = calcGlobalOrientation(orient_roi, mask_roi, mhi_roi,timestamp, MHI_DURATION);
				    //	cout << " Angle >: " << angle ;
				    cout  << std::fixed << std::setprecision(0) << angle ;
					angle_out = true;
					//cout << "Angle: " << angle << endl;
                }
                else
                {
					double angle = calcGlobalOrientation(orient_roi, mask_roi, mhi_roi,timestamp, MHI_DURATION);
					cout << " Angle <: " << angle ;
					angle_out = true;
					//cout << "Angle: " << angle << endl;
                }

			}
		}       // Angle stuff ---------------------- END -----------

      //      imshow("motion_history", motion_history);
   /*   cvResetImageROI( mhi_roi );
        cvResetImageROI( orient_roi );
        cvResetImageROI( mask_roi );
        cvResetImageROI( silh_roi );  */
  //      seg_mask = seg_mask.clone();     // how to reset images



     // HOG images  ----------------
      stream1 >> img;
      originalImage = img;
      vector<Rect> found, found_filtered;

      //Resize
      resize(GetImg, next, Size(GetImg.size().width/s, GetImg.size().height/s) );
      cvtColor(next, next, CV_BGR2GRAY);



      //    stream2.set(CV_CAP_PROP_POS_FRAMES, count+1);
      //   resize(GetImg2, next2, Size(GetImg2.size().width/s, GetImg2.size().height/s) );

      cvtColor(next2, next2, CV_BGR2GRAY);

      ///////////////////////////////////////////////////////////////////
      cv::Mat flow;
       calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

      cv::Mat cflow;
      cvtColor(prvs, cflow, CV_GRAY2BGR);
      optical_flow_total = drawOptFlowMap(flow, cflow, 10, CV_RGB(255, 0, 0)) ;
     // if (optical_flow_total > optical_flow_threshold)  cout << "Flow: " << optical_flow_total << endl  ; // print out optical flow index


     //   cout << " Flow: " << optical_flow_total << endl  ; // print out optical flow index
        cout  << "," << std::fixed << std::setprecision(3) << optical_flow_total << endl;
       // Hog -----------ONLY if there is movement -------
       if (optical_flow_total > optical_flow_threshold)
        {
        hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);  // detecting HOG pedestrian
        size_t i, j;
        for (i=0; i<found.size(); i++)
        {
            Rect r = found[i];

            for (j=0; j<found.size(); j++)
                if (j!=i && (r & found[j])==r)
                    break;
            if (j==found.size())
                found_filtered.push_back(r);
        }

        for (i=0; i<found_filtered.size(); i++)
        {
	    Rect r = found_filtered[i];

        r.x += cvRound(r.width*0.1);
	    r.width = cvRound(r.width*0.8);
	    r.y += cvRound(r.height*0.06);
	    r.height = cvRound(r.height*0.9);


        // substract image and ADD image --------------------
        cv::subtract(next, prvs,img_D);
        cv::subtract(next2, next,img_E); // new -----------------
        cv::add(img_D, img_E,img_F); // new -----------------

        // Show images
        imshow("Substract image D", img_D);
        imshow("Substracted image E", img_E);
        imshow("Added D+E image F", img_F);
        imshow("video capture", img);

        //  save image -----------------------
  	    faceRect = r;

        if    (0 <= r.x &&  0 <= r.width  && r.x + r.width <= originalImage.cols && 0 <= r.y && 0 <= r.height && r.y + r.height <= originalImage.rows )
           {
            cv::Mat croppedFaceImage = originalImage(faceRect);
            n++;
            filename.str("");
         //   filename_D.str("");
         //   filename_E.str("");
         //   filename_F.str("");
            //filename << "/home/alex/opencv/codeblocks/Test01/images/frame_F_"  << n << ".jpg";
            filename << saving_location << n << ".jpg";
       //     filename_D << saving_location_D << n << ".jpg";
       //     filename_E << saving_location_E << n << ".jpg";
        //    filename_F << saving_location_F << n << ".jpg";

           //  cv::imwrite(filename.str(), croppedFaceImage);
           cv::imwrite(filename.str(), img); // new -----------------
      //     cv::imwrite(filename_D.str(), img_D); // new -----------------
       //    cv::imwrite(filename_E.str(), img_E); // new -----------------
       //    cv::imwrite(filename_F.str(), img_F); // new -----------------
           }

	   // rectangle(img, r.tl(), r.br(), cv::Scalar(255,0,0), 1);  // draw rectangle in HOG ...............
        }

       }
	   // Hog ------------------ end ---------------

  	  imshow("video capture", img);
  	  imshow("OpticalFlowFarneback", cflow);

        //    imshow("Substracted image D", img_D);
        //    imshow("Substracted image E", img_E); // new -----------------
        //	  waitKey();
        //    imshow("Added image E", img_F); // new -----------------
        //    imshow("frame", frame);  // new -----------------


      count_frame ++;   //   cout << count_frame << endl ;


      // -------------- Display ----------------
      //    imshow("mg_orient", mg_mask);
      if (waitKey(5) >= 0)
       break;

      prvs = next.clone();
      prev_frame = frame.clone();  // prev_frame clone was missing
     }
        return 0;
    }
