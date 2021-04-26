// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <utility>
#include <functional>
#include <fstream>


std::vector<std::pair<std::string, std::function <void() > > > functionSet;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

/*laboratory 1*/

uchar bound(int val)
{
	
	if (val > 255)
		return 255;
	if (val < 0)
		return 0;
	return val;

}



void lab1Problem3()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		const char additiveFactor = 50;
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
	
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int tmp = val+ additiveFactor;
				val = bound(tmp);
			
				dst.at<uchar>(i, j) = val;
			}
		}
		imshow("input image", src);
		imshow("additive factor image", dst);
		waitKey();
	}
}



void lab1Problem4()
{
	char fname[MAX_PATH];
	const float multiplicativeFactor = 0.5;
	while (openFileDlg(fname)) {
		
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				val = bound((int)(val * multiplicativeFactor));
				dst.at<uchar>(i, j) = val;
			}
		}

		bool result = true;
		try
		{
			result = imwrite("problem4.bmp", dst);
			

		}
		catch (const cv::Exception& ex)
		{
			fprintf(stdout, "Exception saving the image: %s\n", ex.what());
		}

		imshow("input image", src);
		imshow("multiplicative factor image", dst);
		waitKey();
	}
}


void lab1Problem5()
{
	const uchar multiplicativeFactor = 2;
	

	
		int height = 256;
		int width = 256;
		Mat dst = Mat(height, width, CV_8UC3);
		Vec2b midPoint(256 / 2, 256 / 2);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b val;
				if (i < midPoint[0] && j < midPoint[1])
				{
					val= Vec3b(255, 255, 255);
				}
				else if (i < midPoint[0] && j >= midPoint[1])
				{
					val = Vec3b(0, 0, 255);
				}
				else if (i >= midPoint[0] && j < midPoint[1])
				{
					val = Vec3b(0, 255, 0);
				}
				else
				{
					val = Vec3b(0, 255, 255);
				}

				dst.at<Vec3b>(i, j) = val;
			}
		}
		
		imshow("4 square image", dst);
		waitKey();
	
}

void lab1Problem6()
{
	int height = 3;
	int width = 3;
	Mat_ <float> src (3,3);

	setIdentity(src);
	src[0][0] = 3;
	
	std::cout << "Matrix: \n" << src << "\n";
	Mat_ <float >inverse= src.inv();
	std::cout << "Inverse Matrix \n" << inverse << "\n";
	char buff;
	std::cin >> buff;

}

void lab2Problem1()
{

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat dstB = Mat(height, width, CV_8UC1);
		Mat dstR = Mat(height, width, CV_8UC1);
		Mat dstG = Mat(height, width, CV_8UC1);


		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b val = src.at<Vec3b>(i, j);
				dstB.at<uchar>(i, j) = val[0];
				dstG.at<uchar>(i, j) = val[1];
				dstR.at<uchar>(i, j) = val[2];
	

			}
		}

		imshow("input image", src);
		imshow("B", dstB);
		imshow("R", dstR);
		imshow("G", dstG);
		waitKey();
	}
}


void lab2Problem2()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);
	


		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b val = src.at<Vec3b>(i, j);
				dst.at<uchar>(i, j) = (val[0]+val[1]+val[2])/3;
	

			}
		}
		imshow("input image", src);
		imshow("grayscale", dst);
		waitKey();
	}
}

void lab2Problem3()
{
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);
		uchar treshold;
		std::cout << "treshold:";
		std::cin >> treshold;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				dst.at<uchar>(i, j) = (val < treshold) ? 0 : 255;
			}
		}
		imshow("input image", src);
		imshow("grayscale", dst);
		waitKey();
	}
}

void lab2Problem4()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);


		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b val = src.at<Vec3b>(i, j);
				float b = (float)val[0] / 255;
				float g = (float)val[1] / 255;
				float r = (float)val[2] / 255;
				float M = fmax(fmax(b, g), r);
				float m = fmin(fmin(b, g), r);

				float C = M - m;

				float V = M;
				float S= (V!=0)? C/V :0;

				float H;
				if (C != 0)
				{
					if (M == r)
					{
						H = 60 * (g - b) / C;
					}
					if (M == g)
					{
						H = 120 + 60 * (b - r) / C;
					}
					if (M == b)
					{
						H = 240 + 60 * (r - g) / C;
					}
				}
				else
				{
					H = 0;
				}

				if (H < 0)
					H += 360;
				// normalize
				uchar H_norm = H * 255 / 360;
				uchar S_norm = S * 255;
				uchar V_norm = V * 255;
				dstH.at<uchar>(i, j) = H_norm;
				dstS.at<uchar>(i, j) = S_norm;
				dstV.at<uchar>(i, j) = V_norm;
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}

}

bool isInside(Mat image,int i, int j)
{

	return (0 <= i && i < image.rows && 0 <= j && j < image.cols);

}

void testIsInside()
{
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_COLOR);
		
		int i = 0,j=0;
		std::cout << "test isInside with " << i << " " << j << " result " << isInside(src, i, j) << "\n";

		i = 3000, j = 12000;
		std::cout << "test isInside with " << i << " " << j << " result " << isInside(src, i, j) << "\n";


		i = -10, j = 0;
		std::cout << "test isInside with " << i << " " << j << " result " << isInside(src, i, j) << "\n";
		imshow("input image", src);
	
		waitKey();
	}
}






std::vector<int> computeHistogram(Mat image){
	std::vector <int> hist(256, 0);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			uchar val = image.at<uchar>(i, j);
			hist[val]++;
		}
	}
	return hist;
}

std::vector<int> computeHistogram(Mat image,int bins_nr) {
	std::vector <int> hist(bins_nr, 0);
	int bin_size = 256 / bins_nr;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			uchar val = image.at<uchar>(i, j);
			int position = val / bin_size;
			hist[position]++;
		}
	}
	return hist;
}

std::vector<float> computePDF(Mat image)
{
	std::vector<float> pdf(256,0.0f);
	std::vector<int> hist;
	hist = computeHistogram(image);
	for (int i = 0; i <= 255; i++) {
		pdf[i] = (float)hist[i] / (float)(image.cols * image.rows);
	}
	return pdf;
}

void displayHistogram()
{
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		
		std::vector<int> histogram;
		histogram=computeHistogram(src);
		showHistogram("pixel intensity", histogram.data(), histogram.size(), 200);
		imshow("input image", src);

		waitKey();
	}
}

void displayBinHistogram()
{
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		
		unsigned int m;
		std::cout << "introduce the number of bins:";
		std::cin >> m;

		std::vector<int> histogram;
		histogram=computeHistogram(src, m);
		showHistogram("pixel intensity", histogram.data(), histogram.size(), 200);
		imshow("input image", src);

		waitKey();

	}
}


std::vector<int> computeLocalMaxima(int wh, float threshold,float pdf[])
{

	std::vector<int> local_maxima;
	local_maxima.push_back(0);

	for (int k=wh;k<=255-wh;k++)
	{
		float v = 0.0f;
		float maximum = 0.0f;
		int flag = 1;
		for (int i = k - wh; i <= k + wh; i++){
			v += pdf[i];

			if (pdf[i] > pdf[k])
				flag = 0;
		}

		v = v / (2 * wh + 1);
		if (pdf[k] > (v + threshold) && flag == 1)
		{
			local_maxima.push_back(k);

		}
	}
	local_maxima.push_back(255);

	return local_maxima;
}

int findClosestValue(std::vector<int> vec, int target)
{
	int distance;
	int ans=0;
	int min_distance = INT_MAX;
	for (auto it : vec)
	{
		distance = abs(target - it);
		if (min_distance > distance)
		{
			min_distance = distance;
			ans = it;
		}
	}
	//std::cout << target << " " << ans << "\n";
	return ans;
}
void multiLevelTreshold(Mat src,Mat &dst)
{
	dst = Mat(src.rows, src.cols, CV_8UC1);

	std::vector<float> pdf = computePDF(src);

	std::vector<int> local_maxima = computeLocalMaxima(5, 0.0003, pdf.data());

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			uchar val = src.at<uchar>(i, j);
			dst.at<uchar>(i, j) = (uchar)findClosestValue(local_maxima, val);
		}
	}
}
void showMultiLevelTreshold()
{
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_GRAYSCALE);

		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		multiLevelTreshold(src, dst);

		imshow("input image", src);
		imshow("multi level image", dst);
		std::vector<int> hist;
		hist=computeHistogram(dst);
		showHistogram("multi level", hist.data(), 255, 200);

		waitKey();
	}
}


void showFloydSteinberg()
{
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		//multiLevelTreshold(src, dst);
		dst = src;
		std::vector<float> pdf = computePDF(src);

		std::vector<int> local_maxima = computeLocalMaxima(5, 0.0003, pdf.data());

		imshow("input image", src);

		Mat levelTreshold = Mat(src.rows, src.cols, CV_8UC1);
		multiLevelTreshold(src, levelTreshold);
		imshow("levelTreshold", levelTreshold);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				uchar oldpixel = src.at<uchar>(i, j);
				uchar newpixel = findClosestValue(local_maxima, oldpixel);
				dst.at<uchar>(i, j) = newpixel;

				int error = oldpixel - newpixel;
				//std::cout << error << " ";
				if (isInside(dst, i, j + 1)) {
					dst.at<uchar>(i, j + 1) = bound(dst.at<uchar>(i, j + 1) + (7 * error) / 16);
				}

				if (isInside(dst, i+1, j-1))
					dst.at<uchar>(i + 1, j-1) = bound(dst.at<uchar>(i + 1, j-1) + (3 * error )/ 16);

				if (isInside(dst, i + 1, j	))
					dst.at<uchar>(i + 1, j) = bound(dst.at<uchar>(i + 1, j) + (5 * error )/ 16);

				if (isInside(dst, i + 1, j+1))
					dst.at<uchar>(i + 1, j+1) = bound(dst.at<uchar>(i + 1, j+1) +  error / 16);

			}
		}
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
			}
		}


		imshow("dither image", dst);
		

		waitKey();
	}
}
/*
lab 3 problem 7
*/
void showMultiLevelTresholdHSV()
{
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_COLOR);
		Mat HSVimage;
		cvtColor(src, HSVimage, COLOR_BGR2HSV);
		
		Mat H = Mat(src.rows, src.cols, CV_8UC1);
		//extract H
		for (int i = 0; i < src.rows;i++){
			for (int j = 0; j < src.cols; j++){
				Vec3b pixel = HSVimage.at<Vec3b>(i, j);
				uchar hvalue = pixel[0];
				H.at<uchar>(i, j) = hvalue;
			}
		}

		std::vector<int> hist;
		hist = computeHistogram(H);
		showHistogram("h histo", hist.data(), 255, 200);

		Mat HThreshold = Mat(src.rows, src.cols, CV_8UC1);;
		multiLevelTreshold(H, HThreshold);

		hist = computeHistogram(HThreshold);
		showHistogram("multi level", hist.data(), 255, 200);

		Mat HSVout = Mat(src.rows, src.cols, CV_8UC3);

		//combine HSV
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				Vec3b pixel = HSVimage.at<Vec3b>(i, j);
				pixel[0] = HThreshold.at<uchar>(i, j);
				HSVout.at<Vec3b>(i, j) = pixel;
			}
		}
		Mat BGRout= Mat(src.rows, src.cols, CV_8UC3);
		cvtColor(HSVout, BGRout, COLOR_HSV2BGR);

		imshow("input image", src);
		imshow("H", H);
		imshow("H treshold", HThreshold);
		imshow("HVS out", HSVout);
		imshow("HVS in", HSVimage);
		imshow("BGR out", BGRout);




		waitKey();
	}
}

/*lab4*/

int area(Mat label_image)
{
	int total_area = 0;
	for (int i = 0; i < label_image.rows; i++)
	{
		for (int j = 0; j < label_image.cols; j++)
		{
			uchar val = label_image.at<uchar>(i, j);
			if (val == 1)
			{
				total_area++;
			}
		}
	}
	return total_area;
}

std::pair<float, float> centerMass(Mat label_image)
{
	float center_row = 0.0f;
	float center_col = 0.0f;

	for (int i = 0; i < label_image.rows; i++)
	{
		for (int j = 0; j < label_image.cols; j++)
		{
			uchar val = label_image.at<uchar>(i, j);
			center_row += i * val;
			center_col += j * val;
		}
	}
	int object_area = area(label_image);
	center_row = center_row / object_area;
	center_col = center_col / object_area;
	return std::make_pair(center_row, center_col);
}

float elongationAxis(Mat label_image)
{
	float numerator = 0.0f;
	float denominator1 = 0.0f;
	float denominator2 = 0.0f;
	std::pair<float, float> center = centerMass(label_image);

	for (int r = 0; r < label_image.rows; r++)
	{
		for (int c = 0; c < label_image.cols; c++)
		{
			uchar I = label_image.at<uchar>(r, c);
			numerator += (r - center.first) * (c - center.second) * I;
			denominator1 += (c - center.second) * (c - center.second) * I;
			denominator2 += (r - center.first) * (r - center.first) * I;

		}
	}
	numerator *= 2;
	float tan_2phi = numerator/(denominator1 - denominator2);
	float phi = atan(tan_2phi) / 2;
	return phi;
}

bool isOnContour(Mat label_image,int r,int c)
{
	int dx[] = { -1,-1,-1,0,0,1,1,1 };
	int dy[] = { -1,0,1,-1,0,1,-1,0,1 };
	for (int i = 0; i < 8; i++)
	{
		int new_r = r + dx[i];
		int new_c = c + dy[i];
		if (isInside(label_image, new_r, new_c))
		{
			if (label_image.at<uchar>(new_r, new_c) == 0)
			{
				return true;
			}
		}
	}
	return false;
}

float perimeter(Mat label_image)
{

	int perimeter_size = 0;
	for (int r = 0; r < label_image.rows; r++)
	{
		for (int c = 0; c < label_image.cols; c++)
		{
			uchar I = label_image.at<uchar>(r, c);
			if (isOnContour(label_image,r,c) && I == 1)
			{
				perimeter_size++;
			}

		}
	}
	return perimeter_size;
}

float thinnessRatio(Mat label_image)
{
	float p= perimeter(label_image);
	float t = 4 * PI*(area(label_image) / p*p );
	return t;
}

float aspectRatio(Mat label_image)
{
	int c_max = 0;
	int r_max = 0;
	int c_min = label_image.cols;
	int r_min = label_image.rows;

	for (int r = 0; r < label_image.rows; r++)
	{
		for (int c = 0; c < label_image.cols; c++)
		{
			uchar I = label_image.at<uchar>(r, c);
			if (I == 1) {
				c_max = max(c_max, c);
				r_max = max(r_max, r);
				c_min = min(c_min, c);
				r_min = min(r_min, r);
			}
		}
	}
	float R = (float)((c_max - c_min + 1) /( r_max - r_min + 1));

	return R;
}

int projection_h(Mat label_image,int r)
{
	int h = 0;
	for (int c = 0; c < label_image.cols; c++)
	{
		uchar I = label_image.at<uchar>(r, c);
		if (I == 1) {
			h++;
		}
	}
	return h;
}

int projection_v(Mat label_image, int c)
{
	int v = 0;
	for (int r = 0; r < label_image.rows; r++)
	{
		uchar I = label_image.at<uchar>(r, c);
		if (I == 1) {
			v++;
		}
	}
	return v;
}


Mat getLabelImage(Mat image,Vec3b color)
{
	Mat label(image.rows, image.cols, CV_8UC1);
	for (int r = 0; r < image.rows; r++)
	{
		for (int c = 0; c < image.cols; c++)
		{
			Vec3b col=image.at<Vec3b>(r, c);
			if (col == color){
				label.at<uchar>(r, c) = 1;
			}
			else {
				label.at<uchar>(r, c) = 0;
			}
		}
	}
	return label;
}

void computeFeaturesAllObjects()
{
	char fname[MAX_PATH];
	std::vector<Mat> label_images;
	while (openFileDlg(fname)) {

		Mat src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);


		// label images based on color
		//gray
		label_images.push_back(getLabelImage(src, Vec3b(128, 128, 128)));
		//red 
		label_images.push_back(getLabelImage(src, Vec3b(0, 0, 255)));
		//light blue		
		label_images.push_back(getLabelImage(src, Vec3b(192, 128, 0)));
		//purple
		label_images.push_back(getLabelImage(src, Vec3b(255, 0, 255)));
		//  blue
		label_images.push_back(getLabelImage(src, Vec3b(255, 0, 0)));
		//green
		label_images.push_back(getLabelImage(src, Vec3b(0, 255, 0)));
		//brown
		label_images.push_back(getLabelImage(src, Vec3b(0, 64, 128)));



		for (int i = 0; i < label_images.size(); i++) {
			std::cout << "object " << i << " has the following:\n";
			std::cout << "area " << area(label_images[i]) << "\n";
			std::cout << "centerMass " << centerMass(label_images[i]).first << " " << centerMass(label_images[i]).second << "\n";
			std::cout << "elongationAxis " << elongationAxis(label_images[i]) << "\n";
			std::cout << "perimeter " << perimeter(label_images[i]) << "\n";
			std::cout << "thinnessRatio " << thinnessRatio(label_images[i]) << "\n";
			std::cout << "aspectRatio " << aspectRatio(label_images[i]) << "\n";
		
		}

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);


		//show the image
		imshow("My Window", src);

		// Wait until user press some key

		waitKey(0);
	}
}


void drawcontour(Mat &dst, Mat label, Vec3b color)
{	
	for (int r = 0; r < dst.rows; r++)
	{
		for (int c = 0; c < dst.cols; c++)
		{
			if (isOnContour(label, r, c) && label.at<uchar>(r, c) == 1)
			{
				dst.at<Vec3b>(r, c) = color;
			}
		}
	}
}

void MyCallBackFunc2(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		std::cout << src->rows << " " << src->cols << "\n";

		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);

		Vec3b pixel = (*src).at<Vec3b>(y, x);

		Mat label_image = getLabelImage(*src,pixel);
		std::pair<float, float> center_mass = centerMass(label_image);

		std::cout << "object with color " << pixel << " has the following:\n";
		std::cout << "area " << area(label_image) << "\n";
		std::cout << "centerMass " << center_mass.first << " " << center_mass.second << "\n";
		std::cout << "elongationAxis " << elongationAxis(label_image) << "\n";
		std::cout << "perimeter " << perimeter(label_image) << "\n";
		std::cout << "thinnessRatio " << thinnessRatio(label_image) << "\n";
		std::cout << "aspectRatio " << aspectRatio(label_image) << "\n";

		Mat dst = *src;
		drawcontour(dst, label_image, Vec3b(0, 0, 0));
		dst.at<Vec3b>((int)center_mass.first, (int)center_mass.second) =Vec3b(0,0,0);
		float phi = elongationAxis(label_image);
		float slope = tan(phi);
		Point p1(center_mass.first, center_mass.second);

		float y2 = 0;
		float  x2 = p1.x - (p1.y - y2) / slope;
		
		Point p2(x2, y2);

		std::cout << p1 << " " << p2 << "line \n";
		line(dst, p1, p2, Vec3b(0, 0, 0), 3);
		

		std::cout << "projection_h : " << projection_h(label_image, y) << "\n";
		std::cout << "projectoin_v : " << projection_v(label_image, x) << "\n";
		imshow("object selected", dst);

	}
}

void computeClickFeaturesObject()
{
	char fname[MAX_PATH];
	std::vector<Mat> label_images;
	while (openFileDlg(fname)) {

		Mat src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc2, &src);

		//show the image
		imshow("My Window", src);
		// Wait until user press some key

		waitKey(0);
	}
}

void deleteObjects(Mat & image,Vec3b overwriteColor,Mat label)
{
	for (int r = 0; r < image.rows; r++)
	{
		for (int c = 0; c < image.cols; c++)
		{
			uchar I = label.at<uchar>(r, c);

			if (I == 1) {
				image.at<Vec3b>(r, c)= overwriteColor;
			}
		}
	}
}
void selectObjects()
{
	char fname[MAX_PATH];
	std::vector<Mat> label_images;
	while (openFileDlg(fname)) {
		int TH_area;
		float phi_LOW, phi_HIGH;
		std::cout << "TH_area:";
		std::cin >> TH_area;
		std::cout << " phi_LOW:";
		std::cin >> phi_LOW;
		std::cout << " phi_HIGH:";
		std::cin >> phi_HIGH;

		Mat src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);
		// label images based on color
		//gray
		label_images.push_back(getLabelImage(src, Vec3b(128, 128, 128)));
		//red 
		label_images.push_back(getLabelImage(src, Vec3b(0, 0, 255)));
		//light blue		
		label_images.push_back(getLabelImage(src, Vec3b(192, 128, 0)));
		//purple
		label_images.push_back(getLabelImage(src, Vec3b(255, 0, 255)));
		//  blue
		label_images.push_back(getLabelImage(src, Vec3b(255, 0, 0)));
		//green
		label_images.push_back(getLabelImage(src, Vec3b(0, 255, 0)));
		//brown
		label_images.push_back(getLabelImage(src, Vec3b(0, 64, 128)));

		for (int i = 0; i < label_images.size(); i++) {
			std::cout << "object " << i << " has the following:\n";
			std::cout << "area " << area(label_images[i]) << "\n";
			std::cout << "elongationAxis " << elongationAxis(label_images[i]) << "\n";
			if (!(area(label_images[i]) < TH_area && elongationAxis(label_images[i]) >= phi_LOW && elongationAxis(label_images[i]) <= phi_HIGH) )
			{
				std::cout << "object " << i << "deleted\n";
				deleteObjects(src, Vec3b(255, 255, 255), label_images[i]);
			}
		}

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);


		//show the image
		imshow("My Window", src);

		// Wait until user press some key

		waitKey(0);
	}
}


std::vector<Point2i> getNeighbors4(Point2i pos, Mat image) {
	int dx[] = { -1,0,0,1 };
	int dy[] = { 0,-1,1,0};
	std::vector<Point2i> result;
	for (int i = 0; i < 8; i++)
	{
		int new_x = pos.x + dx[i];
		int new_y = pos.y + dy[i];
		if (isInside(image, new_x, new_y))
		{
			result.push_back(Point2i(new_x, new_y));
		}
	}
	return result;
}


std::vector<Point2i> getNeighbors8(Point2i pos, Mat image) {
	int dx[] = { -1,-1,-1,0,0,1,1,1 };
	int dy[] = { -1,0,1,-1,0,1,-1,0,1 };
	std::vector<Point2i> result;
	for (int i = 0; i < 8; i++)
	{
		int new_x = pos.x + dx[i];
		int new_y = pos.y + dy[i];
		if (isInside(image, new_x, new_y))
		{
			result.push_back(Point2i(new_x, new_y));
		}
	}
	return result;
}

Mat_<int> traversal(Mat image, std::function<std::vector<Point2i> (Point2i,Mat)> getNeighbors)
{
	int label = 0;
	Mat_ <int> labels(image.rows,image.cols);
	//std::cout << image.rows << " "
	labels.setTo(0);
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			//std::cout << i << " " << j << "\n";
			if (image.at<uchar>(i, j) == 0 && labels[i][j] == 0)
			{
				label++;
				std::queue<Point2i> q;
				q.push((Point2i(i, j)));
				while (!q.empty()) {
					Point2i pixel = q.front();
					//std::cout << pixel;
					q.pop();
					for (auto neighbor : getNeighbors(pixel, image)) {
						if (image.at<uchar>(neighbor.x, neighbor.y) == 0 && labels[neighbor.x][neighbor.y] == 0) {
							q.push(neighbor);
							labels[neighbor.x][neighbor.y] = label;
						}
					}
				}
			}
		}
	}

	return labels;
}

Mat_<int> twoPassLabeling(Mat image, std::function<std::vector<Point2i>(Point2i, Mat)> getNeighbors) {
	int label = 0;
	Mat_ <int> labels(image.rows, image.cols);
	labels.setTo(0);
	std::vector<std::vector<int> > edges;

	edges.push_back(std::vector<int>());

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++)
		{
			if (image.at<uchar>(i, j) == 0 && labels[i][j] == 0)
			{
				std::vector<int> L;
				for (auto neighbor : getNeighbors(Point2i(i, j), image)) {
					if (labels[neighbor.x][neighbor.y] > 0) {
						L.push_back(labels[neighbor.x][neighbor.y]);
					}
				}

				if (L.size() == 0) {
					label++;
					labels[i][j] = label;
					edges.push_back(std::vector<int>());
				}
				else
				{
					int x = *std::min_element(L.begin(), L.end());
					labels[i][j] = x;
					for (auto it : L)
					{
						if (it != x) {
							edges[x].push_back(it);
							edges[it].push_back(x);
						}
					}
				}


			}
		}
	}

	int newLabel = 0;
	std::vector<int> newLabels;
	for (int i = 0; i < label + 1; i++) {
		newLabels.push_back(0);
	}

	for (int i = 1; i <= label; i++) {
		if (newLabels[i] == 0) {
			newLabel++;
			std::queue<int>  q;
			q.push(i);
			while (!q.empty()) {
				int x = q.front();
				q.pop();
				for (auto y : edges[x])
				{
					if (newLabels[y] == 0) {
						newLabels[y] = newLabel;
						q.push(y);
					}
				}
			}
		}
	}

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++) {
			labels[i][j] = newLabels[labels[i][j]];
		}
	}
	return labels;
}

Mat setColorsToLables( Mat_<int> labels)
{
	Mat image(labels.rows, labels.cols, CV_8UC3);
	int maximumLabel=0;
	for (int i = 0; i < labels.rows; i++)
	{
		for (int j = 0; j < labels.cols; j++) {
			maximumLabel = max(maximumLabel, labels[i][j]);
		}
	}
	std::vector<Vec3b> colors;
	colors.push_back(Vec3b(255, 255, 255));
	for(int i=1;i<=maximumLabel;i++)
		colors.push_back(Vec3b(rand()%255, rand() % 255,rand()%255));

	for (int i = 0; i < labels.rows; i++)
	{
		for (int j = 0; j < labels.cols; j++) {
			image.at<Vec3b>(i, j) = colors[labels[i][j]];
		}
	}
	return image;

}

void showTraversal()
{
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_GRAYSCALE);

		Mat dst = Mat(src.rows, src.cols, CV_8UC3);

		Mat_<int> labels  = traversal(src, getNeighbors8);
		dst = setColorsToLables(labels);
		imshow("input image", src);
		imshow("out image", dst);
		std::vector<int> hist;
		

		waitKey();
	}
}

void showTwoPassLabeling()
{
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_GRAYSCALE);

		Mat dst = Mat(src.rows, src.cols, CV_8UC3);

		Mat_<int> labels = twoPassLabeling(src, getNeighbors8);
		dst = setColorsToLables(labels);
		imshow("input image", src);
		imshow("out image", dst);


		waitKey();
	}
}

void borderTracing(Mat image,Point2i startPoint,Mat &dst,std::vector <int> &chain_code){

	Point2i directions[] = { Point2i(0,1),
							Point2i(-1,1),
							Point2i(-1,0),
							Point2i(-1,-1),
							Point2i(0,-1),
							Point2i(1,-1),
							Point2i(1,0),
							Point2i(1,1) };
	int dir = 7;
	std::cout << directions[0].x << " " << directions[0].y << "\n";

	Point2i currentPoint = startPoint;
	std::vector<Point2i> points;
	std::vector<int> path;
	//find p1
	Point2i nextPoint;
	if (dir % 2 == 0)
	{
		dir = (dir + 7) % 8;
	}
	else {
		dir = (dir + 6) % 8;
	}
	// search the next point in 3X3
	for (int i = 0; i < 8; i++)
	{
		nextPoint = currentPoint + directions[dir];
		if (image.at<uchar>(nextPoint.x, nextPoint.y) == 0)
			break;
		dir = (dir + 1) % 8;
	}
	points.push_back(startPoint);
	points.push_back(nextPoint);
	std::cout << points[0] << " " << points[1] << "\n";
	bool firstPass = true;
	while((points[0]!= points[points.size()-2] && points[1]!= points[points.size()-1]) || firstPass){
		currentPoint = points.back();
		//find the start_direction 
		if (dir % 2 == 0)
		{
			dir = (dir + 7) % 8;
		}
		else {
			dir = (dir + 6) % 8;
		}
		// search the next point in 3X3
		for (int i = 0; i < 8; i++)
		{
			nextPoint = currentPoint + directions[dir];
			if (image.at<uchar>(nextPoint.x, nextPoint.y) == 0)
				break;

			dir = (dir + 1) % 8;
		}

		std::cout << nextPoint << " " << dir << "\n";
		path.push_back(dir);
		points.push_back(nextPoint);
		firstPass = false;
	}

	std::cout << "start to draw";

	for (auto it : points)
	{
		dst.at<uchar>(it.x, it.y) = 0;
	}
	path.pop_back();
	chain_code = path;
}

void showBorderTracing() {
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_GRAYSCALE);

		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++) {
				dst.at<uchar>(i,j) = 255;
			}
		}
		bool found = false;
		std::vector<int> chain_code;
		for (int i = 0; i < src.rows && !found; i++)
		{
			for (int j = 0; j < src.cols && !found; j++) {
				if (src.at<uchar>(i, j) == 0) {
					std::cout << i << " " << j << "\n";
					borderTracing(src, Point2i(i, j), dst, chain_code);
					found = true;
				}
			}
		}
		//dst = setColorsToLables(labels);
		imshow("input image", src);
		imshow("out image", dst);
		std::cout << "size of codes:" << chain_code.size() << "\n";
		for (auto it : chain_code) {
			std::cout << it << " ";
		}
		std::vector<int> derivative;

		std::cout << "\nderivative \n";
		for (int i = 0; i < chain_code.size()-1; i++) {
			if (chain_code[i + 1] - chain_code[i] > 0)
				derivative.push_back(chain_code[i + 1] - chain_code[i]);
			else
				derivative.push_back(8 + (chain_code[i + 1] - chain_code[i]));
			std::cout << derivative.back() << " ";
		}
		waitKey();
	}
}

void showReconstruct() {
	Point2i directions[] = { Point2i(0,1),
						Point2i(-1,1),
						Point2i(-1,0),
						Point2i(-1,-1),
						Point2i(0,-1),
						Point2i(1,-1),
						Point2i(1,0),
						Point2i(1,1) };

	std::ifstream fin("Images/reconstruct.txt");
	int startx, starty;
	fin >> startx >> starty;
	Mat dst = Mat(1000, 1000, CV_8UC1);
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++) {
			dst.at<uchar>(i, j) = 255;
		}
	}
	int size;
	fin >> size;
	dst.at<uchar>(startx, starty) = 0;
	Point2i postion(startx, starty);
	for (int i = 0; i < size; i++)
	{
		int dir;
		fin >> dir;
		postion += directions[dir];
		dst.at<uchar>(postion.x, postion.y) = 0;

	}
	imshow("out image", dst);

	waitKey();

}


void dilationStudents(Mat src, Mat& dst)
{
	dst = Mat(src.rows, src.cols, CV_8UC1);

	int nh[][2] = {
		{0, 1},
		{0, -1},
		{1, 0},
		{-1, 0},
		{1, 1},
		{-1, -1},
		{1, -1},
		{-1, 1}
	};

	int h = src.rows;
	int w = src.cols;

	dst.setTo(255);
	for (int i = 1; i < h - 1; i++) {
		for (int j = 1; j < w - 1; j++) {
			if (src.at<uchar>(i, j) == 0) {
		

				for (int k = 0; k < 8; k++)
				{

					int newi = i + nh[k][0];
					int newj = j + nh[k][1];

					dst.at<uchar>(newi, newj) = 0;

				}
				dst.at<uchar>(i, j) = 0;
			}
		}
	}

}





void erosionStudents(Mat src, Mat& dst)
{
	dst = Mat(src.rows, src.cols, CV_8UC1);

	int nh[][2] = {
		{0, 1},
		{0, -1},
		{1, 0},
		{-1, 0},
		{1, 1},
		{-1, -1},
		{1, -1},
		{-1, 1}
	};

	int h = src.rows;
	int w = src.cols;

	dst.setTo(255);

	for (int i = 1; i < h - 1; i++) {
		for (int j = 1; j < w - 1; j++) {
			bool object = true;
			if (src.at<uchar>(i, j) == 0) {
				for (int k = 0; k < 8; k++)
				{
					int newi = i + nh[k][0];
					int newj = j + nh[k][1];

					if (src.at<uchar>(newi, newj) == 255) {
						object = false;
						break;
					}


				}
				dst.at<uchar>(i, j) = 0;
				if (!object)
					dst.at<uchar>(i, j) = 255;
			}
		}
	}
}

void openingStudents(Mat src, Mat& dst2)
{
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	dst2 = Mat(src.rows, src.cols, CV_8UC1);

	erosionStudents(src, dst);
	dilationStudents(dst, dst2);
}

void closingStudents(Mat src, Mat& dst) {
	Mat dst_aux = Mat(src.rows, src.cols, CV_8UC1);
	dst = Mat(src.rows, src.cols, CV_8UC1);

	dilationStudents(src, dst_aux);
	erosionStudents(dst_aux, dst);

}


void applyRepeated(void (*operation) (Mat src, Mat& dst), int times, const Mat& src, Mat& dst) {
	Mat curr_src = src.clone();
	Mat curr_dst = src.clone();

	for (int i = 0; i < times; i++) {
		operation(curr_src, curr_dst);
		curr_src = curr_dst;
	}

	dst = curr_dst;
}

void boundaryStudents(Mat src, Mat& dst) {
	Mat erodedImage = Mat(src.rows, src.cols, CV_8UC1);
	dst = Mat(src.rows, src.cols, CV_8UC1);

	erosionStudents(src, erodedImage);

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = abs(src.at<uchar>(i, j) - erodedImage.at<uchar>(i, j));
		}

	//Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	//dst = src.clone()
}

void complementaryImage(Mat src, Mat &dst)
{
	dst = Mat(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) == 0)
				dst.at<uchar>(i, j) = 255;
			else 
				dst.at<uchar>(i, j) = 0;

		}
}

bool compareMat(Mat src, Mat dst)
{
	//dst = Mat(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) != dst.at<uchar>(i, j))
				return false;
		}

	return true;
}

void regionFilling(Point2i start,Mat A, Mat& dst)
{

	imshow("initia21l", A);

	dst = Mat(A.rows, A.cols, CV_8UC1);
	
	Mat x = Mat(A.rows, A.cols, CV_8UC1);
	Mat last_x = Mat(A.rows, A.cols, CV_8UC1);
	for (int i = 0; i < A.rows; i++)
		for (int j = 0; j < A.cols; j++)
		{
			x.at<uchar>(i, j) = 255;
		}

	last_x = x.clone();
	x.at<uchar>(start.x, start.y) = 0;
	int k=0;

	while (!compareMat(last_x,x ))
	{
		
		k++;
		std::cout << " " << k << "\n";
		last_x = x.clone();

		Mat dilation;
		dilationStudents(last_x, dilation);
	
		//intersection;
		for (int i = 0; i < A.rows; i++)
			for (int j = 0; j < A.cols; j++)
			{
				if(dilation.at<uchar>(i, j) == 0 && A.at<uchar>(i, j) == 0)
					x.at<uchar>(i, j) = 0;
				else 
					x.at<uchar>(i, j) = 255;
			}


	}
	dst = x;
}



void testBoundaryExtractionStudents()
{
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst;
		boundaryStudents(src, dst);
		imshow("initial", src);
		imshow("boundary image", dst);
		waitKey(0);
	}
}

void testRegionFilling()
{
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst;
		Mat border;
		boundaryStudents(src, border);

		regionFilling(Point(110, 110), border, dst);
	
		imshow("initial", src);
		imshow("region filling image", dst);
		waitKey(0);
	}
}

float computeMean(Mat src) {

	float total_sum = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			total_sum += (float)src.at<uchar>(i, j);
		}
	}
	return( total_sum / (float)(src.rows * src.cols));
}

float computeStandard(Mat src) {
	float ans;
	float mean = computeMean(src);
	float total_sum = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			total_sum += ((float)src.at<uchar>(i, j)-mean)* ((float)src.at<uchar>(i, j) - mean);
		}
	}
	ans= total_sum /(float) (src.rows * src.cols);
	return sqrt(ans);

}

void histogramCumulative(Mat src)
{
	std::vector<int> histogram;
	histogram = computeHistogram(src);
	std::vector<int> histogramCumulative;
	histogramCumulative.push_back(histogram[0]);
	for (int i = 1; i < histogram.size(); i++)
	{
		int val = histogramCumulative.back() + histogram[i];
		histogramCumulative.push_back(val);
	}
	showHistogram("histogramCumulative", histogramCumulative.data(), histogramCumulative.size(), 200);
}
void lab8Problem1()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_COLOR);
		std::cout << "mean " << computeMean(src) << "\n";
		std::cout << "Standard " << computeStandard(src) << "\n";

		imshow("Source img", src);
		histogramCumulative(src);
		waitKey();
	}
}
void computeMeanBasedOnThreshold(Mat src, float T, float &mean1, float &mean2)
{
	float total_sum1 = 0, total_sum2 = 0;
	float n1 = 0, n2 = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if ((float)src.at<uchar>(i, j) <= T) {
				total_sum1 += (float)src.at<uchar>(i, j);
				n1++;
			}
			else
			{
				total_sum2 += (float)src.at<uchar>(i, j);
				n2++;
			}
		}
	}
	mean1=(total_sum1 / n1);
	mean2 = (total_sum2 / n2);
}
void globalThresholding(Mat src,float err,Mat &dst)
{
	int Imax = 0, Imin = 300;
	std::vector<int> histogram;
	histogram = computeHistogram(src);
	for (int i = 0; i < histogram.size(); i++)
	{
		if (histogram[i] != 0)
		{
			Imax = max(Imax, i);
			Imin = min(Imin, i);
		}
	}

	float threshold = 0;
	float threshold2 = ((float)(Imax + Imin)) / 2.0;
	float mean1, mean2;
	while (fabs(threshold2 - threshold) < err) {

		threshold = threshold2;
		computeMeanBasedOnThreshold(src, threshold2, mean1, mean2);
		threshold2 = (mean1 + mean2) / 2.0;
	}
	std::cout << threshold2 << "\n";
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			if ((float)src.at<uchar>(i, j) <= threshold2) {
				dst.at<uchar>(i, j) = 0;
			}else
			{
				dst.at<uchar>(i, j) = 255;
			}
		}
	}


}

void lab8Problem2()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		imshow("Source img", src);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		globalThresholding(src, 0.1, dst);
		imshow("dst", dst);

		waitKey();
	}
}
void histogramStretchingShrinking(Mat src, int gOutMin, int gOutMax) {
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);

	int gIn, gInMin = 256, gInMax = 0;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			uchar val = src.at<uchar>(i, j);
			if (val > gInMax) {
				gInMax = val;
			}
			else if (val < gInMin) {
				gInMin = val;
			}
		}
	}

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			uchar aux = gOutMin + ((src.at<uchar>(i, j) - gInMin) * (gOutMax - gOutMin) / (gInMax - gInMin));
			if (aux <= 0) {
				dst.at<uchar>(i, j) = 0;
			}
			else if (aux >= 255) {
				dst.at<uchar>(i, j) = 255;
			}
			else {
				dst.at<uchar>(i, j) = aux;
			}
		}
	}

	imshow("StretchingShrinking", dst);
	std::vector<int> histogram;
	histogram = computeHistogram(dst);
	showHistogram("histogramStretchingShrinking", histogram.data(), histogram.size(), 200);
}



void histogramGamma(Mat src, float gamma) {
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			float aux = 255 * pow(((float)src.at<uchar>(i, j) / 255.0), gamma);
			if (aux <= 0) {
				dst.at<uchar>(i, j) = 0;
			}
			else if (aux >= 255) {
				dst.at<uchar>(i, j) = 255;
			}
			else {
				dst.at<uchar>(i, j) = floor(aux);
			}
		}
	}

	imshow("Gamma", dst);
	std::vector<int> histogram;
	histogram = computeHistogram(dst);
	showHistogram("histogramGamma", histogram.data(), histogram.size(), 200);
}

void histogramSlide(Mat src, int offset) {
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			int val = src.at<uchar>(i, j) + offset;
			if (val <= 0) {
				dst.at<uchar>(i, j) = 0;
			}
			else if (val >= 255) {
				dst.at<uchar>(i, j) = 255;
			}
			else {
				dst.at<uchar>(i, j) = val;
			}
		}
	}

	imshow("Brightness", dst);
	std::vector<int> histogram;
	histogram = computeHistogram(dst);
	showHistogram("histogramSlide", histogram.data(), histogram.size(), 200);
}
void lab8Problem3() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		float gamma;
		int gOutMin, gOutMax, offset;
		std::cout << "gOutMin  ";
		std::cin >> gOutMin;
		std::cout << "gOutMax  ";
		std::cin >> gOutMax;
		std::cout << "gamma ";
		std::cin >> gamma;
		std::cout << "offset  ";
		std::cin >> offset;
		Mat dst = Mat(height, width, CV_8UC1);
		imshow("Source img", src);

		histogramStretchingShrinking(src, gOutMin, gOutMax);

		histogramGamma(src, gamma);

		histogramSlide(src, offset);
		std::vector<int> histogram;
		histogram = computeHistogram(src);
		showHistogram("src histogram", histogram.data(), histogram.size(), 200);
		waitKey();
	}
}


void histogramEqualization(Mat src,Mat &dst)
{
	std::vector<float>pdf = computePDF(src);
	std::vector<float> cpdf;
	cpdf.push_back(pdf[0]);
	for (int i = 1; i < pdf.size(); i++)
	{
		float val = (cpdf.back() + pdf[i]);
		cpdf.push_back(val);
	}
	std::vector<int> table;
	for (int i = 0; i < pdf.size(); i++)
	{
		int val = (int) (255.0 * cpdf[i]);
		table.push_back(val);
	}

	for (int i = 0; i < table.size(); i++) {
		std::cout << i << " " << table[i] << " " << cpdf[i] << " " << cpdf[i]*255.0 <<"\n";
	}
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<char>(i, j) = table[ src.at<uchar>(i, j)];
		}
	}
}

void lab8Problem4()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		imshow("Source img", src);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		std::vector<int> histogram;
		histogram = computeHistogram(src);
		showHistogram("src histogram", histogram.data(), histogram.size(), 200);

		histogramEqualization(src, dst);
		imshow("dst", dst);


		std::vector<int> histogram2;
		histogram2 = computeHistogram(dst);
		showHistogram("histogramEqualization", histogram2.data(), histogram2.size(), 200);
		waitKey();
	}
}
void applyFilter(float H[7][7], Mat src, Mat& dst, int sizeH = 7) {
	int start = (sizeH - 1) / 2;

	for (int i = start; i < src.rows - start; i++) {
		for (int j = start; j < src.cols - start; j++) {
			uchar pixel = src.at<uchar>(i, j);

			float sum = 0;
			float sPlus = 0;
			float sMinus = 0;
			float s = 0;
			for (int u = 0; u < sizeH; u++) {
				for (int v = 0; v < sizeH; v++) {
					sum = sum + (H[u][v] * src.at<uchar>(i + u - start, j + v - start));
					if (H[u][v] > 0) {
						sPlus = sPlus + H[u][v];
					}
					else {
						sMinus = sMinus - H[u][v];
					}

					if (sMinus > 0) {
						s = (float)(1 / (2 * max(sPlus, sMinus)));
					}
					else {
						s = sPlus;
					}
				}
			}
			float initSum = sum;
			sum = sum / s;
			if (sMinus == 0) {
				dst.at < uchar >(i,j)= bound((int)floor(sum));
			}
			else {
				dst.at<uchar>(i, j) = floor(s * initSum + 255 / 2);
			}

		}
	}

}


void lab9Problem1() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		float H[7][7] = {
			-1, -1, -1, 
			-1, 8, -1, 
			-1, -1, -1, 
		};
		applyFilter(H, src, dst,3);
		imshow("input image", src);
		imshow("out", dst);
		waitKey();
	}
}

void lab9Problem2()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		Mat dst2 = Mat(src.rows, src.cols, CV_8UC1);
		Mat dst3 = Mat(src.rows, src.cols, CV_8UC1);

		float H[7][7] = {
			1, 1, 1,
			1, 1,1,
			1, 1, 1,
		};
		float H2[7][7] = {
		0, 0, 0,
		-1, 3, -1,
		0, -1, 0,
		};
		float H3[7][7] = {
		-1, -1, -1,
		-1, 9, -1,
		-1, -1, -1,
		};
		applyFilter(H, src, dst);
		applyFilter(H2, src, dst2);
		applyFilter(H3, src, dst3);

		imshow("input image", src);
		imshow("dst", dst);
		imshow("dst2", dst2);
		imshow("dst3", dst3);

		waitKey();
	}
}

void medianMaxMinFilter(Mat src,Mat &dst,int w,int choice)
{
	int start = (w - 1) / 2;
	std::vector<uchar> allValues;

	std::cout << "start median filter\n";
	std::cout << src.rows << " " << src.cols << "\n";
	for (int i = start; i < src.rows - start; i++) {
		for (int j = start; j < src.cols - start; j++) {
			uchar pixel = src.at<uchar>(i, j);
			allValues.clear();
			
			for (int u = 0; u < w; u++) {
				for (int v = 0; v < w; v++) {
					allValues.push_back(src.at<uchar>(i + u - start, j + v - start));
				}
			}
			sort(allValues.begin(), allValues.end());
			if (choice == 1) {
				dst.at<uchar>(i, j) = allValues[allValues.size() / 2];
			}
			else if(choice == 2)
			{
				dst.at<uchar>(i, j) = allValues.front();
			}
			else
			{
				dst.at<uchar>(i, j) = allValues.back();
			}
			//std::cout << dst.at<uchar>(i, j) << " ";
		}
	}

	std::cout << "done with median filter";
}

float gaussianCell(float x, float y, float x0, float y0,float deviations)
{
	const float euler = exp(1);
	float expValue = (-(x - x0) * (x - x0) + (y - y0) * (y - y0))/(2* deviations* deviations);
	return pow(euler, expValue) / (2 * PI * deviations * deviations);
}

void lab10Problem1()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		int w;
		std::cout << "w= \n";
		std::cin >> w;
		int choice = 0;

		std::cout << "1 - median \n 2 - min \n 3- max\n";
		std::cin >> choice;
		double t = (double)getTickCount();
		medianMaxMinFilter( src, dst,w, choice);
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Display (in the console window) the processing time in [ms]
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("dst ", dst);



		waitKey();
	}
}
void lab10Problem2()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		float H[7][7];
		int w;
		std::cout << "w= ";
		std::cin >> w;

		double t = (double)getTickCount(); // Get the current time [ms]
	// … Actual processing …
	// Get the current time again and compute the time difference [ms]
	
		for (int i = 0; i < w; i++)
		{
			for (int j = 0; j < w; j++)
			{
				H[i][j] = gaussianCell(i, j, (int)(w / 2), (int)(w / 2), (float)w / 6.0);
				std::cout << H[i][j] << " ";
			}
			std::cout << "\n";
		}
		applyFilter(H,src, dst, w);
		imshow("input image", src);
		imshow("dst", dst);

		t = ((double)getTickCount() - t) / getTickFrequency();
		// Display (in the console window) the processing time in [ms]
		printf("Time = %.3f [ms]\n", t * 1000);
		waitKey();
	}
}

int main()
{
	int  op;
	functionSet.push_back(std::make_pair("testMouseClick", testMouseClick));
	functionSet.push_back(std::make_pair("testBGR2HSV", testBGR2HSV));
	functionSet.push_back(std::make_pair("lab1Problem3", lab1Problem3));
	functionSet.push_back(std::make_pair("lab1Problem4", lab1Problem4));
	functionSet.push_back(std::make_pair("lab1Problem5", lab1Problem5));
	functionSet.push_back(std::make_pair("lab1Problem6", lab1Problem6));
	functionSet.push_back(std::make_pair("lab2Problem1", lab2Problem1));
	functionSet.push_back(std::make_pair("lab2Problem2", lab2Problem2));
	functionSet.push_back(std::make_pair("lab2Problem3", lab2Problem3));
	functionSet.push_back(std::make_pair("lab2Problem4", lab2Problem4));
	functionSet.push_back(std::make_pair("testIsInside", testIsInside));
	functionSet.push_back(std::make_pair("showHistogram", displayHistogram));
	functionSet.push_back(std::make_pair("show m bin histogram", displayBinHistogram));
	functionSet.push_back(std::make_pair("showMultiLevelTreshold", showMultiLevelTreshold));
	functionSet.push_back(std::make_pair("showFloydSteinberg", showFloydSteinberg));
	functionSet.push_back(std::make_pair("showMultiLevelTresholdHSV", showMultiLevelTresholdHSV));
	functionSet.push_back(std::make_pair("computeFeaturesAllObjects", computeFeaturesAllObjects));
	functionSet.push_back(std::make_pair("computeClickFeaturesObject", computeClickFeaturesObject));
	functionSet.push_back(std::make_pair("selectObjects", selectObjects));
	functionSet.push_back(std::make_pair("showTraversal", showTraversal));
	functionSet.push_back(std::make_pair("showTwoPassLabeling", showTwoPassLabeling));
	functionSet.push_back(std::make_pair("showBorderTracing", showBorderTracing));
	functionSet.push_back(std::make_pair("showReconstruct", showReconstruct));
	functionSet.push_back(std::make_pair("testBoundaryExtractionStudents", testBoundaryExtractionStudents));
	functionSet.push_back(std::make_pair("testRegionFilling", testRegionFilling));
	functionSet.push_back(std::make_pair("lab8Problem1", lab8Problem1));
	functionSet.push_back(std::make_pair("lab8Problem2", lab8Problem2));
	functionSet.push_back(std::make_pair("lab8Problem3", lab8Problem3));
	functionSet.push_back(std::make_pair("lab8Problem4", lab8Problem4));
	functionSet.push_back(std::make_pair("lab9Problem1", lab9Problem1));
	functionSet.push_back(std::make_pair("lab9Problem2", lab9Problem2));
	functionSet.push_back(std::make_pair("lab10Problem1", lab10Problem1));
	functionSet.push_back(std::make_pair("lab10Problem2", lab10Problem2));

	
	do {
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		int operationValue = 1;
		for (auto& it : functionSet)
		{
			std::cout << operationValue << "-" << it.first << "\n";
			operationValue++;
		}
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		if (op <= functionSet.size() && op>=1)
		{
			functionSet[op-1].second();
		}

	} while (op != 0);

	return 0;

}
