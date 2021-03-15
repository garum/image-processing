// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <utility>
#include <functional>

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
		

		std::cout << "projection_h : " << projection_h(label_image, x) << "\n";
		std::cout << "projectoin_v : " << projection_v(label_image, y) << "\n";
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
