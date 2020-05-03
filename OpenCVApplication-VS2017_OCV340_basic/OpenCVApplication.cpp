// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <random>
#include <stack>
#include <fstream>


using namespace cv;

Mat dstGlobal;

typedef struct _boundingBox
{
	int xMin;
	int xMax;
	int yMin;
	int yMax;
}BoundingBox;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
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
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

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
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
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
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
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
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

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

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
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
	if (event == CV_EVENT_LBUTTONDOWN)
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
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(0, 255, 0)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void additiveGrey()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				if (val + 200 > 255)
				{
					dst.at<uchar>(i, j) = 255;
				}
				else
				{
					dst.at<uchar>(i, j) = val + 200;
				}
			}
		}

		imshow("input image", src);
		imshow("Additive Gray", dst);
		waitKey();
	}
}

void multiplicativeGrey()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				if (2 * val > 255)
				{
					dst.at<uchar>(i, j) = 255;
				}
				else
				{
					dst.at<uchar>(i, j) = 2 * val;
				}
			}
		}

		imshow("input image", src);
		imshow("Additive Gray", dst);
		waitKey();
	}
}

void color256()
{
	int height = 256;
	int width = 256;

	Mat dst = Mat(height, width, CV_8UC3);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Vec3b pixel = dst.at<Vec3b>(i, j);
			if (i >= 0 && i <= 128 && j >= 0 && j <= 128)
			{
				pixel[0] = 255;
				pixel[1] = 255;
				pixel[2] = 255;
			}
			if (i >= 0 && i <= 128 && j >= 129 && j <= 256)
			{
				pixel[0] = 0;
				pixel[1] = 0;
				pixel[2] = 255;
			}
			if (i >= 129 && i <= 256 && j >= 0 && j <= 128)
			{
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = 0;
			}
			if (i >= 129 && i <= 256 && j >= 129 && j <= 256)
			{
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = 255;
			}
			dst.at<Vec3b>(i, j) = pixel;
		}
	}
	imshow("Colored areas", dst);
	waitKey();
}

void matrixInverse() {
	float v[9] = { 0, 1, 2,
				   3, 4, 1,
				   1, 1, 2 };

	Mat M(3, 3, CV_32FC1, v);

	std::cout << M.inv() << std::endl;
	getchar();
	getchar();
}
// LAB2

void threeColors()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat dstR = Mat(height, width, CV_8UC1);
		Mat dstG = Mat(height, width, CV_8UC1);
		Mat dstB = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);

				dstR.at<uchar>(i, j) = pixel[2];
				dstG.at<uchar>(i, j) = pixel[1];
				dstB.at<uchar>(i, j) = pixel[0];
			}
		}

		imshow("input image", src);
		imshow("R", dstR);
		imshow("G", dstG);
		imshow("B", dstB);
		waitKey();
	}
}

Mat colorToGreyscale(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Vec3b pixelSrc = src.at<Vec3b>(i, j);

			uchar val = (pixelSrc[0] + pixelSrc[1] + pixelSrc[2]) / 3;


			dst.at<uchar>(i, j) = val;

		}
	}

	return dst;
}

Mat greyscaleToBlackWhite(Mat src, int threshold)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar valSrc = src.at<uchar>(i, j);

			if (valSrc < threshold)
			{
				dst.at<uchar>(i, j) = 0;
			}
			else
			{
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
	return dst;
}

void rgbtohsv()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);


		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);

				float r = (float)pixel[2] / 255;
				float g = (float)pixel[1] / 255;
				float b = (float)pixel[0] / 255;

				float M = max(r, g);
				M = max(M, b);

				float m = min(r, g);
				m = min(m, b);

				float C = M - m;

				// value
				float V = M;

				// saturation
				float S = 0;

				if (V != 0)
				{
					S = C / V;
				}

				// hue
				float H = 0;
				if (C != 0)
				{
					if (M == r)
						H = 60 * (g - b) / C;
					if (M == g)
						H = 120 + 60 * (b - r) / C;
					if (M == b)
						H = 240 + 60 * (r - g) / C;
				}
				else
					H = 0;

				if (H < 0)
					H += 360;

				float H_norm = H * 255 / 360;
				float S_norm = S * 255;
				float V_norm = V * 255;

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

bool isInside(Mat img, int i, int j)
{
	int rows = img.rows;
	int cols = img.cols;

	if (i >= 0 && i < rows && j >= 0 && j < cols)
		return true;

	return false;
}

void computeHistogram()
{
	char fname[MAX_PATH];
	int* histogram = (int*)calloc(256, sizeof(int));

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar valSrc = src.at<uchar>(i, j);
				histogram[(int)valSrc]++;

			}
		}

		for (int i = 0; i < 256; i++)
		{
			std::cout << i << ": " << histogram[i] << std::endl;
		}

		imshow("input image", src);
		waitKey();
	}
	free(histogram);
	histogram = NULL;
}

void computeFDP()
{
	char fname[MAX_PATH];
	float* FDP = (float*)calloc(256, sizeof(float));

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar valSrc = src.at<uchar>(i, j);
				FDP[(int)valSrc] += 1;
			}
		}

		for (int i = 0; i < 256; i++)
		{
			FDP[i] /= (height * width);
			std::cout << i << ": " << FDP[i] << std::endl;
		}

		imshow("input image", src);
		waitKey();
	}
	free(FDP);
	FDP = NULL;
}

void computeHistogramAndShowHistogram()
{
	char fname[MAX_PATH];
	int* histogram = (int*)calloc(256, sizeof(int));

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar valSrc = src.at<uchar>(i, j);
				histogram[(int)valSrc]++;
			}
		}

		showHistogram("Histogram", histogram, 255, 250);

		imshow("input image", src);
		waitKey();
	}
	free(histogram);
	histogram = NULL;
}

void computeReducedAccHistogram(int m)
{
	if (m >= 2 && 256 % m == 0)
	{
		char fname[MAX_PATH];
		int* histogram = (int*)calloc(m, sizeof(int));

		while (openFileDlg(fname))
		{
			Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
			int height = src.rows;
			int width = src.cols;

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					uchar valSrc = src.at<uchar>(i, j);
					int bucket = valSrc / m;
					histogram[bucket]++;
				}
			}

			showHistogram("Histogram", histogram, m, 255);

			imshow("input image", src);
			waitKey();
		}
		free(histogram);
		histogram = NULL;
	}
}

Mat computeMultiple(Mat src, float* FDP, int* histogram)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar valSrc = src.at<uchar>(i, j);
			FDP[(int)valSrc] += 1;
		}
	}

	// Calculation of the maxima
	std::vector<int> maxima;

	for (int i = 0; i < 256; i++)
	{
		FDP[i] /= (256 * 256);
	}

	int WH = 5;
	float TH = 0.0003;

	maxima.push_back(0);
	for (int k = WH; k <= 255 - WH; k++)
	{
		float avg = 0;
		bool isGreater = true;

		for (int l = k - WH; l <= k + WH; l++)
		{
			avg += FDP[l];
			if (FDP[k] < FDP[l])
				isGreater = false;
		}

		avg /= (2 * WH + 1);
		if (FDP[k] > (avg + TH) && isGreater)
		{
			maxima.push_back(k);
		}
	}
	maxima.push_back(255);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar valSrc = src.at<uchar>(i, j);

			int minDistance = INT_MAX;
			int value = 0;

			for (int k = 0; k < maxima.size(); k++)
			{
				int current = std::abs(valSrc - maxima.at(k));
				if (current < minDistance)
				{
					minDistance = current;
					value = maxima.at(k);
				}
			}
			dst.at<uchar>(i, j) = value;
			histogram[value]++;
		}
	}
	free(FDP);
	FDP = NULL;
	return dst;
}

void computeMultipleWithFloydSteinberg()
{
	char fname[MAX_PATH];
	float* FDP = (float*)calloc(256, sizeof(float));
	int* histogram = (int*)calloc(256, sizeof(int));

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		Mat dst1 = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar valSrc = src.at<uchar>(i, j);
				FDP[(int)valSrc] += 1;
			}
		}

		// Calculation of the maxima
		std::vector<int> maxima;

		for (int i = 0; i < 256; i++)
		{
			FDP[i] /= (256 * 256);
		}

		int WH = 5;
		float TH = 0.0003;

		maxima.push_back(0);
		for (int k = WH; k <= 255 - WH; k++)
		{
			float avg = 0;
			bool isGreater = true;

			for (int l = k - WH; l <= k + WH; l++)
			{
				avg += FDP[l];
				if (FDP[k] < FDP[l])
					isGreater = false;
			}

			avg /= (2 * WH + 1);
			if (FDP[k] > (avg + TH) && isGreater)
			{
				maxima.push_back(k);
			}
		}
		maxima.push_back(255);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar valSrc = src.at<uchar>(i, j);

				int minDistance = INT_MAX;
				int value = 0;

				for (int k = 0; k < maxima.size(); k++)
				{
					int current = std::abs(valSrc - maxima.at(k));
					if (current < minDistance)
					{
						minDistance = current;
						value = maxima.at(k);
					}
				}

				dst.at<uchar>(i, j) = value;

				uchar oldVal = src.at<uchar>(i, j);
				uchar newVal = value;
				float error = oldVal - newVal;

				dst1.at<uchar>(i, j) = newVal;

				dst1.at<uchar>(i, j + 1) = max(min(dst1.at<uchar>(i, j + 1) + 7 * error / 16, 255), 0);
				dst1.at<uchar>(i + 1, j - 1) = max(min(dst1.at<uchar>(i + 1, j - 1) + 3 * error / 16, 255), 0);
				dst1.at<uchar>(i + 1, j) = max(min(dst1.at<uchar>(i + 1, j) + 5 * error / 16, 255), 0);
				dst1.at<uchar>(i + 1, j + 1) = max(min(dst1.at<uchar>(i + 1, j + 1) + error / 16, 255), 0);

				histogram[value]++;
			}
		}

		imshow("output image Floyd", dst1);
		imshow("output image", dst);
		imshow("input image", src);
		waitKey();
	}
	free(FDP);
	FDP = NULL;
}

//auxiliary HSV to RGB function
void HSVtoRGB(float H, float S, float V, uchar* R, uchar* G, uchar* B) {

	float C = S * V;
	float X = C * (1 - abs(fmod(H / 60.0, 2) - 1));
	float m = V - C;
	float Rs, Gs, Bs;

	if (H >= 0 && H < 60)
	{
		Rs = C;
		Gs = X;
		Bs = 0;
	}
	else if (H >= 60 && H < 120)
	{
		Rs = X;
		Gs = C;
		Bs = 0;
	}
	else if (H >= 120 && H < 180)
	{
		Rs = 0;
		Gs = C;
		Bs = X;
	}
	else if (H >= 180 && H < 240)
	{
		Rs = 0;
		Gs = X;
		Bs = C;
	}
	else if (H >= 240 && H < 300)
	{
		Rs = X;
		Gs = 0;
		Bs = C;
	}
	else
	{
		Rs = C;
		Gs = 0;
		Bs = X;
	}

	*R = (uchar)((Rs + m) * 255);
	*G = (uchar)((Gs + m) * 255);
	*B = (uchar)((Bs + m) * 255);
}

void reduceHue()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;

		float** rawH = (float**)calloc(height, sizeof(float*));
		for (int i = 0; i < height; i++)
		{
			*(rawH + i) = (float*)calloc(width, sizeof(float));
		}

		float** rawS = (float**)calloc(height, sizeof(float*));
		for (int i = 0; i < height; i++)
		{
			*(rawS + i) = (float*)calloc(width, sizeof(float));
		}

		float** rawV = (float**)calloc(height, sizeof(float*));
		for (int i = 0; i < height; i++)
		{
			*(rawV + i) = (float*)calloc(width, sizeof(float));
		}


		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);

		Mat dst = Mat(height, width, CV_8UC3);


		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);

				float r = (float)pixel[2] / 255;
				float g = (float)pixel[1] / 255;
				float b = (float)pixel[0] / 255;

				float M = max(r, g);
				M = max(M, b);

				float m = min(r, g);
				m = min(m, b);

				float C = M - m;

				// value
				float V = M;

				// saturation
				float S = 0;

				if (V != 0)
				{
					S = C / V;
				}

				// hue
				float H = 0;
				if (C != 0)
				{
					if (M == r)
						H = 60 * (g - b) / C;
					if (M == g)
						H = 120 + 60 * (b - r) / C;
					if (M == b)
						H = 240 + 60 * (r - g) / C;
				}
				else
					H = 0;

				if (H < 0)
					H += 360;

				rawH[i][j] = H;
				rawS[i][j] = S;
				rawV[i][j] = V;

				float H_norm = H * 255 / 360;
				float S_norm = S * 255;
				float V_norm = V * 255;

				dstH.at<uchar>(i, j) = H_norm;
				dstS.at<uchar>(i, j) = S_norm;
				dstV.at<uchar>(i, j) = V_norm;
			}
		}

		int* hist_dir = (int*)calloc(256, sizeof(int));
		float* FDP = (float*)calloc(256, sizeof(float));
		dstH = computeMultiple(dstH, FDP, hist_dir);

		imshow("Before", src);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel;

				uchar R = 255;
				uchar G = 255;
				uchar B = 255;

				float H = ((float)dstH.at<uchar>(i, j) * 360) / 255;

				HSVtoRGB(H, rawS[i][j], rawV[i][j], &R, &G, &B);

				pixel[0] = B;
				pixel[1] = G;
				pixel[2] = R;

				dst.at<Vec3b>(i, j) = pixel;
			}
		}

		imshow("After", dst);
		waitKey();
	}
}

bool testColor(Vec3b a, Vec3b b)
{
	return (a[0] == b[0] &&
		a[1] == b[1] &&
		a[2] == b[2]);
}

void onMouse(int event, int x, int y, int flags, void* param)
{
	Mat* src = (Mat*)param;
	Mat source = (*src).clone();
	int height = source.rows;
	int width = source.cols;


	if (event == CV_EVENT_LBUTTONDOWN)
	{
		Vec3b white(255, 255, 255);
		Vec3b customColor(255, 0, 255);
		Mat dstContour = Mat(height, width, CV_8UC3);
		Mat hor = Mat(height, width, CV_8UC3);
		Mat ver = Mat(height, width, CV_8UC3);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				dstContour.at<Vec3b>(i, j) = white;
			}
		}

		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);

		// Getting the color of the object
		Vec3b pixel = source.at<Vec3b>(y, x);

		// Compute area
		float area = 0.0f;
		// Compute mass center
		int rC = 0, cC = 0;
		// Compute perimeter
		int perimeter = 0;
		// Compute aspect ratio
		float aspectRatio = 0;
		// Compute projections
		int* vertical = (int*)calloc(height, sizeof(int));
		int* horizontal = (int*)calloc(width, sizeof(int));


		int xMax = INT_MIN, xMin = INT_MAX, yMax = INT_MIN, yMin = INT_MAX;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (testColor(pixel, source.at<Vec3b>(i, j)))
				{
					// Area
					area += 1;

					// Mass center
					rC += i;
					cC += j;

					// Perimeter
					Vec3b currentPixel = source.at<Vec3b>(i, j);
					for (int k = i - 1; k <= i + 1; k++)
					{
						for (int l = j - 1; l <= j + 1; l++)
						{
							if (isInside(source, k, l) && !testColor(currentPixel, source.at<Vec3b>(k, l)))
							{
								dstContour.at<Vec3b>(i, j) = currentPixel;
								perimeter++;
							}
						}
					}

					// Aspect ratio
					if (testColor(pixel, currentPixel))
					{
						yMax = (i > yMax) ? i : yMax;
						yMin = (i < yMin) ? i : yMin;
						xMax = (j > xMax) ? j : xMax;
						xMin = (j < xMin) ? j : xMin;
					}

					// Projections
					if (testColor(currentPixel, pixel))
					{
						vertical[i]++;
						horizontal[j]++;
					}

				}
			}
		}
		rC /= area;
		cC /= area;

		std::cout << "Area: " << area << std::endl;
		std::cout << "Mass center: (" << cC << "," << rC << ")" << std::endl;

		// Compute line
		float num = 0, den = 0, t1Den = 0, t2Den = 0, phi = 0, slope = 0, b = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b currentPixel = source.at<Vec3b>(i, j);

				if (testColor(pixel, currentPixel))
				{
					num += (i - rC) * (j - cC);
					t1Den += (j - cC) * (j - cC);
					t2Den += (i - rC) * (i - rC);
				}
			}
		}
		num *= 2;
		den = t1Den - t2Den;
		phi = atan2(num, den) / 2;
		slope = tan(phi);

		// Point 1
		int x1, y1;
		// Point 2
		int x2, y2;

		b = -slope * cC + rC;

		x1 = cC;
		x2 = cC;

		Vec3b color(255, 0, 255);

		source.at<Vec3b>(rC, cC) = color;

		int lastX1 = x1, lastX2 = x2;
		int error = 5;
		do
		{
			lastX1 = x1;
			lastX2 = x2;

			y1 = (slope * x1 + b > 0 && slope * x1 + b < height) ? slope * x1 + b : y1;
			y2 = (slope * x2 + b > 0 && slope * x2 + b < height) ? slope * x2 + b : y2;

			if (x1 < width && y1 > error && y1 < height - error)
				x1++;

			if (x2 > 0 && y2 > error && y2 < height - error)
				x2--;

		} while ((x1 < width || x2 > 0) && (lastX1 != x1 || lastX2 != x2));

		std::cout << "Line: P1(" << x1 << "," << y1 << "); P2(" << x2 << "," << y2 << ")" << std::endl;
		std::cout << "Perimeter: " << perimeter << std::endl;

		Point P1(x1, y1);
		Point P2(x2, y2);
		Point C(cC, rC);

		Scalar s(0, 125, 255);

		// Draw center
		cv::circle(source, C, 3, s, 3);

		// Draw line
		cv::line(source, P1, P2, s);

		float thinness = 4 * PI * (area / (perimeter * perimeter));
		std::cout << "Thinness ratio: " << thinness << std::endl;

		aspectRatio = (float)(xMax - xMin + 1) / (float)(yMax - yMin + 1);
		std::wcout << "Aspect ratio: " << aspectRatio << std::endl;

		// Projections
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < vertical[i]; j++)
			{
				ver.at<Vec3b>(i, j) = white;
			}
		}

		for (int j = 0; j < width; j++)
		{
			for (int i = 0; i < horizontal[j]; i++)
			{
				hor.at<Vec3b>(i, j) = white;
			}
		}

		imshow("Axis of elongation", source);
		imshow("Contour", dstContour);
		imshow("Vertical projecton", ver);
		imshow("Horizontal projecton", hor);

		waitKey(0);

	}
}

int area(Mat* src, uchar objectColor) {
	int area = 0;
	for (int i = 0; i < src->rows; i++) {
		for (int j = 0; j < src->cols; j++) {
			if (src->at<uchar>(i, j) == objectColor) {
				area += 1;
			}
		}
	}
	//printf("[Area] for object with color [%d,%d,%d]: %d\n", objectColor[2], objectColor[1], objectColor[0], area);
	return area;
}


int Ni[4] = { -1, 0, 1, 0 };
int Nj[4] = { 0, -1, 0, 1 };

int Ni8[8] = { -1,-1,0,1,1,1,0,-1 };
int Nj8[8] = { 0,-1,-1,-1,0,1,1,1 };

bool isInside_4(int r, int c, Mat src) {
	for (int i = 0;i < 4;i++) {
		if (!(r + Ni[i] >= 0 && r + Ni[i] < src.rows && c + Nj[i] >= 0 && c + Nj[i] < src.cols)) {
			return false;
		}
	}
	return true;
}

int perimeter(Mat* src, uchar objectColor) {
	Mat contour = Mat(src->rows, src->cols, CV_8UC3);
	contour.setTo(Scalar(255, 255, 255));

	int perimeterValue = 0;

	for (int i = 0; i < src->rows; i++) {
		for (int j = 0; j < src->cols; j++) {
			if(isInside_4(i,j,*src) == true){
				if (src->at<uchar>(i, j) == objectColor) {
					if (src->at<uchar>(i - 1, j) != objectColor ||
						src->at<uchar>(i, j - 1) != objectColor ||
						src->at<uchar>(i + 1, j) != objectColor ||
						src->at<uchar>(i, j + 1) != objectColor
						/*src->at<Vec3b>(i + 1, j + 1) != objectColor ||
						src->at<Vec3b>(i + 1, j - 1) != objectColor ||
						src->at<Vec3b>(i - 1, j - 1) != objectColor ||
						src->at<Vec3b>(i - 1, j + 1) != objectColor*/) {

						/*
						contour.at<Vec3b>(i, j)[0] = 200.0;
						contour.at<Vec3b>(i, j)[1] = 0;
						contour.at<Vec3b>(i, j)[2] = 0;
						*/
						perimeterValue++;
					}
				}
			}
		}
	}

	//printf("[Perimeter] for object with color [%d,%d,%d]: %d\n", objectColor[2], objectColor[1], objectColor[0], perimeterValue);
	return perimeterValue;
}

float thinessRatio(Mat* src, uchar objectColor) {

	float P = perimeter(src, objectColor);
	float A = area(src, objectColor);

	return (4 * PI) * (A / (P * P));
}

float aspectRatio(float xMax, float xMin, float yMax, float yMin)
{
	return (xMax - xMin + 1) / (yMax - yMin + 1);
}

bool checkArea(Mat src, Vec3b color, int TH_area)
{
	int height = src.rows;
	int width = src.cols;
	int area = 0;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (testColor(color, src.at<Vec3b>(i, j)))
			{
				area++;
			}
		}
	}
	if (area < TH_area)
		return true;
	return false;
}

bool checkPhi(Mat src, Vec3b color, float phi_LOW, float phi_HIGH)
{
	int height = src.rows;
	int width = src.cols;
	int area = 0;
	int rC = 0, cC = 0;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (testColor(color, src.at<Vec3b>(i, j)))
			{
				// Area
				area += 1;

				// Mass center
				rC += i;
				cC += j;
			}
		}
	}
	rC /= area;
	cC /= area;

	// Compute line
	float num = 0, den = 0, t1Den = 0, t2Den = 0, phi = 0, slope = 0, b = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			num += (i - rC) * (j - cC);
			t1Den += (j - cC) * (j - cC);
			t2Den += (i - rC) * (i - rC);
		}
	}
	num *= 2;
	den = t1Den - t2Den;
	phi = atan2(num, den) / 2;

	if (phi > phi_LOW && phi < phi_HIGH)
		return true;
	return false;
}

void checkObjects(Mat src, int TH_area, float phi_LOW, float phi_HIGH)
{
	int height = src.rows;
	int width = src.cols;

	std::vector<Vec3b> colorVector;

	// Get distinct color objects
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Vec3b color = src.at<Vec3b>(i, j);
			bool exists = false;
			int k = 0;

			while (k < colorVector.size() && exists == false)
			{
				if (testColor(color, colorVector.at(k)))
				{
					exists = true;
				}
				k++;
			}

			if (exists == false)
			{
				colorVector.push_back(color);
			}
		}
	}

	// Checking objects
	std::vector<Vec3b> goodColorVector;

	for (Vec3b color : colorVector)
	{
		if (checkArea(src, color, TH_area) && checkPhi(src, color, phi_LOW, phi_HIGH))
		{
			goodColorVector.push_back(color);
		}
	}

	Mat dst = Mat(height, width, CV_8UC3);
	Vec3b white(255, 255, 255);
	Vec3b black(0, 0, 0);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst.at<Vec3b>(i, j) = white;
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Vec3b currentColor = src.at<Vec3b>(i, j);

			for (Vec3b color : goodColorVector)
			{
				if (testColor(color, currentColor))
				{
					dst.at<Vec3b>(i, j) = currentColor;
				}
			}
		}
	}


	imshow("Old image", src);
	imshow("New image", dst);
	waitKey(0);
}

void computeTraits(Mat src)
{
	//Create a window
	namedWindow("My Window", 1);

	//set the callback function for any mouse event
	setMouseCallback("My Window", onMouse, &src);

	//show the image
	imshow("My Window", src);

	// Wait until user press some key
	waitKey(0);
}

Mat labeledObjectsMatrix(Mat src, int vecType)
{
	int width = src.cols;
	int height = src.rows;

	int label = 0;
	Mat source = greyscaleToBlackWhite(src, 250);
	Mat labels = Mat::zeros(height, width, CV_8UC1);


	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (source.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0)
			{
				label += 10;
				std::queue<Point> Q;
				labels.at<uchar>(i, j) = label;
				Point p(j, i);
				Q.push(p);

				while (!Q.empty())
				{
					Point q = Q.front();
					Q.pop();

					if (vecType == 1)
					{
						int di[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };
						int dj[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };
						for (int k = 0; k < 8; k++)
						{
							if (isInside(source, q.y + di[k], q.x + dj[k]))
							{
								if (source.at<uchar>(q.y + di[k], q.x + dj[k]) == 0 && labels.at<uchar>(q.y + di[k], q.x + dj[k]) == 0)
								{
									labels.at<uchar>(q.y + di[k], q.x + dj[k]) = label;
									Point neighbor(q.x + dj[k], q.y + di[k]);
									Q.push(neighbor);
								}
							}
						}
					}
					else if (vecType == 2)
					{
						int di[4] = { -1, 0, 1, 0 };
						int dj[4] = { 0, -1, 0, 1 };
						for (int k = 0; k < 4; k++)
						{
							if (isInside(source, q.y + di[k], q.x + dj[k]))
							{
								if (source.at<uchar>(q.y + di[k], q.x + dj[k]) == 0 && labels.at<uchar>(q.y + di[k], q.x + dj[k]) == 0)
								{
									labels.at<uchar>(q.y + di[k], q.x + dj[k]) = label;
									Point neighbor(q.x + dj[k], q.y + di[k]);
									Q.push(neighbor);
								}
							}
						}
					}
				}
			}
		}
	}

	return labels;
}

Mat colorObjects(Mat src, int vecType)
{
	int height = src.rows;
	int width = src.cols;

	Scalar white(255, 255, 255);

	Mat dst = labeledObjectsMatrix(src, vecType);
	Mat colored(height, width, CV_8UC3, white);

	std::map<int, Vec3b> colors;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int val = (int)dst.at<uchar>(i, j);
			if (val != 0)
			{
				if (colors.count(val) == 0)
				{
					int v1 = rand() % 256;
					int v2 = rand() % 256;
					int v3 = rand() % 256;

					Vec3b pixel(v1, v2, v3);
					colors[val] = pixel;
					colored.at<Vec3b>(i, j)[0] = v1;
					colored.at<Vec3b>(i, j)[1] = v2;
					colored.at<Vec3b>(i, j)[2] = v3;
				}
				else
				{
					Vec3b pixel = colors.at(val);
					colored.at<Vec3b>(i, j)[0] = pixel[0];
					colored.at<Vec3b>(i, j)[1] = pixel[1];
					colored.at<Vec3b>(i, j)[2] = pixel[2];
				}
			}
		}
	}

	return colored;
}

Mat equivalenceClassesLabeling(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	int label = 0;
	Mat source = greyscaleToBlackWhite(src, 250);
	Mat labels = Mat::zeros(height, width, CV_8UC1);

	std::vector<std::vector<uchar>> edges(256);

	int di[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };
	int dj[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (source.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0)
			{
				std::vector<uchar> L;
				for (int k = 0; k < 8; k++)
				{
					if (isInside(source, i + di[k], j + dj[k]))
					{
						if (labels.at<uchar>(i + di[k], j + dj[k]) > 0)
						{
							L.push_back(labels.at<uchar>(i + di[k], j + dj[k]));
						}
					}
				}

				if (L.size() == 0)
				{
					label++;
					labels.at<uchar>(i, j) = label;
				}
				else
				{
					uchar x = *std::min_element(L.begin(), L.end());
					labels.at<uchar>(i, j) = x;
					for (uchar y : L)
					{
						if (y != x)
						{
							edges[x].push_back(y);
							edges[y].push_back(x);
						}
					}
				}

			}
		}
	}

	int newLabel = 0;
	std::vector<int> newLabels(label + 1, 0);

	imshow("After first pass", labels);

	for (int i = 1; i < label + 1; i++)
	{
		if (newLabels[i] == 0)
		{
			newLabel++;
			std::queue<int> Q;

			std::cout << newLabel << std::endl;

			newLabels[i] = newLabel;
			Q.push(i);
			while (!Q.empty())
			{
				int x = Q.front();
				Q.pop();
				for (int y : edges[x])
				{
					if (newLabels[y] == 0)
					{
						newLabels[y] = newLabel;
						Q.push(y);
					}
				}
			}
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			labels.at<uchar>(i, j) = newLabels[labels.at<uchar>(i, j)];
		}
	}

	return labels;
}

Mat labeledObjectsMatrixPaused(Mat src)
{
	int width = src.cols;
	int height = src.rows;

	// init la 20 pentru vizualizare
	int label = 20;
	Mat source = greyscaleToBlackWhite(src, 250);
	Mat labels = Mat::zeros(height, width, CV_8UC1);

	int di[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };
	int dj[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0)
			{
				label++;
				std::queue<Point> Q;
				labels.at<uchar>(i, j) = label;
				Point p(j, i);
				Q.push(p);

				while (!Q.empty())
				{
					Point q = Q.front();
					Q.pop();
					for (int k = 0; k < 8; k++)
					{
						if (isInside(src, q.y + di[k], q.x + dj[k]))
						{
							if (src.at<uchar>(q.y + di[k], q.x + dj[k]) == 0 && labels.at<uchar>(q.y + di[k], q.x + dj[k]) == 0)
							{

								imshow("Lableled image", labels);
								waitKey(1);

								labels.at<uchar>(q.y + di[k], q.x + dj[k]) = label;
								Point neighbor(q.x + dj[k], q.y + di[k]);
								Q.push(neighbor);
							}
						}
					}
				}
			}
		}
	}

	return labels;
}

Mat stackLabelling(Mat src, int vecType)
{
	int width = src.cols;
	int height = src.rows;

	int label = 0;
	Mat source = src;
	Mat labels = Mat::zeros(height, width, CV_8UC1);


	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (source.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0)
			{
				label += 10;
				std::stack<Point> Q;
				labels.at<uchar>(i, j) = label;
				Point p(j, i);
				Q.push(p);

				while (!Q.empty())
				{
					Point q = Q.top();
					Q.pop();

					if (vecType == 1)
					{
						int di[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };
						int dj[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };
						for (int k = 0; k < 8; k++)
						{
							if (isInside(source, q.y + di[k], q.x + dj[k]))
							{
								if (source.at<uchar>(q.y + di[k], q.x + dj[k]) == 0 && labels.at<uchar>(q.y + di[k], q.x + dj[k]) == 0)
								{
									labels.at<uchar>(q.y + di[k], q.x + dj[k]) = label;
									Point neighbor(q.x + dj[k], q.y + di[k]);
									Q.push(neighbor);
								}
							}
						}
					}
					else if (vecType == 2)
					{
						int di[4] = { -1, 0, 1, 0 };
						int dj[4] = { 0, -1, 0, 1 };
						for (int k = 0; k < 4; k++)
						{
							if (isInside(source, q.y + di[k], q.x + dj[k]))
							{
								if (source.at<uchar>(q.y + di[k], q.x + dj[k]) == 0 && labels.at<uchar>(q.y + di[k], q.x + dj[k]) == 0)
								{
									labels.at<uchar>(q.y + di[k], q.x + dj[k]) = label;
									Point neighbor(q.x + dj[k], q.y + di[k]);
									Q.push(neighbor);
								}
							}
						}
					}
				}
			}
		}
	}

	return labels;
}

Mat contourTracing(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat::zeros(height, width, CV_8UC1);

	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	bool check = false;
	Point firstPoint;
	Point secondPoint;
	Point lastButOnePoint;
	Point currentPoint;

	int dir = 7;

	for (int i = 0; i < height && !check; i++)
	{
		for (int j = 0; j < width && !check; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{
				dst.at<uchar>(i, j) = 255;
				currentPoint.x = firstPoint.x = j;
				currentPoint.y = firstPoint.y = i;
				check = true;
			}
		}
	}

	bool second = false;
	// Assures that 2 points have been checked (for the stopping condition to be validly checked)
	int iteration = 0;
	bool stopCondition = false;

	do
	{
		int n;
		bool moved = false;

		if (dir % 2 == 0)
		{
			n = (dir + 7) % 8;
		}
		else
		{
			n = (dir + 6) % 8;
		}

		for (int k = 0; k < 8 && !moved; k++)
		{
			int tempDir = (n + k) % 8;
			if (src.at<uchar>(currentPoint.y + di[tempDir], currentPoint.x + dj[tempDir]) == src.at<uchar>(currentPoint.y, currentPoint.x))
			{
				dir = tempDir;
				moved = true;
				lastButOnePoint.x = currentPoint.x;
				lastButOnePoint.y = currentPoint.y;
				currentPoint.x = currentPoint.x + dj[tempDir];
				currentPoint.y = currentPoint.y + di[tempDir];

				if (!second)
				{
					second = true;
					secondPoint.x = currentPoint.x;
					secondPoint.y = currentPoint.y;
				}

				dst.at<uchar>(currentPoint.y, currentPoint.x) = 255;
			}
		}

		iteration++;
		if (iteration >= 2 && ((currentPoint.x == secondPoint.x) && (currentPoint.y == secondPoint.y)) && ((lastButOnePoint.x == firstPoint.x) && (lastButOnePoint.y == firstPoint.y)))
		{
			stopCondition = true;
		}

	} while (!stopCondition);

	return dst;
}

void contourTracing(Mat src, std::vector<int>* chainCodes, std::vector<int>* diff)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat::zeros(height, width, CV_8UC1);

	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	bool check = false;
	Point firstPoint;
	Point secondPoint;
	Point lastButOnePoint;
	Point currentPoint;

	int dir = 7;

	for (int i = 0; i < height && !check; i++)
	{
		for (int j = 0; j < width && !check; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{
				dst.at<uchar>(i, j) = 255;
				currentPoint.x = firstPoint.x = j;
				currentPoint.y = firstPoint.y = i;
				check = true;
			}
		}
	}

	bool second = false;
	// Assures that 2 points have been checked (for the stopping condition to be validly checked)
	int iteration = 0;
	bool stopCondition = false;

	do
	{
		bool moved = false;
		int n;

		if (dir % 2 == 0)
		{
			n = (dir + 7) % 8;
		}
		else
		{
			n = (dir + 6) % 8;
		}

		for (int k = 0; k < 8 && !moved; k++)
		{
			int tempDir = (n + k) % 8;
			if (src.at<uchar>(currentPoint.y + di[tempDir], currentPoint.x + dj[tempDir]) == src.at<uchar>(currentPoint.y, currentPoint.x))
			{
				dir = tempDir;
				moved = true;
				lastButOnePoint.x = currentPoint.x;
				lastButOnePoint.y = currentPoint.y;
				currentPoint.x = currentPoint.x + dj[tempDir];
				currentPoint.y = currentPoint.y + di[tempDir];

				if (!second)
				{
					second = true;
					secondPoint.x = currentPoint.x;
					secondPoint.y = currentPoint.y;
				}
				(*chainCodes).push_back(tempDir);
			}
		}

		iteration++;
		if (iteration >= 2 && ((currentPoint.x == secondPoint.x) && (currentPoint.y == secondPoint.y)) && ((lastButOnePoint.x == firstPoint.x) && (lastButOnePoint.y == firstPoint.y)))
		{
			stopCondition = true;
		}

	} while (!stopCondition);

	int size = (*chainCodes).size();
	for (int i = 0; i < size; i++)
	{
		int j = (i + 1) % size;
		int d = (*chainCodes).at(j) - (*chainCodes).at(i);

		if (d < 0)
			d += 8;

		(*diff).push_back(d % 8);
	}

}

Mat contourTracing(int startX, int startY, std::vector<int> directions, Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC3);
	Vec3b color1(0, 0, 0);
	Vec3b color2(0, 255, 0);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst.at<Vec3b>(i, j) = color1;
		}
	}


	dst.at<Vec3b>(startY, startX) = color2;

	int size = directions.size();
	int posX = startX, posY = startY;


	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	int counter = 0;

	for (int dir = 0; dir < size; dir++)
	{
		posX += dj[directions[dir]];
		posY += di[directions[dir]];
		counter++;
		dst.at<Vec3b>(posY, posX) = color2;
	}

	return dst;
}

Mat dilate(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1, 255);

	int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{
				dst.at<uchar>(i, j) = 0;
				for (int k = 0; k < 8; k++)
				{
					int ni = i + di[k];
					int nj = j + dj[k];
					if (isInside(src, ni, nj))
					{
						dst.at<uchar>(ni, nj) = 0;
					}
				}
			}
		}
	}

	return dst;
}

Mat erode(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1, 255);

	int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			bool cond = true;

			for (int k = 0; k < 8; k++)
			{
				int ni = i + di[k];
				int nj = j + dj[k];
				if (cond && isInside(src, ni, nj) && ((src.at<uchar>(i, j) == 255) || (src.at<uchar>(ni, nj) == 255)))
				{
					cond = false;
				}
			}

			if (cond)
			{
				dst.at<uchar>(i, j) = 0;
			}
			else
			{
				dst.at<uchar>(i, j) = 255;
			}
		}
	}

	return dst;
}

Mat open(Mat src)
{
	Mat temp = erode(src);
	return dilate(temp);
}

Mat close(Mat src)
{
	Mat temp = dilate(src);
	return erode(temp);
}


Mat morphOperation(Mat src, int opcode, int numberOfTimes)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1, 255);
	Mat prevDst = src;

	for (int i = 0; i < numberOfTimes; i++)
	{
		switch (opcode)
		{
		case 0: dst = dilate(prevDst); break;
		case 1: dst = erode(prevDst); break;
		case 2: dst = open(prevDst); break;
		case 3: dst = close(prevDst); break;
		default: dst = src; break;
		}
		prevDst = dst;

	}
	return dst;
}

Mat difference(Mat a, Mat b)
{
	int height = a.rows;
	int width = a.cols;

	Mat dst = Mat(height, width, CV_8UC1, 255);
	if (height == b.rows && width == b.cols)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (a.at<uchar>(i, j) != b.at<uchar>(i, j))
				{
					dst.at<uchar>(i, j) = a.at<uchar>(i, j);
				}
			}
		}
	}

	return dst;
}

Mat contourExtraction(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	return difference(src, erode(src));
}

Mat complement(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1, 255);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0)
				dst.at<uchar>(i, j) = 255;
			else
				dst.at<uchar>(i, j) = 0;
		}
	}
	return dst;
}

Mat intersection(Mat a, Mat b)
{
	int height = a.rows;
	int width = a.cols;

	Mat dst = Mat(height, width, CV_8UC1, 255);
	if (height == b.rows && width == b.cols)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (a.at<uchar>(i, j) == b.at<uchar>(i, j))
				{
					dst.at<uchar>(i, j) = a.at<uchar>(i, j);
				}
			}
		}
	}

	return dst;
}

Mat reunion(Mat a, Mat b)
{
	int height = a.rows;
	int width = a.cols;

	Mat dst = Mat(height, width, CV_8UC1, 255);
	if (height == b.rows && width == b.cols)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (a.at<uchar>(i, j) == 0 || b.at<uchar>(i, j) == 0)
				{
					dst.at<uchar>(i, j) = 0;
				}
			}
		}
	}

	return dst;
}

bool equalMatrices(Mat a, Mat b)
{
	int height = a.rows;
	int width = a.cols;

	if (height == b.rows && width == b.cols)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (a.at<uchar>(i, j) != b.at<uchar>(i, j))
				{
					return false;
				}
			}
		}
	}
	return true;
}

void regionFilling(int event, int x, int y, int flags, void* param)
{
	Mat* mats = (Mat*)param;
	Mat src = mats[0].clone();

	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1, 255);

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		Mat contour = contourExtraction(src);
		Mat contourComplement = complement(contour);
		Mat equalityTest;

		int pi = y, pj = x;
		std::cout << x << " " << y << "\n";

		int dm[] = { 0, -1, 0, 1 };
		int dn[] = { 1, 0, -1, 0 };

		if (src.at<uchar>(pi, pj) != 0)
		{
			dst.at<uchar>(pi, pj) = 0;
			Mat prevDst = Mat(height, width, CV_8UC1, 255);

			while (!equalMatrices(prevDst, dst))
			{
				prevDst = dst.clone();
				Mat dilatedDst = dilate(dst);
				dst = intersection(dilatedDst, contourComplement);
			}
			mats[1] = reunion(dst, contour);
			mats[1] = reunion(src, mats[1]);
			imshow("Result image", mats[1]);
			waitKey(0);
		}
	}
}

//EX 8.1
int* computeHistogram(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	int* histogram = (int*)calloc(256, sizeof(int));

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			histogram[(int)src.at<uchar>(i, j)]++;
		}
	}
	return histogram;
}

int* computeCumulativeHistogram(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	int* histogram = computeHistogram(src);
	int* cumulativeHistogram = (int*)calloc(256, sizeof(int));

	int sum = 0;
	for (int i = 0; i < 256; i++)
	{
		sum += histogram[i];
		cumulativeHistogram[i] = sum;
	}
	return cumulativeHistogram;
}

float averageIntensity(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	float average = 0;

	int* histogram = computeHistogram(src);

	for (int g = 0; g < 256; g++)
	{
		average += g * histogram[g];
	}

	average /= (float)height * (float)width;

	free(histogram);
	histogram = NULL;

	return average;
}

float standardDeviation(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	float stdDev = 0;
	float avg = averageIntensity(src);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar valSrc = src.at<uchar>(i, j);
			stdDev += (valSrc - avg) * (valSrc - avg);
		}
	}
	stdDev /= height * width;
	stdDev = sqrt((double)stdDev);
	return stdDev;
}

//EX 8.2
Mat automaticGlobalBinarization(Mat src)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	int* histogram = computeHistogram(src);

	int n = 255;

	int Imax = 0;
	int Imin = n;

	// Finding min
	int i = 0;
	while (i < n && histogram[i] == 0)
	{
		i++;
	}
	Imin = i;

	// Finding max
	i = n;
	while (i >= 0 && histogram[i] == 0)
	{
		i--;
	}
	Imax = i;

	int T = (Imin + Imax) / 2;
	int Tnew = (Imin + Imax) / 2;

	do
	{
		float N1 = 0;
		float s1 = 0;

		T = Tnew;

		for (int g = Imin; g <= T; g++)
		{
			N1 += histogram[g];
			s1 += g * histogram[g];
		}

		float N2 = 0;
		float s2 = 0;
		for (int g = T + 1; g <= Imax; g++)
		{
			N2 += histogram[g];
			s2 += g * histogram[g];
		}

		float uG1 = s1 / N1;
		float uG2 = s2 / N2;

		Tnew = (uG1 + uG2) / 2;

	} while (abs(Tnew - T) >= 0.1);

	std::cout << "T:" << Tnew << std::endl;
	free(histogram);
	histogram = NULL;

	return greyscaleToBlackWhite(src, Tnew);
}

//EX 8.3
Mat negative(Mat src)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst.at<uchar>(i, j) = 255 - src.at<uchar>(i, j);
		}
	}

	return dst;
}

Mat contrastAdjustment(Mat src, float gminout, float gmaxout)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	int* histogram = computeHistogram(src);
	int n = 256;

	float gminin = 0;
	float gmaxin = 0;

	// Finding min
	int i = 0;
	while (i < n && histogram[i] == 0)
	{
		i++;
	}
	gminin = i;

	// Finding max
	i = n;
	while (i >= 0 && histogram[i] == 0)
	{
		i--;
	}
	gmaxin = i;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int val = gminout + ((float)src.at<uchar>(i, j) - gminin) * ((gmaxout - gminout) / (gmaxin - gminin));
			if (val >= 0 && val <= 255)
				dst.at<uchar>(i, j) = val;
			else if (val > 255)
				dst.at<uchar>(i, j) = 255;
			else
				dst.at<uchar>(i, j) = 0;
		}
	}

	free(histogram);
	histogram = NULL;

	return dst;
}

Mat gammaCorrection(Mat src, double gamma)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			double gin = src.at<uchar>(i, j);
			gin /= 255.0;
			double val = 255.0 * pow(gin, gamma);
			if (val >= 0 && val <= 255)
				dst.at<uchar>(i, j) = (uchar)val;
			else if (val > 255)
				dst.at<uchar>(i, j) = 255;
			else
				dst.at<uchar>(i, j) = 0;
		}
	}
	return dst;
}

Mat brightnessAdjustment(Mat src, int offset)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int val = src.at<uchar>(i, j) + offset;
			if (val >= 0 && val <= 255)
				dst.at<uchar>(i, j) = val;
			else if (val > 255)
				dst.at<uchar>(i, j) = 255;
			else
				dst.at<uchar>(i, j) = 0;
		}
	}
	return dst;
}

//EX 8.4
Mat histogramEqualization(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1);
	int* histogram = computeHistogram(src);
	int* cumulativeHistogram = computeCumulativeHistogram(src);

	std::map<int, float> tab;
	for (int i = 0; i < 256; i++)
	{
		tab[i] = (cumulativeHistogram[i] * 255.0) / (height * width);
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int val = src.at<uchar>(i, j);
			dst.at<uchar>(i, j) = tab[val];
		}
	}

	free(histogram);
	free(cumulativeHistogram);
	histogram = NULL;
	cumulativeHistogram = NULL;

	return dst;
}

//EX 9.1 & 9.2
Mat filter(Mat src, Mat h, int type)
{
	int height = src.rows;
	int width = src.cols;
	int hHeight = h.rows;
	int hWidth = h.cols;

	Mat dst = Mat(height, width, CV_8UC1);

	if (hHeight == hWidth)
	{
		int w = hWidth;
		int k = (w - 1) / 2;
		int sp = 0;
		int sm = 0;
		int c = 0;

		switch (type)
		{
		case 0:
			for (int u = 0; u < w; u++)
			{
				for (int v = 0; v < w; v++)
				{
					c += h.at<int>(u, v);
				}
			}

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					int val = 0;
					for (int u = 0; u < w; u++)
					{
						for (int v = 0; v < w; v++)
						{
							val += h.at<int>(u, v) * (int)src.at<uchar>(i + u - k, j + v - k);
						}
					}
					val /= c;
					dst.at<uchar>(i, j) = (uchar)val;
				}
			}
			break;
		case 1:
			for (int u = 0; u < w; u++)
			{
				for (int v = 0; v < w; v++)
				{
					if (h.at<int>(u, v) >= 0)
						sp += h.at<int>(u, v);
					else
						sm -= h.at<int>(u, v);
				}
			}

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					int val = 0;
					for (int u = 0; u < w; u++)
					{
						for (int v = 0; v < w; v++)
						{
							val += h.at<int>(u, v) * (int)src.at<uchar>(i + u - k, j + v - k);
						}
					}
					val /= 2 * max(sp, sm);
					val += 255 / 2;
					dst.at<uchar>(i, j) = (uchar)val;
				}
			}
			break;
		default: break;
		}
	}
	return dst;
}

//EX 9.3
void centeringTransform(Mat img)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat genericFrequencyDomainFilter(Mat src)
{
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	centeringTransform(srcf);

	Mat fourier;
	cv::dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	cv::split(fourier, channels);

	Mat mag, phi;
	cv::magnitude(channels[0], channels[1], mag);
	cv::phase(channels[0], channels[1], phi);

	//inserare operatii de filtrare aplicate pe coef. Fourier

	//memorarea partii reale in channels[0] si img. in channels[1]

	Mat dst, dstf;
	cv::merge(channels, 2, fourier);
	cv::dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	centeringTransform(dstf);

	cv::normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

	return dst;
}

//EX 9.4
Mat logarithm(Mat src)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat::zeros(height, width, CV_32FC1);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			float val = src.at<float>(i, j) + 1;
			dst.at<float>(i, j) += std::log(val);

		}

	return dst;
}


Mat computeFTMagnitude(Mat src, bool c)
{
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	if (c)
		centeringTransform(srcf);

	Mat fourier;
	cv::dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	cv::split(fourier, channels);

	Mat mag, phi;
	cv::magnitude(channels[0], channels[1], mag);
	cv::phase(channels[0], channels[1], phi);

	Mat logMag = logarithm(mag);
	Mat res;
	Mat dst;

	logMag.convertTo(res, CV_8UC1);
	cv::normalize(res, dst, 0, 255, NORM_MINMAX, CV_8UC1);

	return dst;
}

//EX 9.5
Mat idealFilter(Mat src, int type, float val)
{
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	centeringTransform(srcf);

	Mat fourier;
	cv::dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	cv::split(fourier, channels);



	int height = channels[0].rows;
	int width = channels[0].cols;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float expr = 0;
			float power = 0;
			switch (type)
			{
			case 0:
				expr = (height / 2 - i) * (height / 2 - i) +
					(width / 2 - j) * (width / 2 - j);
				if (expr > (val * val))
				{
					channels[0].at<float>(i, j) = 0;
					channels[1].at<float>(i, j) = 0;
				}
				break;
			case 1:
				expr = (height / 2 - i) * (height / 2 - i) +
					(width / 2 - j) * (width / 2 - j);
				if (expr <= (val * val))
				{
					channels[0].at<float>(i, j) = 0;
					channels[1].at<float>(i, j) = 0;
				}
				break;
			case 2:
				power = (-1) * ((height / 2 - i) * (height / 2 - i) +
					(width / 2 - j) * (width / 2 - j));
				power /= (val * val);
				expr = exp(power);
				channels[0].at<float>(i, j) = channels[0].at<float>(i, j) * expr;
				channels[1].at<float>(i, j) = channels[1].at<float>(i, j) * expr;
				break;
			default:
				power = (-1) * ((height / 2 - i) * (height / 2 - i) +
					(width / 2 - j) * (width / 2 - j));
				power /= (val * val);
				expr = exp(power);
				channels[0].at<float>(i, j) = channels[0].at<float>(i, j) * (1 - expr);
				channels[1].at<float>(i, j) = channels[1].at<float>(i, j) * (1 - expr);
				break;
			}
		}
	}


	Mat dst, dstf;
	cv::merge(channels, 2, fourier);
	cv::dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	centeringTransform(dstf);

	cv::normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

	return dst;
}

Mat filterRoundObjects(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat::zeros(height, width, CV_8UC1);
	std::map<uchar, bool> colorMap;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar currentColor = src.at<uchar>(i, j);
			if (currentColor != 0)
			{
				if (colorMap.count(currentColor) == 0)
				{
					//not exists
					float tr = thinessRatio(&src, currentColor);
					float a = area(&src, currentColor);
					std::cout << tr << " " << currentColor << std::endl;
					if (tr > 0.7 && a > 400)
					{
						colorMap[currentColor] = false;
						dst.at<uchar>(i, j) = currentColor;
					}
					else
					{
						colorMap[currentColor] = true;
						dst.at<uchar>(i, j) = 0;
					}
				}
				else
				{
					//exists
					if (colorMap[currentColor] == true)
						dst.at<uchar>(i, j) = 0;
					else
						dst.at<uchar>(i, j) = currentColor;
				}
			}
		}
	}
	return dst;
}


////
int di_4[4] = { -1,0,1,0 };
int dj_4[4] = { 0,-1,0,1 };

int di[4] = { -1,-1,-1,0 };
int dj[4] = { -1,0,1,-1 };

int di_8[8] = { -1,-1,0,1,1,1,0,-1 };
int dj_8[8] = { 0,-1,-1,-1,0,1,1,1 };

Mat displayRandomColours(int label, Mat labeledMatrix) {
	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);

	Vec3b* hashmap = (Vec3b*)calloc(label + 1, sizeof(Vec3b));

	for (int i = 0; i <= label; i++) {
		hashmap[i] = Vec3b(d(gen), d(gen), d(gen));
	}

	Mat coloredMatrix = Mat(labeledMatrix.rows, labeledMatrix.cols, CV_8UC3);
	coloredMatrix.setTo(Scalar(255, 255, 255));

	for (int i = 0; i < labeledMatrix.rows; i++) {
		for (int j = 0; j < labeledMatrix.cols; j++) {
			if (labeledMatrix.at<uchar>(i, j) != 0) {
				coloredMatrix.at<Vec3b>(i, j) = hashmap[labeledMatrix.at<uchar>(i, j)];
			}
		}
	}

	return coloredMatrix;
}

Mat bfsLabeling(Mat* src, int neighborhoodType) {
	//Should check image type, might get an image that is not binary/grayscale
	int label = 0;
	Mat labeledMatrix = Mat::zeros(src->rows, src->cols, CV_8UC1);

	for (int i = 0; i < src->rows; i++) {
		for (int j = 0; j < src->cols; j++) {
			if (src->at<uchar>(i, j) == 0 && labeledMatrix.at<uchar>(i, j) == 0) {
				label++;
				std::queue<Point> Q;
				labeledMatrix.at<uchar>(i, j) = label;
				Q.push(Point(i, j));
				while (Q.empty() == false) {
					Point q = Q.front();
					Q.pop();

					if (neighborhoodType == 4) {
						for (int k = 0; k < 4; k++) {
							if ((q.x + di_4[k] < src->rows && q.y + dj_4[k] < src->cols) && (q.x + di_4[k] >= 0 && q.y + dj_4[k] >= 0)) {
								uchar srcNeighbour = src->at<uchar>(q.x + di_4[k], q.y + dj_4[k]);
								uchar labelNeighbour = labeledMatrix.at<uchar>(q.x + di_4[k], q.y + dj_4[k]);

								if (srcNeighbour == 0 && labelNeighbour == 0) {
									labeledMatrix.at<uchar>(q.x + di_4[k], q.y + dj_4[k]) = label;
									Q.push(Point(q.x + di_4[k], q.y + dj_4[k]));
								}
							}
						}
					}
					else if (neighborhoodType == 8) {
						for (int k = 0; k < 8; k++) {
							if ((q.x + di_8[k] < src->rows && q.y + dj_8[k] < src->cols) && (q.x + di_8[k] >= 0 && q.y + dj_8[k] >= 0)) {
								uchar srcNeighbour = src->at<uchar>(q.x + di_8[k], q.y + dj_8[k]);
								uchar labelNeighbour = labeledMatrix.at<uchar>(q.x + di_8[k], q.y + dj_8[k]);

								if (srcNeighbour == 0 && labelNeighbour == 0) {
									labeledMatrix.at<uchar>(q.x + di_8[k], q.y + dj_8[k]) = label;
									Q.push(Point(q.x + di_8[k], q.y + dj_8[k]));
								}
							}
						}
					}
				}
			}
		}
	}
	return labeledMatrix;
}

Mat markPolyp(Mat source)
{
	// Res mats
	Mat dsts[15];
	Mat src = colorToGreyscale(source);

	// Useful data
	int v1[] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
	Mat h1 = Mat(3, 3, CV_32S, v1);
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC3);

	// Negative
	int s = 0;
	imshow("SOURCE", src);

	dsts[s] = histogramEqualization(src);
	s++;

	dsts[s] = gammaCorrection(dsts[s - 1], 3.45);
	s++;

	dsts[s] = automaticGlobalBinarization(dsts[s - 1]);
	s++;

	dsts[s] = negative(dsts[s - 1]);
	s++;

	dsts[s] = morphOperation(dsts[s - 1], 1, 7);
	s++;

	/*
	dsts[s] = morphOperation(dsts[s - 1], 3, 4);
	s++;
	*/

	Mat color1 = bfsLabeling((dsts + s - 1), 8);
	Mat segmentedImage = filterRoundObjects(color1);

	imshow("s", brightnessAdjustment(color1, 60));
	waitKey(0);

	int counter = 0;
	std::map<uchar, BoundingBox> boundingBoxes;
	std::vector<uchar> values;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar currentValue = segmentedImage.at<uchar>(i, j);
			if (currentValue != 0)
			{
				counter++;
				//std::cout << (int) currentValue << std::endl;
				if (boundingBoxes.count(currentValue) == 0)
				{
					BoundingBox box;
					box.xMax = 0;
					box.xMin = width;
					box.yMax = 0;
					box.yMin = height;
					
					std::cout << (int)currentValue << std::endl;

					values.push_back(currentValue);
					boundingBoxes[currentValue] = box;
				}
				else
				{
					if (j > boundingBoxes[currentValue].xMax)
						boundingBoxes[currentValue].xMax = j;
					if (j < boundingBoxes[currentValue].xMin)
						boundingBoxes[currentValue].xMin = j;
					if (i > boundingBoxes[currentValue].yMax)
						boundingBoxes[currentValue].yMax = i;
					if (i < boundingBoxes[currentValue].yMin)
						boundingBoxes[currentValue].yMin = i;
				}
			}
		}
	}

	for (uchar x : values)
	{
		BoundingBox box = boundingBoxes[x];
		float ar = aspectRatio(box.xMax, box.xMin, box.yMax, box.yMin);
		if (ar < 0.9 || ar > 1.21)
			boundingBoxes.erase(x);

	}
	std::cout << "Identified polyps: " << boundingBoxes.size() << "\n";

	Mat markedPoints = Mat::zeros(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (auto const& x : boundingBoxes)
			{
				if ((i == x.second.yMin || i == x.second.yMax) && (j > x.second.xMin && j < x.second.xMax) ||
					((j == x.second.xMin || j == x.second.xMax) && (i > x.second.yMin && i < x.second.yMax)))
				{
					markedPoints.at<uchar>(i, j) = 255;
					dst.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
				}
				else
				{
					if (markedPoints.at<uchar>(i, j) == 0)
						dst.at<Vec3b>(i, j) = source.at<Vec3b>(i, j);
				}
			}
		}
	}
	

	//s++;

	/*
	for (int i = 0; i < s; i++)
	{
		char c[4];
		_itoa(i, c, 10);
		imshow(c, dsts[i]);
	}
	*/

	return dst;
}


int main()
{
	Mat src, dst;
	char fname[255];

	if (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
	}

	dst = markPolyp(src);
	imshow("Final", dst);
	waitKey();
	return 0;
}
