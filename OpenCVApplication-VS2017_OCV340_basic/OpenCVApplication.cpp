// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <random>
#include <stack>
#include <fstream>
#define MIN_AR 0.9
#define MAX_AR 1.21


using namespace cv;

Mat dstGlobal;

typedef struct _boundingBox
{
	int xMin;
	int xMax;
	int yMin;
	int yMax;
}BoundingBox;

int di_4[4] = { -1,0,1,0 };
int dj_4[4] = { 0,-1,0,1 };

int di[4] = { -1,-1,-1,0 };
int dj[4] = { -1,0,1,-1 };

int di_8[8] = { -1,-1,0,1,1,1,0,-1 };
int dj_8[8] = { 0,-1,-1,-1,0,1,1,1 };

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

bool isInside(Mat img, int i, int j)
{
	int rows = img.rows;
	int cols = img.cols;

	if (i >= 0 && i < rows && j >= 0 && j < cols)
		return true;

	return false;
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

bool testColor(Vec3b a, Vec3b b)
{
	return (a[0] == b[0] &&
		a[1] == b[1] &&
		a[2] == b[2]);
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
					if (tr > 0.7 && a > 50)
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
	float ai = averageIntensity(src);

	// Preprocessing
	dsts[s] = histogramEqualization(src);
	s++;
	
	float v = (210 / ai);

	dsts[s] = gammaCorrection(dsts[s - 1], v);
	s++;

	// Binarization for geometric-property filtering
	dsts[s] = automaticGlobalBinarization(dsts[s - 1]);
	s++;

	dsts[s] = negative(dsts[s - 1]);
	s++;

	// Erode
	dsts[s] = morphOperation(dsts[s - 1], 1, 7);
	s++;

	// Labelling
	Mat color = bfsLabeling((dsts + s - 1), 8);

	// Filter by geometric properties
	Mat segmentedImage = filterRoundObjects(color);

	int counter = 0;
	std::map<uchar, BoundingBox> boundingBoxes;
	std::vector<uchar> values;

	// Finding bounding boxes
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar currentValue = segmentedImage.at<uchar>(i, j);
			if (currentValue != 0)
			{
				counter++;
				if (boundingBoxes.count(currentValue) == 0)
				{
					BoundingBox box;
					box.xMax = 0;
					box.xMin = width;
					box.yMax = 0;
					box.yMin = height;
					
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

	// Filter polyps with aspect ratio out of range
	for (uchar x : values)
	{
		BoundingBox box = boundingBoxes[x];
		float ar = aspectRatio(box.xMax, box.xMin, box.yMax, box.yMin);
		if (ar < MIN_AR|| ar > MAX_AR)
			boundingBoxes.erase(x);
	}
	
	std::cout << "Number of identified polyps: " << boundingBoxes.size() << "\n";

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

	imshow("SOURCE", src);
	dst = markPolyp(src);

	imshow("MARKED SOURCE", dst);
	waitKey();
	return 0;
}
