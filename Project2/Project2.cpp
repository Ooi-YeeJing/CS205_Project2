// Project2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "face_binary_cls.h"
#include <iostream>
#include <vector>
#include "CNNBruteforce.cpp"
#include "CNNOptimized.cpp"
#include "CNNPlayground.cpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

CNNBase* CNNBase::make_cnnbase(int choice) {
	if (choice == 0)
		return new CNNBruteforce;
	else if (choice == 1)
		return new CNNOptimized;
	else
		return new CNNPlayground;
}

typedef struct cnn_arg {
	int option;
	string image;
}cnn_arg;

static void show_usage()
{
	cout << "Developer: Ooi Yee Jing\n";
	cout << "Project 2 Usage:\n";
	cout << "\t-h,--help\tShow this help message:\n";
	cout << "\t-o,--options\tUse different implementation to run CNN\n";
	cout << "\t\t0:CNNBruteforce\n";
	cout << "\t\t1:CNNOptimized\n";
	cout << "\t\t2:CNNPlayground\n";
	cout << "\t-img,--image\tFull path for the image\n";
	cout << "Example:Project2 -o=<option> -img=<fullpath image>\n";
	cout << "Example:Project2 -o=1 -img=c:\\temp\\sample\\face.jpg\n";
}

/*
 * Erase First Occurrence of given  substring from main string.
 */
void eraseSubStr(std::string& mainStr, const std::string& toErase)
{
	// Search for the substring in string
	size_t pos = mainStr.find(toErase);
	if (pos != std::string::npos)
	{
		// If found then erase it from string
		mainStr.erase(pos, toErase.length());
	}
}

/// <summary>
/// CNN Execution
/// 1. Read Image > Convert Image to 3d values (RGB) with normalized values.
///		a. Note> There is no 32 bit opencv, will need to cmake. so change to x64 Debug.
/// 2. ConvolutionLayer1
///		a. BatchNormalizationLayer
///		b. ReluLayer
///		c. MaxPoolingLayer (2x2)
/// 3. ConvolutionLayer2
///		a. BatchNormalizationLayer
///		b. ReluLayer
///		c. MaxPoolingLayer (2x2)
/// 4. ConvolutionLayer3
///		a. BatchNormalizationLayer
///		b. ReluLayer
/// 5. FlatternLayer
/// 6. FullyConnectedLayer
/// 7. SoftMaxLayer.
/// </summary>
/// <param name="cnnarg"></param>
/// <returns></returns>
int cnn_execute(cnn_arg cnnarg) {

	//Initialize CNN Factory
	CNNBase* cnn = CNNBase::make_cnnbase(cnnarg.option);
	//CNNBase* cnn = CNNBase::make_cnnbase(0);
	
	cout << "CNN implementation:";
	cnn->GetClassName();
	cout << endl;
	cout << "*****************************\n";

	// 1.Read Image
	//Mat image = imread("samples\\face.jpg");
	Mat image = imread(cnnarg.image, COLOR_BGR2RGB);
	if (image.empty()) {
		cout << "Invalid Image, try again" << endl;
		return 0;
	}

	
	TickMeter cvtmall;
	cvtmall.start();
	TickMeter cvtm;
	cvtm.start();

	// 1. Image pixel 3 channels, Mat3d Image is BGR
	vector<vector<vector<float>>> imagePixels = cnn->MatToVector3d(image);
	
	//cnn->PrintMatrix(imagePixels);
	cvtm.stop();
	printf("MatToVector3d = %gms\n", cvtm.getTimeMilli());

	cvtm.reset();
	cvtm.start();
	// 2. Convolutional Layer
	vector<vector<vector<float>>> output = cnn->ConvolutionalLayer(imagePixels, &conv_params[0]);
	output = cnn->BatchNormalizationLayer(output);
	output = cnn->ActivationReluLayer(output);
	output = cnn->MaxPoolingLayer(output, 2);
	//cnn->PrintMatrix(output);
	cvtm.stop();
	printf("1st ConvolutionalLayer = %gms\n", cvtm.getTimeMilli());

	cvtm.reset();
	cvtm.start();
	// 3. Convolutional Layer
	output = cnn->ConvolutionalLayer(output, &conv_params[1]);
	output = cnn->BatchNormalizationLayer(output);
	output = cnn->ActivationReluLayer(output);
	output = cnn->MaxPoolingLayer(output, 2);
	cvtm.stop();
	printf("2nd ConvolutionalLayer = %gms\n", cvtm.getTimeMilli());

	cvtm.reset();
	cvtm.start();
	// 4. Convolutional Layer
	output = cnn->ConvolutionalLayer(output, &conv_params[2]);
	output = cnn->BatchNormalizationLayer(output);
	output = cnn->ActivationReluLayer(output);
	cvtm.stop();
	printf("3rd ConvolutionalLayer = %gms\n", cvtm.getTimeMilli());

	cvtm.start();
	// 5. Flatten Layer
	vector<float> flatten = cnn->FlattenLayer(output);
	cvtm.stop();
	printf("FlattenLayer = %gms\n", cvtm.getTimeMilli());

	cvtm.reset();
	cvtm.start();
	// 6. Fully Connected Layer
	vector<float> fullyConnected = cnn->FullyConnectedLayer(flatten, &fc_params[0]);
	cvtm.stop();
	printf("FullyConnectedLayer = %gms\n", cvtm.getTimeMilli());

	cvtm.reset();
	cvtm.start();
	// 7. SoftMax Layer
	fullyConnected = cnn->SoftMaxLayer(fullyConnected);
	cvtm.stop();
	printf("SoftMaxLayer = %gms\n", cvtm.getTimeMilli());

	//cout << "*****************************\n";
	cout << "bg:" << fullyConnected[0] << " face:" << fullyConnected[1] << endl;
	cout << "*****************************\n";
	cvtmall.stop();
	printf("overall = %gms\n", cvtmall.getTimeMilli());

	return 0;
}

int main(int argc, char** argv)
{
	if (argc <= 2) {
		show_usage();
		return 1;
	}
	cnn_arg cnnargs;
	
	for (int i = 0; i < argc; ++i) {
		string arg = argv[i];
		if ((arg.rfind("-h", 0)==0) || (arg.rfind("--help", 0)==0)) {
			show_usage();
			return 0;
		}
		else if ((arg.rfind("-o=", 0)==0) || (arg.rfind("--options=", 0)==0)) {
			eraseSubStr(arg, "-o="); 
			eraseSubStr(arg, "--options=");
			cnnargs.option =stoi(arg);
		}
		else if ((arg.rfind("-img=", 0)==0) || (arg.rfind("--image=", 0)==0)){
			eraseSubStr(arg, "-img=");
			eraseSubStr(arg, "--image=");
			cnnargs.image = arg;
		}
	}
	cout << "Ooi Yee Jing\n";
	cnn_execute(cnnargs);
}
