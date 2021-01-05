#pragma once

#include <iostream>
#include <vector>
#include "face_binary_cls.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class CNNBase {
private:

public:
	static const int CONVOLUTION_FILTER = 3; // 3x3

	// Factory Method
	static CNNBase* make_cnnbase(int choice);

	// Virtual Methods
	virtual vector<vector<vector<float>>> MatToVector3d(Mat image) = 0;
	virtual vector<vector<vector<float>>> ConvolutionalLayer(vector<vector<vector<float>>> input, conv_param *cp) = 0;
	virtual vector<vector<vector<float>>> BatchNormalizationLayer(vector<vector<vector<float>>> input) = 0;
	virtual vector<vector<vector<float>>> ActivationReluLayer(vector<vector<vector<float>>> input) = 0;
	virtual vector<vector<vector<float>>> MaxPoolingLayer(vector<vector<vector<float>>> input, int psize) = 0;
	virtual vector<float> FlattenLayer(vector<vector<vector<float>>> input) = 0;
	virtual vector<float> FullyConnectedLayer(vector<float> input, fc_param* fcp) = 0;
	virtual vector<float> SoftMaxLayer(vector<float> input) = 0;
	virtual void GetClassName() = 0;


	void PrintMatrix(vector<vector<vector<float>>> input) {
		for (int i = 0; i < input.size(); i++) {
			cout << "channel:" << i << endl;
			for (int j = 0; j < input[i].size(); j++) {
				for (int x = 0; x < input[i][j].size(); x++) {
					cout << input[i][j][x] << ",";
				}
				cout << endl;
			}
		}
	}
};