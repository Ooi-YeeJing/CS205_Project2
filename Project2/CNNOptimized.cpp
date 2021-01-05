#pragma once
#include "CNNBase.h"
#include <vector>
#include "face_binary_cls.h"
using namespace std;

class CNNOptimized : public CNNBase {

public:

	void GetClassName() {
		cout << "CNNOptimized";
	}
	/// <summary>
	/// Image Normalization[Range 0.0 to 1.0] - using cv.converTo with 1/255.
	/// </summary>
	/// <param name="image"></param>
	/// <returns></returns>

	vector<vector<vector<float>>> MatToVector3d(Mat image) {
		vector<vector<vector<float>>> imagePixels(3, vector<vector<float>>(image.rows, vector<float>(image.cols)));

		for (int x = 0; x < image.rows; x++) {
			for (int y = 0; y < image.cols; y++) {
				Vec3b intensity = image.at<Vec3b>(x, y);

				// red = 2; green = 1; blue = 0				
				int blue = intensity.val[0];
				int green = intensity.val[1];
				int red = intensity.val[2];

				//int sum = red + green + blue;
				//cout << "C[" << x << "," << y << "]" << "|R[" << red << "]|G[" << green << "]|B[" << blue << "]" << endl;
				//imagePixels[0][x][y] = sum == 0 ? 0 : (float)red / (float)sum; // R
				//imagePixels[1][x][y] = sum == 0 ? 0 : (float)green / (float)sum; // G
				//imagePixels[2][x][y] = sum == 0 ? 0 : (float)blue / (float)sum; // B

				imagePixels[0][x][y] = (float)red / 255; // R
				imagePixels[1][x][y] = (float)green / 255; // G
				imagePixels[2][x][y] = (float)blue / 255; // B

				//cout << "imagePixels[0][y][x]:" << imagePixels[0][y][x] << "|imagePixels[1][y][x]:" << imagePixels[1][y][x]  << "|imagePixels[2][y][x]:" << imagePixels[2][y][x] << endl;
			}
		}
		return imagePixels;
	}

	vector<vector<vector<float>>> ConvolutionalLayer(vector<vector<vector<float>>> input, conv_param* cp) {

		// Initialize Input
		vector<vector<vector<float>>> paddedInput = input;

		// Calculate output dimension
		int padding = cp->pad;
		int padsize = 0;
		int stride = cp->stride;
		int ch_size = input.size(); // channel/kernel size 
		int r_size = input[0].size(); // row
		int c_size = input[0][0].size(); // column

		// Padding Required?
		if (padding) {
			padsize = 2;
			// Add padding to input.
			paddedInput.clear();
			paddedInput.resize(ch_size, vector<vector<float>>(r_size + padsize, vector<float>(c_size + padsize)));

			for (int r = 0; r < r_size; r++)
			{
				for (int c = 0; c < c_size; c++)
				{
					paddedInput[0][r + 1][c + 1] = input[0][r][c];
					paddedInput[1][r + 1][c + 1] = input[1][r][c];
					paddedInput[2][r + 1][c + 1] = input[2][r][c];
				}
			}
		}

		// Image is always a square
		// Initialize the output dimension.
		int row_size = paddedInput[0].size() - 2;
		int col_size = paddedInput[0][0].size() - 2;
		int dimension = (r_size - CONVOLUTION_FILTER + padsize) / stride + 1;

		// filters
		int out_channels = cp->out_channels;
		int in_channels = cp->in_channels;

		// output 
		// kernel size = out_channels;
		// row and col = new calculated dimension based on padding and stride.
		vector<vector<vector<float>>> output(out_channels, vector<vector<float>>(dimension, vector<float>(dimension)));

#pragma omp parallel
#pragma omp for
		for (int f = 0; f < out_channels; f++)
		{
			int row = 0;
			for (int r = 0; r < row_size; r += stride)
			{
				int col = 0;
				for (int c = 0; c < col_size; c += stride)
				{
					float sum = 0;
					for (int ch = 0; ch < in_channels; ch++)
					{
						int wIndex = f * (in_channels * 3 * 3) + ch * (3 * 3);

						sum += (paddedInput[ch][r][c]			* cp->p_weight[wIndex + 0]) +
							   (paddedInput[ch][r][c + 1]		* cp->p_weight[wIndex + 1]) +
							   (paddedInput[ch][r][c + 2]		* cp->p_weight[wIndex + 2]) +
							   (paddedInput[ch][r + 1][c]		* cp->p_weight[wIndex + 3]) +
							   (paddedInput[ch][r + 1][c + 1]	* cp->p_weight[wIndex + 4]) +
							   (paddedInput[ch][r + 1][c + 2]	* cp->p_weight[wIndex + 5]) +
							   (paddedInput[ch][r + 2][c]		* cp->p_weight[wIndex + 6]) +
							   (paddedInput[ch][r + 2][c + 1]	* cp->p_weight[wIndex + 7]) +
							   (paddedInput[ch][r + 2][c + 2]	* cp->p_weight[wIndex + 8]);

					}
					output[f][row][col] = sum + cp->p_bias[f]; // include bias
					col++;
				}
				row++;
			}
		}

		return output;
	}

	vector<vector<vector<float>>> BatchNormalizationLayer(vector<vector<vector<float>>> input) {
		int channels = input.size();
		int row = input[0].size();
		int col = input[0][0].size();
		int dimension = row * col;

#pragma omp parallel
#pragma omp for
		for (int ch = 0; ch < channels; ch++) {
			float sumMean = 0;
			float mean = 0;
			float sumVariance = 0;
			float sqrtChannel = 0;
			for (int r = 0; r < row; r++)
			{
				for (int c = 0; c < col; c++)
				{
					sumMean += input[ch][r][c];
					sumVariance += input[ch][r][c] * input[ch][r][c];
				}
			}

			mean = sumMean / dimension;
			sqrtChannel = sqrt(sumVariance / dimension);

			//cout <<"ch:" << ch << "mean:" << mean << ", variance:" << variance << endl;
			// Back populate

#pragma omp parallel
#pragma omp for
			for (int r = 0; r < row; r++)
			{
				for (int c = 0; c < col; c++)
				{
					// Formula: x* = (x - E[x]) / sqrt(var(x))
					// x* new value of a single component
					// E[x] - mean within the batch
					// var(x) - variance within a batch (sqrt(var(x) - Standard Diviation))
					input[ch][r][c] = (input[ch][r][c] - mean) / sqrtChannel;
				}
			}
		}
		return input;
	}

	vector<vector<vector<float>>> ActivationReluLayer(vector<vector<vector<float>>> input) {
		int channels = input.size();
		int row = input[0].size();
		int col = input[0][0].size();
#pragma omp parallel
#pragma omp for
		for (int ch = 0; ch < channels; ch++)
		{
			for (int r = 0; r < row; r++)
			{
				for (int c = 0; c < col; c++)
				{
					input[ch][r][c] = std::max((float)0, input[ch][r][c]);
				}
			}
		}
		return input;
	}

	vector<vector<vector<float>>> MaxPoolingLayer(vector<vector<vector<float>>> input, int psize) {
		int channels = input.size();
		int row_size = input[0].size();
		int col_size = input[0][0].size();

		// Get new dimension
		int dimension = row_size / psize;
		// channel size remains unchanged.
		vector<vector<vector<float>>> output(channels, vector<vector<float>>(dimension, vector<float>(dimension)));
#pragma omp parallel
#pragma omp for
		for (int ch = 0; ch < channels; ch++)
		{
			int row = 0;
			for (int r = 0; r < row_size; r += psize)
			{
				int col = 0;
				for (int c = 0; c < col_size; c += psize)
				{
					output[ch][row][col] = max(output[ch][row][col], input[ch][r][c]);
					output[ch][row][col] = max(output[ch][row][col], input[ch][r][c + 1]);
					output[ch][row][col] = max(output[ch][row][col], input[ch][r + 1][c]);
					output[ch][row][col] = max(output[ch][row][col], input[ch][r + 1][c + 1]);
					col++;
				}
				row++;
			}
		}
		return output;
	}

	vector<float> FlattenLayer(vector<vector<vector<float>>> input) {
		int channels = input.size();
		int row = input[0].size();
		int col = input[0][0].size();
		int dimension = channels * row * col;
		vector<float> output(dimension);
		int idx = 0;
		for (int ch = 0; ch < channels; ch++)
		{
			for (int r = 0; r < row; r++)
			{
				for (int c = 0; c < col; c++)
				{
					output[idx++] = input[ch][r][c];
				}
			}
		}
		return output;
	}

	vector<float> FullyConnectedLayer(vector<float> input, fc_param* fcp) {
		int in_features = fcp->in_features; // 2048
		int out_features = fcp->out_features; // 2

		vector<float> fc_output(out_features);
		for (int o = 0; o < out_features; o++)
		{
			float sum = 0;
			for (int i = 0; i < in_features; i++) {
				//sum += input[i] * fcp->p_weight[o * out_features + i];
				sum += input[i] * fcp->p_weight[(o * in_features) + i];
				//cout << "sum:" << sum << "|input:" << i << "|" << input[i] << ",p_weight:" << o * out_features + i << "|" << fcp->p_weight[o * out_features + i] << endl;
			}
			fc_output[o] = sum + fcp->p_bias[o];
		}

		return fc_output;
	}

	vector<float> SoftMaxLayer(vector<float> input) {
		int size = input.size();
		float sum = 0;
		vector<float> exponent(size);
		for (int i = 0; i < size; i++) {
			exponent[i] = exp(input[i]);
			sum += exponent[i];
		}
		for (int i = 0; i < size; i++) {
			input[i] = exponent[i] / sum;
		}
		return input;
	}
};