#pragma once
#include "CNNBase.h"
#include <vector>
#include "face_binary_cls.h"
using namespace std;

class CNNBruteforce : public CNNBase {

public:
	void GetClassName() {
		cout << "CNNBruteforce";
	}

	/// <summary>
	/// Image Normalization[Range 0.0 to 1.0] - iteration through all pixel /255
	/// </summary>
	/// <param name="image"></param>
	/// <returns></returns>
	vector<vector<vector<float>>> MatToVector3d(Mat image) {

		vector<vector<vector<float>>> imagePixels(3, vector<vector<float>>(image.rows, vector<float>(image.cols)));
		// Image Normalization
		// 0.0 to 0.1 (range)
		image.convertTo(image, CV_32F, 1.f / 255, 0);
		for (int x = 0; x < image.rows; x++) {
			for (int y = 0; y < image.cols; y++) {
				Vec3f intensity = image.at<Vec3f>(x, y);
				// red = 2; green = 1; blue = 0	
				float blue = intensity.val[0];
				float green = intensity.val[1];
				float red = intensity.val[2];

				imagePixels[0][x][y] = (float)red; // R
				imagePixels[1][x][y] = (float)green; // G
				imagePixels[2][x][y] = (float)blue; // B
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

		for (int f = 0; f < out_channels; f++)
		{
			int row = 0;
			for (int r = 0; r < row_size; r += stride)
			{
				int col = 0;
				for (int c = 0; c < col_size; c += stride)
				{
					float sum = 0;
					float test = 0;
					for (int ch = 0; ch < in_channels; ch++)
					{
						int ch_index = 0;
						for (int ch_row = 0; ch_row < 3; ch_row++) {
							for (int ch_col = 0; ch_col < 3; ch_col++) {
								sum += (paddedInput[ch][r + ch_row][c + ch_col] * cp->p_weight[f * (in_channels * 3 * 3) + ch * (3 * 3) + ch_index++]);
							}
						}
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

		for (int ch = 0; ch < channels; ch++) {
			float sumMean = 0;
			float mean = 0;
			float sumVariance = 0;
			float variance = 0;
			for (int r = 0; r < row; r++)
			{
				for (int c = 0; c < col; c++)
				{
					sumMean += input[ch][r][c];
					sumVariance += input[ch][r][c] * input[ch][r][c];
				}
			}

			mean = sumMean / (row * col);
			variance = sumVariance / (row * col);
			//cout <<"ch:" << ch << "mean:" << mean << ", variance:" << variance << endl;
			// Back populate
			for (int r = 0; r < row; r++)
			{
				for (int c = 0; c < col; c++)
				{
					// Formula: x* = (x - E[x]) / sqrt(var(x))
					// x* new value of a single component
					// E[x] - mean within the batch
					// var(x) - variance within a batch (sqrt(var(x) - Standard Diviation))
					input[ch][r][c] = (input[ch][r][c] - mean) / sqrt(variance);
				}
			}
		}
		return input;
	}

	vector<vector<vector<float>>> ActivationReluLayer(vector<vector<vector<float>>> input) {
		int channels = input.size();
		int row = input[0].size();
		int col = input[0][0].size();
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
		int bsize = psize * psize;

		// Get new dimension
		int dimension = row_size / psize;
		// channel size remains unchanged.
		vector<vector<vector<float>>> output(channels, vector<vector<float>>(dimension, vector<float>(dimension)));

		for (int ch = 0; ch < channels; ch++)
		{
			int row = 0;
			for (int r = 0; r < row_size; r += psize)
			{
				int col = 0;
				for (int c = 0; c < col_size; c += psize)
				{
					vector<float> blocks;
					for (int rb = 0; rb < psize; rb++) {
						for (int cb = 0; cb < psize; cb++) {
							blocks.push_back(input[ch][r+rb][c+cb]);
						}
					}
					output[ch][row][col] = *max_element(blocks.begin(), blocks.end());
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