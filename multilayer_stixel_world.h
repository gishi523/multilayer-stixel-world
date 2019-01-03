#ifndef __MULTILAYER_STIXEL_WORLD_H__
#define __MULTILAYER_STIXEL_WORLD_H__

#include <opencv2/opencv.hpp>

struct Stixel
{
	int u;
	int vT;
	int vB;
	int width;
	float disp;
};

class MultiLayerStixelWorld
{
public:

	enum
	{
		ROAD_ESTIMATION_AUTO = 0,
		ROAD_ESTIMATION_CAMERA
	};

	struct CameraParameters
	{
		float fu;
		float fv;
		float u0;
		float v0;
		float baseline;
		float height;
		float tilt;

		// default settings
		CameraParameters()
		{
			fu = 1.f;
			fv = 1.f;
			u0 = 0.f;
			v0 = 0.f;
			baseline = 0.2f;
			height = 1.f;
			tilt = 0.f;
		}
	};

	struct Parameters
	{
		// stixel width
		int stixelWidth;

		// disparity range
		float dmin;
		float dmax;

		// disparity measurement uncertainty
		float sigmaG;
		float sigmaO;
		float sigmaS;

		// camera height and tilt uncertainty
		float sigmaH;
		float sigmaA;

		// outlier rate
		float pOutG;
		float pOutO;
		float pOutS;

		// probability of invalid disparity
		float pInvD;
		float pInvG;
		float pInvO;
		float pInvS;

		// probability for regularization
		float pOrd;
		float pGrav;
		float pBlg;

		float deltaz;
		float eps;

		// road disparity estimation
		int roadEstimation;

		// camera parameters
		CameraParameters camera;

		// default settings
		Parameters()
		{
			// stixel width
			stixelWidth = 7;

			// disparity range
			dmin = 0;
			dmax = 64;

			// disparity measurement uncertainty
			sigmaG = 1.5f;
			sigmaO = 1.5f;
			sigmaS = 1.2f;

			// camera height and tilt uncertainty
			sigmaH = 0.05f;
			sigmaA = 0.05f * static_cast<float>(CV_PI) / 180.f;

			// outlier rate
			pOutG = 0.15f;
			pOutO = 0.15f;
			pOutS = 0.4f;

			// probability of invalid disparity
			pInvD = 0.25f;
			pInvG = 0.34f;
			pInvO = 0.3f;
			pInvS = 0.36f;

			// probability for regularization
			pOrd = 0.1f;
			pGrav = 0.1f;
			pBlg = 0.001f;

			deltaz = 3.f;
			eps = 1.f;

			// road disparity estimation
			roadEstimation = ROAD_ESTIMATION_AUTO;

			// camera parameters
			camera = CameraParameters();
		}
	};

	MultiLayerStixelWorld(const Parameters& param);

	void compute(const cv::Mat& disparity, std::vector<Stixel>& stixels);

private:

	Parameters param_;
};

#endif // !__STIXEL_WORLD_H__