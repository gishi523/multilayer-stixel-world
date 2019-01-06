#ifndef __MULTILAYER_STIXEL_WORLD_H__
#define __MULTILAYER_STIXEL_WORLD_H__

#include <opencv2/opencv.hpp>

/** @brief Stixel struct
*/
struct Stixel
{
	int u;                        //!< stixel center x position
	int vT;                       //!< stixel top y position
	int vB;                       //!< stixel bottom y position
	int width;                    //!< stixel width
	float disp;                   //!< stixel average disparity
};

/** @brief MultiLayerStixelWorld class.

The class implements the Multi-Layered Stixel computation based on [1].
[1] D. Pfeiffer: "The Stixel World - A Compact Medium-level Representation for Efficiently Modeling Three-dimensional Environments"
*/
class MultiLayerStixelWorld
{
public:

	enum
	{
		ROAD_ESTIMATION_AUTO = 0, //!< road disparity are estimated by input disparity
		ROAD_ESTIMATION_CAMERA    //!< road disparity are estimated by camera tilt and height
	};

	/** @brief CameraParameters struct
	*/
	struct CameraParameters
	{
		float fu;                 //!< focal length x (pixel)
		float fv;                 //!< focal length y (pixel)
		float u0;                 //!< principal point x (pixel)
		float v0;                 //!< principal point y (pixel)
		float baseline;           //!< baseline (meter)
		float height;             //!< height position (meter), ignored when ROAD_ESTIMATION_AUTO
		float tilt;               //!< tilt angle (radian), ignored when ROAD_ESTIMATION_AUTO

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

	/** @brief Parameters struct
	*/
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

		// scale down factor in height
		// this reduces the computation time significantly
		float verticalScaleDown;

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
			sigmaS = 1.5f;

			// camera height and tilt uncertainty
			sigmaH = 0.01f;
			sigmaA = 0.001f;

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

			// scale down factor in height
			// this reduces the computation time significantly
			verticalScaleDown = 2.f;
		}
	};

	/** @brief The constructor
	@param param input parameters
	*/
	MultiLayerStixelWorld(const Parameters& param);

	/** @brief Computes stixels in a disparity map
	@param disparity Input 32-bit single-channel disparity map
	@param stixels Output array of stixels
	*/
	void compute(const cv::Mat& disparity, std::vector<Stixel>& stixels);

private:

	Parameters param_;
};

#endif // !__STIXEL_WORLD_H__
