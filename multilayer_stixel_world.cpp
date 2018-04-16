#include "multilayer_stixel_world.h"
#include "matrix.h"
#include "cost_function.h"

#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using CameraParameters = MultiLayerStixelWrold::CameraParameters;

// Implementation of road disparity estimation by RANSAC
class RoadEstimation
{
public:

	struct Parameters
	{
		int samplingStep;      //!< pixel step of disparity sampling
		int minDisparity;      //!< minimum disparity used for RANSAC

		int maxIterations;     //!< number of iterations of RANSAC
		float inlierRadius;    //!< inlier radius of RANSAC
		float maxCameraHeight; //!< maximum acceptable camera height

		// default settings
		Parameters()
		{
			samplingStep = 2;
			minDisparity = 10;

			maxIterations = 32;
			inlierRadius = 1;
			maxCameraHeight = 5;
		}
	};

	RoadEstimation(const Parameters& param = Parameters()) : param_(param)
	{
	}

	cv::Vec2f compute(const cv::Mat1f& disparity, const CameraParameters& camera)
	{
		CV_Assert(disparity.type() == CV_32F);

		const int w = disparity.rows;
		const int h = disparity.cols;
		const int dmin = param_.minDisparity;

		// sample v-disparity points
		points_.reserve(h * w);
		points_.clear();
		for (int u = 0; u < w; u += param_.samplingStep)
		{
			for (int v = 0; v < h; v += param_.samplingStep)
			{
				const float d = disparity.ptr<float>(u)[v];
				if (d >= dmin)
					points_.push_back(cv::Point2f(static_cast<float>(h - 1 - v), d));
			}
		}
		if (points_.empty())
			return cv::Vec2f(0, 0);

		// estimate line by RANSAC
		const int npoints = static_cast<int>(points_.size());
		cv::RNG random;
		cv::Vec2f bestLine(0, 0);
		int maxInliers = 0;
		for (int iter = 0; iter < param_.maxIterations; iter++)
		{
			const cv::Point2f& pt1 = points_[random.next() % npoints];
			const cv::Point2f& pt2 = points_[random.next() % npoints];
			const float a = (pt2.y - pt1.y) / (pt2.x - pt1.x);
			const float b = -a * pt1.x + pt1.y;

			// estimate camera tilt and height
			const float tilt = atanf((a * camera.v0 + b) / (camera.fu * a));
			const float height = camera.baseline * cosf(tilt) / a;

			// skip if not within valid range
			if (height <= 0.f || height > param_.maxCameraHeight)
				continue;

			// count inliers within a radius and update the best line
			int inliers = 0;
			for (int i = 0; i < npoints; i++)
			{
				const float y = points_[i].x;
				const float x = points_[i].y;
				const float yhat = a * y + b;
				if (fabs(yhat - x) <= param_.inlierRadius)
					inliers++;
			}

			if (inliers > maxInliers)
			{
				maxInliers = inliers;
				bestLine = cv::Vec2f(a, b);
			}
		}

		// apply least squares fitting using inliers around the best line
		double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
		int n = 0;
		for (int i = 0; i < npoints; i++)
		{
			const float x = points_[i].x;
			const float y = points_[i].y;
			const float yhat = bestLine[0] * x + bestLine[1];
			if (fabs(yhat - y) <= param_.inlierRadius)
			{
				sx += x;
				sy += y;
				sxx += x * x;
				syy += y * y;
				sxy += x * y;
				n++;
			}
		}

		const float a = static_cast<float>((n * sxy - sx * sy) / (n * sxx - sx * sx));
		const float b = static_cast<float>((sxx * sy - sxy * sx) / (n * sxx - sx * sx));
		return cv::Vec2f(a, b);
	}

private:
	Parameters param_;
	std::vector<cv::Point2f> points_;
};

static cv::Vec2f calcRoadDisparityParams(const cv::Mat1f& columns, const CameraParameters& camera, int mode)
{
	if (mode == MultiLayerStixelWrold::ROAD_ESTIMATION_AUTO)
	{
		// estimate from v-disparity
		RoadEstimation roadEstimation;
		return roadEstimation.compute(columns, camera);
	}
	else if (mode == MultiLayerStixelWrold::ROAD_ESTIMATION_CAMERA)
	{
		// estimate from camera tilt and height
		const float sinTilt = sinf(camera.tilt);
		const float cosTilt = cosf(camera.tilt);
		const float a = (camera.baseline / camera.height) * cosTilt;
		const float b = (camera.baseline / camera.height) * (camera.fu * sinTilt - camera.v0 * cosTilt);
		return cv::Vec2f(a, b);
	}

	CV_Error(cv::Error::StsInternal, "No such mode");
	return cv::Vec2f(0, 0);
}

MultiLayerStixelWrold::MultiLayerStixelWrold(const Parameters& param) : param_(param)
{
}

void MultiLayerStixelWrold::compute(const cv::Mat& disparity, std::vector<Stixel>& stixels)
{
	CV_Assert(disparity.type() == CV_32F);

	const int stixelWidth = param_.stixelWidth;
	const int w = disparity.cols / stixelWidth;
	const int h = disparity.rows;
	const int fnmax = static_cast<int>(param_.dmax);

	// compute horizontal median of each column
	Matrixf columns(w, h);
	std::vector<float> buf(stixelWidth);
	for (int v = 0; v < h; v++)
	{
		for (int u = 0; u < w; u++)
		{
			// compute horizontal median
			for (int du = 0; du < stixelWidth; du++)
				buf[du] = disparity.at<float>(v, u * stixelWidth + du);
			std::sort(std::begin(buf), std::end(buf));
			const float m = buf[stixelWidth / 2];

			// reverse order of data so that v = 0 points the bottom
			columns(u, h - 1 - v) = m;
		}
	}

	// get camera parameters
	CameraParameters camera = param_.camera;

	// compute road model (assumes planar surface)
	const cv::Vec2f line = calcRoadDisparityParams(columns, camera, param_.roadEstimation);
	const float a = line[0];
	const float b = line[1];

	// compute expected ground disparity
	std::vector<float> groundDisparity(h);
	for (int v = 0; v < h; v++)
		groundDisparity[h - 1 - v] = a * v + b;

	// horizontal row from which road dispaliry becomes negative
	const float vhor = h - 1 + b / a;

	// when AUTO mode, update camera tilt and height
	if (param_.roadEstimation == ROAD_ESTIMATION_AUTO)
	{
		camera.tilt = atanf((a * camera.v0 + b) / (camera.fu * a));
		camera.height = camera.baseline * cosf(camera.tilt) / a;
	}

	// create data cost function of each segment
	NegativeLogDataTermGrd dataTermG(param_.dmax, param_.dmin, param_.sigmaG, param_.pOutG, param_.pInvG, camera,
		groundDisparity, vhor, param_.sigmaH, param_.sigmaA);
	NegativeLogDataTermObj dataTermO(param_.dmax, param_.dmin, param_.sigmaO, param_.pOutO, param_.pInvO, camera, param_.deltaz);
	NegativeLogDataTermSky dataTermS(param_.dmax, param_.dmin, param_.sigmaS, param_.pOutS, param_.pInvS);

	// create prior cost function of each segment
	const int G = NegativeLogPriorTerm::G;
	const int O = NegativeLogPriorTerm::O;
	const int S = NegativeLogPriorTerm::S;
	NegativeLogPriorTerm priorTerm(h, vhor, param_.dmax, param_.dmin, camera.baseline, camera.fu, param_.deltaz,
		param_.eps, param_.pOrd, param_.pGrav, param_.pBlg, groundDisparity);

	// data cost LUT
	Matrixf costsG(w, h), costsO(w, h, fnmax), costsS(w, h), sum(w, h);
	Matrixi valid(w, h);

	// cost table
	Matrixf costTable(w, h, 3), dispTable(w, h, 3);
	Matrix<cv::Point> indexTable(w, h, 3);

	// process each column
	int u;
#pragma omp parallel for
	for (u = 0; u < w; u++)
	{
		//////////////////////////////////////////////////////////////////////////////
		// pre-computate LUT
		//////////////////////////////////////////////////////////////////////////////
		float tmpSumG = 0.f;
		float tmpSumS = 0.f;
		std::vector<float> tmpSumO(fnmax, 0.f);

		float tmpSum = 0.f;
		int tmpValid = 0;

		for (int v = 0; v < h; v++)
		{
			// measured disparity
			const float d = columns(u, v);

			// pre-computation for ground costs
			tmpSumG += dataTermG(d, v);
			costsG(u, v) = tmpSumG;

			// pre-computation for sky costs
			tmpSumS += dataTermS(d);
			costsS(u, v) = tmpSumS;

			// pre-computation for object costs
			for (int fn = 0; fn < fnmax; fn++)
			{
				tmpSumO[fn] += dataTermO(d, fn);
				costsO(u, v, fn) = tmpSumO[fn];
			}

			// pre-computation for mean disparity of stixel
			if (d >= 0.f)
			{
				tmpSum += d;
				tmpValid++;
			}
			sum(u, v) = tmpSum;
			valid(u, v) = tmpValid;
		}

		//////////////////////////////////////////////////////////////////////////////
		// compute cost tables
		//////////////////////////////////////////////////////////////////////////////
		for (int vT = 0; vT < h; vT++)
		{
			float minCostG, minCostO, minCostS;
			float minDispG, minDispO, minDispS;
			cv::Point minPosG(G, 0), minPosO(O, 0), minPosS(S, 0);

			// process vB = 0
			{
				// compute mean disparity within the range of vB to vT
				const float d1 = sum(u, vT) / std::max(valid(u, vT), 1);
				const int fn = cvRound(d1);

				// initialize minimum costs
				minCostG = costsG(u, vT) + priorTerm.getG0(vT);
				minCostO = costsO(u, vT, fn) + priorTerm.getO0(vT);
				minCostS = costsS(u, vT) + priorTerm.getS0(vT);
				minDispG = minDispO = minDispS = d1;
			}

			for (int vB = 1; vB <= vT; vB++)
			{
				// compute mean disparity within the range of vB to vT
				const float d1 = (sum(u, vT) - sum(u, vB - 1)) / std::max(valid(u, vT) - valid(u, vB - 1), 1);
				const int fn = cvRound(d1);

				// compute data terms costs
				const float dataCostG = vT < vhor ? costsG(u, vT) - costsG(u, vB - 1) : N_LOG_0_0;
				const float dataCostO = costsO(u, vT, fn) - costsO(u, vB - 1, fn);
				const float dataCostS = vT < vhor ? N_LOG_0_0 : costsS(u, vT) - costsS(u, vB - 1);

				// compute priors costs and update costs
				const float d2 = dispTable(u, vB - 1, 1);

#define UPDATE_COST(C1, C2) \
				const float cost##C1##C2 = dataCost##C1 + priorTerm.get##C1##C2(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, C2); \
				if (cost##C1##C2 < minCost##C1) \
				{ \
					minCost##C1 = cost##C1##C2; \
					minDisp##C1 = d1; \
					minPos##C1 = cv::Point(C2, vB - 1); \
				} \

				UPDATE_COST(G, G);
				UPDATE_COST(G, O);
				UPDATE_COST(G, S);
				UPDATE_COST(O, G);
				UPDATE_COST(O, O);
				UPDATE_COST(O, S);
				UPDATE_COST(S, G);
				UPDATE_COST(S, O);
				UPDATE_COST(S, S);
			}

			costTable(u, vT, G) = minCostG;
			costTable(u, vT, O) = minCostO;
			costTable(u, vT, S) = minCostS;

			dispTable(u, vT, G) = minDispG;
			dispTable(u, vT, O) = minDispO;
			dispTable(u, vT, S) = minDispS;

			indexTable(u, vT, G) = minPosG;
			indexTable(u, vT, O) = minPosO;
			indexTable(u, vT, S) = minPosS;
		}
	}

	//////////////////////////////////////////////////////////////////////////////
	// backtracking step
	//////////////////////////////////////////////////////////////////////////////
	stixels.clear();
	for (int u = 0; u < w; u++)
	{
		float minCost = std::numeric_limits<float>::max();
		cv::Point minPos;
		for (int c = 0; c < 3; c++)
		{
			const float cost = costTable(u, h - 1, c);
			if (cost < minCost)
			{
				minCost = cost;
				minPos = cv::Point(c, h - 1);
			}
		}

		while (minPos.y > 0)
		{
			const cv::Point p1 = minPos;
			const cv::Point p2 = indexTable(u, p1.y, p1.x);
			if (p1.x == O) // object
			{
				Stixel stixel;
				stixel.u = stixelWidth * u + stixelWidth / 2;
				stixel.vT = h - 1 - p1.y;
				stixel.vB = h - 1 - (p2.y + 1);
				stixel.width = stixelWidth;
				stixel.disp = dispTable(u, p1.y, p1.x);
				stixels.push_back(stixel);
			}
			minPos = p2;
		}
	}
}
