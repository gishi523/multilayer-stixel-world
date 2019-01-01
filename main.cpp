#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "multilayer_stixel_world.h"

static cv::Scalar computeColor(float val)
{
	const float hscale = 6.f;
	float h = 0.6f * (1.f - val), s = 1.f, v = 1.f;
	float r, g, b;

	static const int sector_data[][3] =
	{ { 1,3,0 },{ 1,0,2 },{ 3,0,1 },{ 0,2,1 },{ 0,1,3 },{ 2,1,0 } };
	float tab[4];
	int sector;
	h *= hscale;
	if (h < 0)
		do h += 6; while (h < 0);
	else if (h >= 6)
		do h -= 6; while (h >= 6);
	sector = cvFloor(h);
	h -= sector;
	if ((unsigned)sector >= 6u)
	{
		sector = 0;
		h = 0.f;
	}

	tab[0] = v;
	tab[1] = v*(1.f - s);
	tab[2] = v*(1.f - s*h);
	tab[3] = v*(1.f - s*(1.f - h));

	b = tab[sector_data[sector][0]];
	g = tab[sector_data[sector][1]];
	r = tab[sector_data[sector][2]];
	return 255 * cv::Scalar(b, g, r);
}

static cv::Scalar dispToColor(float disp, float maxdisp)
{
	if (disp < 0)
		return cv::Scalar(128, 128, 128);
	return computeColor(std::min(disp, maxdisp) / maxdisp);
}

static void drawStixel(cv::Mat& img, const Stixel& stixel, cv::Scalar color)
{
	const int radius = std::max(stixel.width / 2, 1);
	const cv::Point tl(stixel.u - radius, stixel.vT);
	const cv::Point br(stixel.u + radius, stixel.vB);
	cv::rectangle(img, cv::Rect(tl, br), color, -1);
	cv::rectangle(img, cv::Rect(tl, br), cv::Scalar(255, 255, 255), 1);
}

int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format camera.xml" << std::endl;
		return -1;
	}

	// stereo SGBM
	const int wsize = 11;
	const int numDisparities = 64;
	auto sgbm = cv::StereoSGBM::create(0, numDisparities, wsize);
	sgbm->setP1(8 * wsize * wsize);
	sgbm->setP2(32 * wsize * wsize);
	sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

	// read camera parameters
	const cv::FileStorage fs(argv[3], cv::FileStorage::READ);
	CV_Assert(fs.isOpened());

	// input parameters
	MultiLayerStixelWorld::Parameters param;
	param.camera.fu = fs["FocalLengthX"];
	param.camera.fv = fs["FocalLengthY"];
	param.camera.u0 = fs["CenterX"];
	param.camera.v0 = fs["CenterY"];
	param.camera.baseline = fs["BaseLine"];
	param.camera.height = fs["Height"];
	param.camera.tilt = fs["Tilt"];
	param.dmax = numDisparities;

	cv::Mat disparity;
	MultiLayerStixelWorld stixelWorld(param);

	for (int frameno = 1;; frameno++)
	{
		cv::Mat I1 = cv::imread(cv::format(argv[1], frameno), cv::IMREAD_UNCHANGED);
		cv::Mat I2 = cv::imread(cv::format(argv[2], frameno), cv::IMREAD_UNCHANGED);

		if (I1.empty() || I2.empty())
		{
			std::cerr << "imread failed." << std::endl;
			break;
		}

		CV_Assert(I1.size() == I2.size() && I1.type() == I2.type());
		CV_Assert(I1.type() == CV_8U || I1.type() == CV_16U);

		if (I1.type() == CV_16U)
		{
			cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX);
			cv::normalize(I2, I2, 0, 255, cv::NORM_MINMAX);
			I1.convertTo(I1, CV_8U);
			I2.convertTo(I2, CV_8U);
		}

		// compute dispaliry
		sgbm->compute(I1, I2, disparity);
		disparity.convertTo(disparity, CV_32F, 1. / cv::StereoSGBM::DISP_SCALE);

		// compute stixels
		const auto t1 = std::chrono::system_clock::now();

		std::vector<Stixel> stixels;
		stixelWorld.compute(disparity, stixels);

		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cout << "stixel computation time: " << duration << "[msec]" << std::endl;

		// colorize disparity
		cv::Mat disparityColor;
		disparity.convertTo(disparityColor, CV_8U, 255. / numDisparities);
		cv::applyColorMap(disparityColor, disparityColor, cv::COLORMAP_JET);

		// draw stixels
		cv::Mat draw;
		cv::cvtColor(I1, draw, cv::COLOR_GRAY2BGR);

		cv::Mat stixelImg = cv::Mat::zeros(I1.size(), draw.type());
		for (const auto& stixel : stixels)
			drawStixel(stixelImg, stixel, dispToColor(stixel.disp, 1.f * numDisparities));
		cv::addWeighted(draw, 1, stixelImg, 0.5, 0, draw);

		cv::imshow("disparity", disparityColor);
		cv::imshow("stixels", draw);

		const char c = cv::waitKey(1);
		if (c == 27)
			break;
		if (c == 'p')
			cv::waitKey(0);
	}
}
