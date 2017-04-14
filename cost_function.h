#ifndef __COST_FUNCTION_H__
#define __COST_FUNCTION_H__

#include "matrix.h"

#include <algorithm>
#include <numeric>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>

//////////////////////////////////////////////////////////////////////////////
// data cost functions
//////////////////////////////////////////////////////////////////////////////

static const float PI = static_cast<float>(M_PI);
static const float SQRT2 = static_cast<float>(M_SQRT2);

struct NegativeLogDataTermGrd
{
	NegativeLogDataTermGrd(float dmax, float dmin, float sigma, float pOut, float pInv, const std::vector<float>& groundDisparity)
	{
		init(dmax, dmin, sigma, pOut, pInv, groundDisparity);
	}

	inline float operator()(float d, int v) const
	{
		if (d < 0.f)
			return 0.f;

		return std::min(nLogPUniform_, nLogPGaussian_[v] + cquad_ * (d - fn_[v]) * (d - fn_[v]));
	}

	// pre-compute constant terms
	void init(float dmax, float dmin, float sigma, float pOut, float pInv, const std::vector<float>& groundDisparity)
	{
		// uniform distribution term
		nLogPUniform_ = logf(dmax - dmin) - logf(pOut);

		// Gaussian distribution term
		const int h = static_cast<int>(groundDisparity.size());
		nLogPGaussian_.resize(h);
		fn_.resize(h);
		for (int v = 0; v < h; v++)
		{
			const float fn = groundDisparity[v];
			const float ANorm = 0.5f * (erff((dmax - fn) / (SQRT2 * sigma)) - erff((dmin - fn) / (SQRT2 * sigma)));
			nLogPGaussian_[v] = logf(ANorm) + logf(sigma * sqrtf(2.f * PI)) - logf(1.f - pOut);
			fn_[v] = fn;
		}

		// coefficient of quadratic part
		cquad_ = 1.f / (2.f * sigma * sigma);
	}

	float nLogPUniform_, cquad_;
	std::vector<float> nLogPGaussian_, fn_;
};

struct NegativeLogDataTermObj
{
	NegativeLogDataTermObj(float dmax, float dmin, float sigma, float pOut, float pInv)
	{
		init(dmax, dmin, sigma, pOut, pInv);
	}

	inline float operator()(float d, int fn) const
	{
		if (d < 0.f)
			return 0.f;

		return std::min(nLogPUniform_, nLogPGaussian_[fn] + cquad_ * (d - fn) * (d - fn));
	}

	// pre-compute constant terms
	void init(float dmax, float dmin, float sigma, float pOut, float pInv)
	{
		// uniform distribution term
		nLogPUniform_ = logf(dmax - dmin) - logf(pOut);

		// Gaussian distribution term
		const int fnmax = static_cast<int>(dmax);
		nLogPGaussian_.resize(fnmax);
		for (int fn = 0; fn < fnmax; fn++)
		{
			const float ANorm = 0.5f * (erff((dmax - fn) / (SQRT2 * sigma)) - erff((dmin - fn) / (SQRT2 * sigma)));
			nLogPGaussian_[fn] = logf(ANorm) + logf(sigma * sqrtf(2.f * PI)) - logf(1.f - pOut);
		}

		// coefficient of quadratic part
		cquad_ = 1.f / (2.f * sigma * sigma);
	}

	float nLogPUniform_, cquad_;
	std::vector<float> nLogPGaussian_;
};

struct NegativeLogDataTermSky
{
	NegativeLogDataTermSky(float dmax, float dmin, float sigma, float pOut, float pInv, float fn = 0.f) : fn_(fn)
	{
		init(dmax, dmin, sigma, pOut, pInv, fn);
	}

	inline float operator()(float d) const
	{
		if (d < 0.f)
			return 0.f;

		return std::min(nLogPUniform_, nLogPGaussian_ + cquad_ * (d - fn_) * (d - fn_));
	}

	// pre-compute constant terms
	void init(float dmax, float dmin, float sigma, float pOut, float pInv, float fn)
	{
		// uniform distribution term
		nLogPUniform_ = logf(dmax - dmin) - logf(pOut);

		// Gaussian distribution term
		const float ANorm = 0.5f * (erff((dmax - fn) / (SQRT2 * sigma)) - erff((dmin - fn) / (SQRT2 * sigma)));
		nLogPGaussian_ = logf(ANorm) + logf(sigma * sqrtf(2.f * PI)) - logf(1.f - pOut);

		// coefficient of quadratic part
		cquad_ = 1.f / (2.f * sigma * sigma);
	}

	float nLogPUniform_, cquad_, nLogPGaussian_, fn_;
};

//////////////////////////////////////////////////////////////////////////////
// prior cost functions
//////////////////////////////////////////////////////////////////////////////

static const float N_LOG_0_3 = -static_cast<float>(log(0.3));
static const float N_LOG_0_5 = -static_cast<float>(log(0.5));
static const float N_LOG_0_7 = -static_cast<float>(log(0.7));
static const float N_LOG_0_0 = std::numeric_limits<float>::infinity();
static const float N_LOG_1_0 = 0.f;

struct NegativeLogPriorTerm
{
	static const int G = 0;
	static const int O = 1;
	static const int S = 2;

	NegativeLogPriorTerm(int h, float vhor, float dmax, float dmin, float b, float fu, float deltaz, float eps,
		float pOrd, float pGrav, float pBlg, const std::vector<float>& groundDisparity)
	{
		init(h, vhor, dmax, dmin, b, fu, deltaz, eps, pOrd, pGrav, pBlg, groundDisparity);
	}

	inline float getO0(int vT) const
	{
		return costs0_(vT, O);
	}
	inline float getG0(int vT) const
	{
		return costs0_(vT, G);
	}
	inline float getS0(int vT) const
	{
		return N_LOG_0_0;
	}

	inline float getOO(int vB, int d1, int d2) const
	{
		return costs1_(vB, O, O) + costs2_O_O_(d2, d1);
	}
	inline float getOG(int vB, int d1, int d2) const
	{
		return costs1_(vB, O, G) + costs2_O_G_(vB - 1, d1);
	}
	inline float getOS(int vB, int d1, int d2) const
	{
		return costs1_(vB, O, S) + costs2_O_S_(d1);
	}

	inline float getGO(int vB, int d1, int d2) const
	{
		return costs1_(vB, G, O);
	}
	inline float getGG(int vB, int d1, int d2) const
	{
		return costs1_(vB, G, G);
	}
	inline float getGS(int vB, int d1, int d2) const
	{
		return N_LOG_0_0;
	}

	inline float getSO(int vB, int d1, int d2) const
	{
		return costs1_(vB, S, O) + costs2_S_O_(d2, d1);
	}
	inline float getSG(int vB, int d1, int d2) const
	{
		return N_LOG_0_0;
	}
	inline float getSS(int vB, int d1, int d2) const
	{
		return N_LOG_0_0;
	}

	void init(int h, float vhor, float dmax, float dmin, float b, float fu, float deltaz, float eps,
		float pOrd, float pGrav, float pBlg, const std::vector<float>& groundDisparity)
	{
		const int fnmax = static_cast<int>(dmax);

		costs0_.create(h, 2);
		costs1_.create(h, 3, 3);
		costs2_O_O_.create(fnmax, fnmax);
		costs2_O_S_.create(1, fnmax);
		costs2_O_G_.create(h, fnmax);
		costs2_S_O_.create(fnmax, fnmax);

		for (int vT = 0; vT < h; vT++)
		{
			const float P1 = N_LOG_1_0;
			const float P2 = -logf(1.f / h);
			const float P3_O = vT > vhor ? N_LOG_1_0 : N_LOG_0_5;
			const float P3_G = vT > vhor ? N_LOG_0_0 : N_LOG_0_5;
			const float P4_O = -logf(1.f / (dmax - dmin));
			const float P4_G = N_LOG_1_0;

			costs0_(vT, O) = P1 + P2 + P3_O + P4_O;
			costs0_(vT, G) = P1 + P2 + P3_G + P4_G;
		}

		for (int vB = 0; vB < h; vB++)
		{
			const float P1 = N_LOG_1_0;
			const float P2 = -logf(1.f / (h - vB));

			const float P3_O_O = vB - 1 < vhor ? N_LOG_0_7 : N_LOG_0_5;
			const float P3_G_O = vB - 1 < vhor ? N_LOG_0_3 : N_LOG_0_0;
			const float P3_S_O = vB - 1 < vhor ? N_LOG_0_0 : N_LOG_0_5;

			const float P3_O_G = vB - 1 < vhor ? N_LOG_0_7 : N_LOG_0_0;
			const float P3_G_G = vB - 1 < vhor ? N_LOG_0_3 : N_LOG_0_0;
			const float P3_S_G = vB - 1 < vhor ? N_LOG_0_0 : N_LOG_0_0;

			const float P3_O_S = vB - 1 < vhor ? N_LOG_0_0 : N_LOG_1_0;
			const float P3_G_S = vB - 1 < vhor ? N_LOG_0_0 : N_LOG_0_0;
			const float P3_S_S = vB - 1 < vhor ? N_LOG_0_0 : N_LOG_0_0;

			costs1_(vB, O, O) = P1 + P2 + P3_O_O;
			costs1_(vB, G, O) = P1 + P2 + P3_G_O;
			costs1_(vB, S, O) = P1 + P2 + P3_S_O;

			costs1_(vB, O, G) = P1 + P2 + P3_O_G;
			costs1_(vB, G, G) = P1 + P2 + P3_G_G;
			costs1_(vB, S, G) = P1 + P2 + P3_S_G;

			costs1_(vB, O, S) = P1 + P2 + P3_O_S;
			costs1_(vB, G, S) = P1 + P2 + P3_G_S;
			costs1_(vB, S, S) = P1 + P2 + P3_S_S;
		}

		for (int d1 = 0; d1 < fnmax; d1++)
			costs2_O_O_(0, d1) = N_LOG_0_0;

		for (int d2 = 1; d2 < fnmax; d2++)
		{
			const float z = b * fu / d2;
			const float deltad = d2 - b * fu / (z + deltaz);
			for (int d1 = 0; d1 < fnmax; d1++)
			{
				if (d1 > d2 + deltad)
					costs2_O_O_(d2, d1) = -logf(pOrd / (d2 - deltad));
				else if (d1 <= d2 + deltad)
					costs2_O_O_(d2, d1) = -logf((1.f - pOrd) / (dmax - d2 - deltad));
				else
					costs2_O_O_(d2, d1) = N_LOG_0_0;
			}
		}

		for (int v = 0; v < h; v++)
		{
			const float fn = groundDisparity[v];
			for (int d1 = 0; d1 < fnmax; d1++)
			{
				if (d1 > fn + eps)
					costs2_O_G_(v, d1) = -logf(pGrav / (dmax - fn - eps));
				else if (d1 > fn + eps)
					costs2_O_G_(v, d1) = -logf(pBlg / (fn - eps - dmin));
				else
					costs2_O_G_(v, d1) = -logf((1.f - pGrav - pBlg) / (2.f * eps));
			}
		}

		for (int d1 = 0; d1 < fnmax; d1++)
		{
			costs2_O_S_(d1) = d1 > eps ? -logf((1.f / (dmax - dmin)) / (1.f - eps / (dmax - dmin))) : N_LOG_0_0;
		}

		for (int d2 = 0; d2 < fnmax; d2++)
		{
			for (int d1 = 0; d1 < fnmax; d1++)
			{
				if (d2 < eps)
					costs2_S_O_(d2, d1) = N_LOG_0_0;
				else if (d1 <= 0)
					costs2_S_O_(d2, d1) = N_LOG_1_0;
				else
					costs2_S_O_(d2, d1) = N_LOG_0_0;
			}
		}
	}

	Matrixf costs0_, costs1_;
	Matrixf costs2_O_O_, costs2_O_G_, costs2_O_S_, costs2_S_O_;
};

#endif // !__COST_FUNCTION_H__
