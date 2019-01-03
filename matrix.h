#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <opencv2/opencv.hpp>

template <typename T>
class Matrix : public cv::Mat_<T>
{
public:
	Matrix() : cv::Mat_<T>() {}
	Matrix(int size1) { create(size1); }
	Matrix(int size1, int size2) { create(size1, size2); }
	Matrix(int size1, int size2, int size3) { create(size1, size2, size3); }

	// 1D Matrix
	void create(int size1)
	{
		cv::Mat_<T>::create(size1, 1);

		this->size1 = size1;
		this->size2 = 0;
		this->size3 = 0;

		this->step1 = 0;
		this->step2 = 0;
	}

	// 2D Matrix
	void create(int size1, int size2)
	{
		cv::Mat_<T>::create(size1, size2);

		this->size1 = size1;
		this->size2 = size2;
		this->size3 = 0;

		this->step1 = size2;
		this->step2 = 0;
	}

	// 3D Matrix
	void create(int size1, int size2, int size3)
	{
		const int sizes[3] = { size1, size2, size3 };
		cv::Mat_<T>::create(3, sizes);

		this->size1 = size1;
		this->size2 = size2;
		this->size3 = size3;

		this->step1 = size2 * size3;
		this->step2 = size3;
	}

	// 1D Matrix
	inline T& operator()(int i)
	{
		return *((T*)this->data + i);
	}
	inline const T& operator()(int i) const
	{
		return *((T*)this->data + i);
	}

	// 2D Matrix
	inline T& operator()(int i, int j)
	{
		return *((T*)this->data + i * step1 + j);
	}
	inline const T& operator()(int i, int j) const
	{
		return *((T*)this->data + i * step1 + j);
	}

	// 3D Matrix
	inline T& operator()(int i, int j, int k)
	{
		return *((T*)this->data + i * step1 + j * step2 + k);
	}
	inline const T& operator()(int i, int j, int k) const
	{
		return *((T*)this->data + i * step1 + j * step2 + k);
	}

	int size1, size2, size3;
	int step1, step2;
};

using Matrixf = Matrix<float>;
using Matrixi = Matrix<int>;

#endif // !__MATRIX_H__
