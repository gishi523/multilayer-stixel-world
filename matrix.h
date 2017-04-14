#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <opencv2/opencv.hpp>
#include <array>

template <typename T>
class Matrix : public cv::Mat_<T>
{
public:
	Matrix() : cv::Mat_<T>() {}
	Matrix(int size1, int size2) { create(size1, size2); }
	Matrix(int size1, int size2, int size3) { create(size1, size2, size3); }

	void create(int size1, int size2)
	{
		cv::Mat_<T>::create(size1, size2);
		this->size1 = size1;
		this->size2 = size2;
		this->size3 = 1;
	}

	void create(int size1, int size2, int size3)
	{
		cv::Mat_<T>::create(3, std::array<int, 3>{size1, size2, size3}.data());
		this->size1 = size1;
		this->size2 = size2;
		this->size3 = size3;
	}

	inline T& operator()(int i)
	{
		return *((T*)this->data + i);
	}
	inline const T& operator()(int i) const
	{
		return *((T*)this->data + i);
	}
	inline T& operator()(int i, int j)
	{
		return *((T*)this->data + i * size2 + j);
	}
	inline const T& operator()(int i, int j) const
	{
		return *((T*)this->data + i * size2 + j);
	}
	inline T& operator()(int i, int j, int k)
	{
		return *((T*)this->data + size3 * (i * size2 + j) + k);
	}
	inline const T& operator()(int i, int j, int k) const
	{
		return *((T*)this->data + size3 * (i * size2 + j) + k);
	}
	int size1, size2, size3;
};

using Matrixf = Matrix<float>;
using Matrixi = Matrix<int>;

#endif // !__MATRIX_H__
