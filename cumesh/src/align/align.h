#pragma once

#include "cu_image.hpp"
#include <opencv2/core.hpp>


void align_tiff(
    CuImage<float>& elev,
    cv::Mat img);

void do_corr(CuImage<float>& imga, CuImage<float>& imgb);

