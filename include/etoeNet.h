// #ifdef __ETOE_NET__
// #define __ETOE_NET__

#include "tensorNet.h"
#include "opencv2/opencv.hpp"
#include <unistd.h>

class etoeNet : tensorNet
{
    public:
        etoeNet();
        ~etoeNet();

        void loadOnnxFile(const std::string &onnx_file_path);
        //float runInference(const cv::Mat &img_mat, int* fd,  float* actualAngle);
        void runInference(const cv::Mat &img_mat, int* fd);
        // float getActualAngle();
        // float getActualAngle_ver2();
        // float getModelOutput();

        float m_network_output = 0.0;

    private:
        cv::Mat m_img_cropped_rgb_f_mat;
        float currentAngle = 0.0;
        float actualAngle = 0.0;
        int currentVel = 2.0;
        int actualVel=0.0;

};


// #endif