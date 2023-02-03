#include "etoeNet.h"

etoeNet::etoeNet() : tensorNet(){

}

etoeNet::~etoeNet(){

}

void etoeNet:: loadOnnxFile(const std::string &onnx_file_path){
    std::vector<std::string> input_blocbs = {"input"};
    std::vector<std::string> output_blobs = {"output"};
    LoadNetwork(NULL, onnx_file_path.c_str(), NULL, input_blocbs, output_blobs, 1, TYPE_FP32);

    m_img_cropped_rgb_f_mat = cv::Mat(cv::Size(320, 70), CV_32FC3, mInputs[0].CPU);
}

void etoeNet::runInference(const cv::Mat &img_mat,int* fd){
    std::cout << "\n=======" << std::endl;
    char go[1]= {'w'};
    char left[1] = {'a'};
    char right[1] = {'d'};
    char trash[1] = {'b'};
    char stop[1] = {'s'};
    char back[1] = {'x'};
    char clear[1] = {0x0D};

    //전처리
    //-------------------------------------------
    cv::Mat img_resized_mat;
    cv::resize(img_mat, img_resized_mat, cv::Size(320,160));    //resize the original image to 320x160
    //cv::imshow("test", img_resized_mat);
    cv::Mat img_cropped_mat = img_resized_mat(cv::Rect(0, 35, 320, 70));    //crop resized image (0, 65, 320, 70)
    cv::Mat img_cropped_rgb_mat;
    cv::cvtColor(img_cropped_mat, img_cropped_rgb_mat, cv::COLOR_BGR2RGB);
    img_cropped_rgb_mat.convertTo(m_img_cropped_rgb_f_mat, CV_32FC3, (1.0 / 127.5), -1.0);
    //--------------------------------------------

    ProcessNetwork(true); //sync = TRUE

    // TODO : inference 결과 가져오고난 뒤 Postprocess

    // std::cout <<"run"<<std::endl;

    //todo :inference resul
    float network_output_angle = *(mOutputs[0].CPU); //angle prediction
    float network_output_velocity =-*(mOutputs[0].CPU+1); //velocity prediction

    std::cout << "current angle : " << currentAngle << std::endl;
    std::cout << "current velocity : " << currentVel << std::endl;

    std::cout << "network_output_angle  : " << network_output_angle << std::endl;
    std::cout << "network_output_velocity : " << network_output_velocity << std::endl<<std::endl;

    //angle setting
    std::cout << "[ANGLE]" << std::endl;
    if(network_output_angle< -0.875)
        currentAngle= -1.00;
    else if(network_output_angle< -0.625)
        currentAngle= -0.75;
    else if(network_output_angle< -0.375)
        currentAngle = -0.5;
    else if(network_output_angle< -0.125)
        currentAngle = -0.25;
    else if(network_output_angle< 0.125)
        currentAngle = 0.0;
    else if(network_output_angle< 0.375)
        currentAngle = 0.25;
    else if(network_output_angle< 0.625)
        currentAngle= 0.5;
    else if(network_output_angle< 0.875)
        currentAngle = 0.75;    
    else
        currentAngle = 1.0;
    currentAngle *= 20;

    int diffAngle = (int)((currentAngle - actualAngle)/5);
    std::cout << "diffAngle  : " << diffAngle << std::endl;

    if(diffAngle == 0){
        std::cout << "Go straight" << std::endl;
    }
    else if(diffAngle < 0){
        for(int i=0; i< -diffAngle; i++){
            // std::cout << "write: d (" << i+1 << ")" << std::endl;
            std::cout << "Turn right" << std::endl;
            write(*fd, right, 1);
            // write(SLAM.fd, clear, 1);
        }
    }
    else{
        for(int i=0; i< diffAngle; i++){
            // std::cout << "write: a (" << i+1 << ")" << std::endl;
            std::cout << "Turn left" << std::endl;
            write(*fd, left, 1);
            // write(SLAM.fd, clear, 1);
        }
    }

    //velocity setting
    std::cout << "\n[VELOCITY]" << std::endl;
    if(network_output_velocity >= 1.5) { //velocity set to 2
        if(currentVel == 1) {
            std::cout << "Increase speed by 1" << std::endl;
            write(*fd, go, 1);
            // write(SLAM.fd, clear, 1);
        }
        else if(currentVel == 0) {
            std::cout << "Increase speed by 2" << std::endl;
            write(*fd, go, 1);
            // write(SLAM.fd, clear, 1);
            write(*fd, go, 1);
            // write(SLAM.fd, clear, 1);
        }
        currentVel = 2;
    }
    else if(network_output_velocity >= 0.5) { //velocity set to 1
        if(currentVel == 2) {
            std::cout << "Decrease speed by 1" << std::endl;
            write(*fd, back, 1);
            // write(SLAM.fd, clear, 1);
        }
        else if(currentVel == 0) {
            std::cout << "Increase speed by 1" << std::endl;
            write(*fd, go, 1);
            // write(SLAM.fd, clear, 1);
        }
        currentVel = 1;
    }
    else { //velocity set to 0
        std::cout << "Speed set to 0" << std::endl;
        write(*fd, stop, 1);
        // write(SLAM.fd, clear, 1);
        currentVel = 0;
    }

}

// float etoeNet::getModelOutput(){
    
//     return m_network_output;
// }

// float etoeNet::getActualAngle(){

//     float actual_angle;
//     if(network_output_angle< -0.75)
//         actual_angle = -1.00;
//     else if(network_output_angle< -0.25)
//         actual_angle = -0.5;
//     else if(network_output_angle< 0.25)
//         actual_angle = 0.0;
//     else if(network_output_angle< 0.75)
//         actual_angle = 0.5;
//     else
//         actual_angle = 1.0;
//     return actual_angle * 20;
// }

// float etoeNet::getActualAngle_ver2(){

//     float actual_angle;
//     if(network_output_angle< -0.875)
//         actual_angle = -1.00;
//     else if(network_output_angle< -0.625)
//         actual_angle = -0.75;
//     else if(network_output_angle< -0.375)
//         actual_angle = -0.5;
//     else if(network_output_angle< -0.125)
//         actual_angle = -0.25;
//     else if(network_output_angle< 0.125)
//         actual_angle = 0.0;
//     else if(network_output_angle< 0.375)
//         actual_angle = 0.25;
//     else if(network_output_angle< 0.625)
//         actual_angle = 0.5;
//     else if(network_output_angle< 0.875)
//         actual_angle = 0.75;    
//     else
//         actual_angle = 1.0;
//     return actual_angle * 20;
// }
