/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

// #include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "System.h"

#include "ViewerAR.h"

using namespace std;

ORB_SLAM3::ViewerAR viewerAR;
bool bRGB = true;

cv::Mat K;
cv::Mat DistCoef;

float fx = 458.654;
float fy = 457.296;
float cx = 367.215;
float cy = 248.375;

float k1 = -0.28340811;
float k2 = 0.07395907;
float p1 = 0.00019359;
float p2 = 1.76187114e-05;

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

int main(int argc, char **argv)
{

    K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    DistCoef = cv::Mat::zeros(4, 1, CV_32F);
    DistCoef.at<float>(0) = k1;
    DistCoef.at<float>(1) = k2;
    DistCoef.at<float>(2) = p1;
    DistCoef.at<float>(3) = p2;

    if (argc < 5)
    {
        std::cerr << std::endl
                  << "Usage: run ORB_SLAM3 Mono path_to_vocabulary path_to_settings" << endl;
        return 1;
    }

    const int num_seq = (argc - 3) / 2;
    cout << "num_seq = " << num_seq << endl;
    bool bFileName = (((argc - 3) % 2) == 1);
    string file_name;
    if (bFileName)
    {
        file_name = string(argv[argc - 1]);
        cout << "file name: " << file_name << endl;
    }

    // Load all sequences:
    int seq;
    vector<vector<string>> vstrImageFilenames;
    vector<vector<double>> vTimestampsCam;
    vector<int> nImages;

    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    nImages.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq < num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";
        LoadImages(string(argv[(2 * seq) + 3]) + "/mav0/cam0/data", string(argv[(2 * seq) + 4]), vstrImageFilenames[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl
         << "-------" << endl;
    cout.precision(17);

    int fps = 20;
    float dT = 1.f / fps;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, false);
    float imageScale = SLAM.GetImageScale();

    cout << endl
         << endl;
    cout << "-----------------------" << endl;
    cout << "Augmented Reality Demo" << endl;
    cout << "1) Translate the camera to initialize SLAM." << endl;
    cout << "2) Look at a planar region and translate the camera." << endl;
    cout << "3) Press Insert Cube to place a virtual cube in the plane. " << endl;
    cout << endl;
    cout << "You can place several cubes in different planes." << endl;
    cout << "-----------------------" << endl;
    cout << endl;

    viewerAR.SetSLAM(&SLAM);
    viewerAR.SetFPS(fps);
    viewerAR.SetCameraCalibration(fx, fy, cx, cy);
    thread tViewer = thread(&ORB_SLAM3::ViewerAR::Run, &viewerAR);

    std::cout <<"start ...."<<std::endl;
    for (seq = 0; seq < num_seq; seq++)
    {
        // cv::Mat im;
        
        int proccIm = 0;

        for (int ni = 0; ni < nImages[seq]; ni++, proccIm++)
        {
            // Read image from file
            cv::Mat im = cv::imread(vstrImageFilenames[seq][ni], cv::IMREAD_UNCHANGED); //,CV_LOAD_IMAGE_UNCHANGED);
            // cv::imshow("input", im);
            // cv::waitKey(0);
            
            double tframe = vTimestampsCam[seq][ni];

            if (imageScale != 1.f)
            {

                int width = im.cols * imageScale;
                int height = im.rows * imageScale;
                cv::resize(im, im, cv::Size(width, height));
            }

            Sophus::SE3f tmp_Tcw = SLAM.TrackMonocular(im, tframe);
            cv::Mat Tcw(cv::Size(4, 3), CV_32FC1, cv::Scalar(0));
            Tcw.at<float>(0, 0) =  tmp_Tcw.rotationMatrix()(0, 0);
            Tcw.at<float>(0, 1) =  tmp_Tcw.rotationMatrix()(0, 1);
            Tcw.at<float>(0, 2) =  tmp_Tcw.rotationMatrix()(0, 2);
            Tcw.at<float>(1, 0) =  tmp_Tcw.rotationMatrix()(0, 0);
            Tcw.at<float>(1, 1) =  tmp_Tcw.rotationMatrix()(1, 1);
            Tcw.at<float>(1, 2) =  tmp_Tcw.rotationMatrix()(1, 2);
            Tcw.at<float>(2, 0) =  tmp_Tcw.rotationMatrix()(1, 0);
            Tcw.at<float>(2, 1) =  tmp_Tcw.rotationMatrix()(2, 0);
            Tcw.at<float>(2, 2) =  tmp_Tcw.rotationMatrix()(2, 1);
            Tcw.at<float>(0, 0) =  tmp_Tcw.rotationMatrix()(2, 2);
            Tcw.at<float>(0, 3) =  tmp_Tcw.translation()(0);
            Tcw.at<float>(1, 3) =  tmp_Tcw.translation()(1);
            Tcw.at<float>(2, 3) =  tmp_Tcw.translation()(2);

            int state = SLAM.GetTrackingState();
            vector<ORB_SLAM3::MapPoint *> vMPs = SLAM.GetTrackedMapPoints();
            vector<cv::KeyPoint> vKeys = SLAM.GetTrackedKeyPointsUn();
            cv::Mat imu;
            cv::undistort(im, imu, K, DistCoef);
            // cv::imshow("imu", imu);
            // cv::waitKey(0);
            // 
            
            cv::cvtColor(imu, imu, CV_GRAY2RGB);
            if(imu.channels() != 3){
                SLAM.Shutdown();
                return -1;
            }
            if (bRGB)
                viewerAR.SetImagePose(imu, Tcw, state, vKeys, vMPs);
            else
            {
                cv::cvtColor(imu, imu, CV_RGB2BGR);
                viewerAR.SetImagePose(imu, Tcw, state, vKeys, vMPs);
            }
            // viewerAR.Run();
        }
        if (seq < num_seq - 1)
        {
            string kf_file_submap = "./SubMaps/kf_SubMap_" + std::to_string(seq) + ".txt";
            string f_file_submap = "./SubMaps/f_SubMap_" + std::to_string(seq) + ".txt";
            SLAM.SaveTrajectoryEuRoC(f_file_submap);
            SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file_submap);

            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }
    }

    // ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    // ros::shutdown();

    return 0;
}

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t * 1e-9);
        }
    }
}
