#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <string_view>

#include <opencv2/videoio.hpp>
#include <opencv2/tracking.hpp>

#include "fastTracker.hpp"
#include "fastTrackerMP.hpp"
#include "fastTrackerCUDA.hpp"
#include "fastTrackerMPCUDA.hpp"

#include "Tracy.hpp"


std::vector<int> getBoundingBox(std::string file){
    std::ifstream truth(file);
    std::string data;
    if (truth.is_open()) { 
        truth >> data; 
    }

    std::vector<int> boundingBox;
    std::string delimiter = ",";
    size_t pos = 0;
    std::string token;
    while ((pos = data.find(delimiter)) != std::string::npos) {
        token = data.substr(0, pos);
        boundingBox.push_back(std::stoi(token));
        data.erase(0, pos + delimiter.length());
    }
    boundingBox.push_back(std::stoi(data));
    return boundingBox;
}

#define TIME_NOW std::chrono::high_resolution_clock::now()

template <typename T>
std::pair<bool, cv::Rect> updateTracker(std::string_view name, T& tracker, std::ostream& output, int frame, const cv::Mat& img, cv::Rect gtBox) {
    bool success = true;
    cv::Rect resultBox = gtBox;
    if (frame == 0) {
        auto start = TIME_NOW;
        tracker->init(img, gtBox);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_NOW - start).count();
        output << name << "_INIT: " << duration << "\n";
    } else {
        auto start = TIME_NOW;
        success = tracker->update(img, resultBox);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_NOW - start).count();
        output << name << "_UPDATE: " << duration << "\n";
    }

    return std::make_pair(success, resultBox);
}

#undef TIME_NOW

void printCVRect(std::ostream& output, const cv::Rect& rect) {
    output << rect.x << "," << rect.y << "," << rect.width << "," << rect.height << "\n";
}

bool checkBoxesEqual(std::string name, const cv::Rect& a, const cv::Rect& b, int frame) {
    bool equal = a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
    if (!equal) {
        std::cout << "  Bounding boxes are not equal for frame " << frame << std::endl;
        std::cout << "  STD: ";
        printCVRect(std::cout, a);
        std::cout << "  " << name << ": ";
        printCVRect(std::cout, b);
    }
    return equal;
}

int main() {
    // create a filesystem instance
    std::string path = "test/";
    std::vector<std::filesystem::path> files;
    std::cout << "Loading all files from " << path << std::endl;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(path)) {
        // if the entry is a file and the extension is not .avi then do nothing
        if (entry.path().extension() != ".avi") {
            continue;
        }
        files.push_back(entry.path());
    }
    std::cout << "Loaded " << files.size() << " files" << std::endl;
    std::cout << "Sorting all files" << std::endl;
    std::sort(files.begin(), files.end());
    std::cout << "Sorted all " << files.size() << " files" << std::endl;

    std::cout << "Processing all files" << std::endl;
    // read every file within each subdirectory
    for (const auto& path : files) {
        std::cout << " Processing: " << path << std::endl;

        // create a file to store the results
        std::ofstream output_file;
        output_file.open("data/" + path.stem().string() + ".txt");

        auto t = cv::TrackerKCF::create();
        auto tracker = cv::FastTracker::create();
        auto tracker_mp = cv::FastTrackerMP::create();
        auto tracker_cuda = cv::FastTrackerCUDA::create();
        auto tracker_mp_cuda = cv::FastTrackerMPCUDA::create();

        // find all videos
        auto capture = cv::VideoCapture(path.string(), cv::CAP_ANY);
        std::cout << "  Capture has been opened: " << capture.isOpened() << "\n";
        std::cout << "  Backend=" << capture.getBackendName() << "\n";
        int frame = 0;
        while (true) {
            capture.grab();
            cv::Mat image;
            capture.retrieve(image);
            if (image.empty())
                break;

            if (frame == 0) {
                // read a cv::Rect from a text file with the format x,y,w,h
                // trim off the file and get the directory
                auto dir = path.parent_path().string();
                auto groundTruthFile = dir + "/groundtruth.txt";
               
                std::vector<int> boundingBox = getBoundingBox(groundTruthFile);

                int x = boundingBox[0];
                int y = boundingBox[1];
                int w = boundingBox[2];
                int h = boundingBox[3];
                int boxSize = w * h;

                output_file << "BBOX_SIZE:" << boxSize << "\n";

                cv::Rect rect(x, y, w, h);
                
                t->init(image, rect);

                updateTracker("BASE", tracker, output_file, frame, image, rect);
                updateTracker("MP", tracker_mp, output_file, frame, image, rect);
                //updateTracker("CUDA", tracker_cuda, output_file, frame, image, rect);
                //updateTracker("MP_CUDA", tracker_mp_cuda, output_file, frame, image, rect);
            } else {
                cv::Rect dummy{ 0,0,0,0 };

                bool t_s = t->update(image, dummy);

                auto [success, bb] = updateTracker("BASE", tracker, output_file, frame, image, dummy);
                auto [success_bb, bb_mp] = updateTracker("MP", tracker_mp, output_file, frame, image, dummy);
                //auto [success_cuda, bb_cuda] = updateTracker("CUDA", tracker_cuda, output_file, frame, image, dummy);
                //auto [success_bb_cuda, bb_mp_cuda] = updateTracker("MP_CUDA", tracker_mp_cuda, output_file, frame, image, dummy);
                
                if (!t_s) {
                    break;
                }

                // compare the bounding boxes
                if (!checkBoxesEqual("MP", bb, bb_mp, frame)) {
                    break;
                }
                // if (!checkBoxesEqual("CUDA", bb, bb_cuda, frame)) {
                //   break;
                // }
                //if (!checkBoxesEqual("MP_CUDA", bb, bb_mp_cuda, frame)) {
                //    break;
                //}

            }

            ++frame;
        }

        FrameMark;

        std::cout << "  Tracked: # " << frame << " frames\n";
        output_file.close();
    
    }
    return 0;
}
