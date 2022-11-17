#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#include <opencv2/videoio.hpp>
#include <opencv2/tracking.hpp>

#include "fastTracker.hpp"


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

        auto tracker = cv::TrackerKCF::create();
        auto fast_tracker = cv::FastTrackerKCF::create(); // TODO ANDREW FIX THIS

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

                cv::Rect rect1(x, y, w, h);
                cv::Rect rect2(x, y, w, h);

                auto start = std::chrono::high_resolution_clock::now();
                tracker->init(image, rect1);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                output_file << "STD_INIT_TIME:" << duration.count() << "\n";

                start = std::chrono::high_resolution_clock::now();
                fast_tracker->init(image, rect2);
                stop = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                output_file << "FAST_INIT_TIME:" << duration.count() << "\n";

            } else {
                cv::Rect bb;
                cv::Rect bb_fast;

                auto start = std::chrono::high_resolution_clock::now();
                bool std_success = tracker->update(image, bb);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                output_file << "STD_UPDATE_TIME: " << duration.count() << "\n";

                start = std::chrono::high_resolution_clock::now();
                bool fast_success = fast_tracker->update(image, bb_fast);
                stop = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                output_file << "FAST_UPDATE_TIME: " << duration.count() << "\n";

                if (!std_success) {
                    break;
                }

                // TODO
                // compare the bounding boxes  
                std::cout << bb.x << bb.y << bb.height << bb.width << std::endl;
                std::cout << bb_fast.x << bb_fast.y << bb_fast.height << bb_fast.width << std::endl;
                if (bb.x == bb_fast.x && bb.y == bb_fast.y && bb.height == bb_fast.height && bb.width == bb_fast.width){
                    std::cout << "success" << std::endl;
                }
                else{
                    std::cout << "BOUNDING BOXES NOT EQUAL" << std::endl;
                }
            }

            ++frame;
        }
        std::cout << "  Tracked: # " << frame << " frames\n";
        output_file.close();
    
    }
    return 0;
}
