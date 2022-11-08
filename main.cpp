#include "fastTracker.hpp"
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

int main() {

    std::string path = "/path/to/directory";
    for (const auto & entry : fs::directory_iterator(path)){
        if (entry.path().extension() == ".avi")
            continue;

        std::cout << "Processing: " << entry.path() << std::endl;

        auto tracker = TrackerKCF::create();

        // find all videos
        auto capture = cv::VideoCapture(entry.path().string());
        int frame = 0;
        while (true) {
            capture.grab();
            auto [err, img] = capture.retrieve();
            if (!err) {
                break;
            }

            if (frame == 0) {
                // prompt user to select target on the image
                auto bb = cv::selectROI("Target selection", img);
                tracker->init(img, bb);
            } else {
                cv::Rect bb;
                bool success = tracker->update(img, bb);
                if (!success) {
                    std::cout << "Failed to update tracking for frame #" << frame << "\n";
                }
            }

            ++frame;
        }
    
    }
    return 0;
}
