/**
 *
 *  \author     Paul Bovbel <pbovbel@clearpathrobotics.com>
 *  \copyright  Copyright (c) 2014, Clearpath Robotics, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Clearpath Robotics, Inc. nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CLEARPATH ROBOTICS, INC. BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Please send comments, questions, or patches to code@clearpathrobotics.com
 *
 */
#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>
#include <string>
#include <vector>

#include "zbar_ros/barcode_reader_node.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

namespace {

constexpr int kMinScanSide = 320;
constexpr double kRoiPadRatio = 0.1;

static inline cv::Rect clampRect(const cv::Rect& r, const cv::Size& sz) {
    return r & cv::Rect(0, 0, sz.width, sz.height);
}

// Pass detector and reusable vector to avoid allocations per-frame
static cv::Rect detectQrRoi(const cv::Mat& gray, cv::QRCodeDetector& det, std::vector<cv::Point>& points) {
    if (gray.empty() || gray.cols <= 0 || gray.rows <= 0) {
        return cv::Rect();
    }

    try {
        points.clear();
        if (!det.detect(gray, points) || points.size() < 4) {
            return cv::Rect(0, 0, gray.cols, gray.rows);
        }

        cv::Rect bbox = cv::boundingRect(points);
        if (bbox.width <= 0 || bbox.height <= 0) {
            return cv::Rect(0, 0, gray.cols, gray.rows);
        }

        const int pad = static_cast<int>(kRoiPadRatio * std::max(bbox.width, bbox.height));
        cv::Rect expanded(bbox.x - pad, bbox.y - pad, bbox.width + 2 * pad, bbox.height + 2 * pad);
        expanded = clampRect(expanded, gray.size());
        if (expanded.width <= 0 || expanded.height <= 0) {
            return cv::Rect(0, 0, gray.cols, gray.rows);
        }
        return expanded;
    } catch (const cv::Exception&) {
        // OpenCV's QR detector may throw internally (e.g. convexHull assertion).
        // Treat it as "no ROI" and keep the process alive.
        return cv::Rect(0, 0, gray.cols, gray.rows);
    }
}

// Reuse output Mat to avoid allocation when resize is needed
static void upscaleIfSmall(const cv::Mat& gray, cv::Mat& out, int min_side = kMinScanSide) {
    const int s = std::min(gray.cols, gray.rows);
    if (s >= min_side) {
        out = gray;
    } else {
        const double scale = static_cast<double>(min_side) / static_cast<double>(std::max(1, s));
        cv::resize(gray, out, cv::Size(), scale, scale, cv::INTER_NEAREST);
    }
}

static void publishCroppedImage(const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& publisher,
                                const sensor_msgs::msg::Image& source_msg, const cv::Mat& cropped) {
    if (!publisher || publisher->get_subscription_count() == 0) {
        return;
    }

    sensor_msgs::msg::Image crop_msg;
    cv_bridge::CvImage crop_cv;
    crop_cv.header = source_msg.header;
    crop_cv.encoding = "mono8";
    crop_cv.image = cropped;
    crop_cv.toImageMsg(crop_msg);
    publisher->publish(crop_msg);
}

} // namespace

using namespace std::chrono_literals;

namespace zbar_ros {

BarcodeReaderNode::BarcodeReaderNode() : Node("barcode_reader_node") {
    scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);

    image_topic_ = this->declare_parameter<std::string>("image_topic", "camera/image/mono8");
    qr_code_topic_ = this->declare_parameter<std::string>("qr_code_topic", "/barcode/code_string");
    qr_image_topic_ = this->declare_parameter<std::string>("qr_image_topic", "/barcode/image");
    RCLCPP_DEBUG(get_logger(), "Subscribing to topics: %s, %s", image_topic_.c_str(), qr_code_topic_.c_str());

    rclcpp::QoS qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
    rclcpp::QoS qr_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(image_topic_, qos,
                                                                     std::bind(&BarcodeReaderNode::imageCb, this, std::placeholders::_1));

    symbol_pub_ = this->create_publisher<zbar_ros_interfaces::msg::Symbol>("symbol", 10);
    barcode_pub_ = this->create_publisher<std_msgs::msg::String>(qr_code_topic_, qr_qos);
    barcode_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(qr_image_topic_, qos);

    throttle_ = this->declare_parameter<double>("throttle_repeated_barcodes", 0.0);
    RCLCPP_DEBUG(get_logger(), "throttle_repeated_barcodes : %f", throttle_);

    if (throttle_ > 0.0) {
        clean_timer_ = this->create_wall_timer(10s, std::bind(&BarcodeReaderNode::cleanCb, this));
    }
}

void BarcodeReaderNode::imageCb(sensor_msgs::msg::Image::ConstSharedPtr image) {
    RCLCPP_DEBUG(get_logger(), "Image received on subscribed topic");

    try {
        cv_bridge::CvImageConstPtr cv_image = cv_bridge::toCvShare(image, "mono8");

        const cv::Mat& gray = cv_image->image;
        if (gray.empty() || gray.cols <= 0 || gray.rows <= 0) {
            return;
        }

        const cv::Rect roi = detectQrRoi(gray, qr_detector_, qr_points_);
        if (roi.width <= 0 || roi.height <= 0) {
            return;
        }

        cv::Mat cropped = gray(roi);
        if (cropped.empty()) {
            return;
        }

        publishCroppedImage(barcode_image_pub_, *image, cropped);

        upscaleIfSmall(cropped, scan_img_, kMinScanSide);
        if (!scan_img_.isContinuous()) {
            scan_img_ = scan_img_.clone();
        }
        cv::Mat& scan_img = scan_img_;

        zbar::Image zimg(scan_img.cols, scan_img.rows, "Y800", scan_img.data, scan_img.cols * scan_img.rows);
        scanner_.scan(zimg);

        auto it_start = zimg.symbol_begin();
        auto it_end = zimg.symbol_end();
        if (it_start != it_end) {
            for (zbar::Image::SymbolIterator symbol_it = it_start; symbol_it != it_end; ++symbol_it) {
                zbar_ros_interfaces::msg::Symbol symbol;
                symbol.data = symbol_it->get_data();
                RCLCPP_DEBUG(get_logger(), "Barcode detected with data: '%s'", symbol.data.c_str());

                if (!shouldPublishBarcode(symbol.data)) {
                    continue;
                }

                symbol_pub_->publish(symbol);

                std_msgs::msg::String barcode_string;
                barcode_string.data = symbol.data;
                barcode_pub_->publish(barcode_string);
                barcode_string.data = "unknown"; // clear for next use
            }
        } else {
            RCLCPP_DEBUG(get_logger(), "No barcode detected in image");
        }

        warnDeprecatedBarcodeTopicOnce();

        zimg.set_data(NULL, 0);
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "OpenCV exception in barcode_reader_node imageCb: %s", e.what());
        return;
    } catch (const std::exception& e) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Exception in barcode_reader_node imageCb: %s", e.what());
        return;
    } catch (...) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Unknown exception in barcode_reader_node imageCb");
        return;
    }
}

bool BarcodeReaderNode::shouldPublishBarcode(const std::string& barcode) {
    if (throttle_ <= 0.0) {
        return true;
    }

    const std::lock_guard<std::mutex> lock(memory_mutex_);
    const rclcpp::Time now_time = now();
    auto it = barcode_memory_.find(barcode);
    if (it != barcode_memory_.end()) {
        if (now_time > it->second) {
            barcode_memory_.erase(it);
        } else {
            return false;
        }
    }

    barcode_memory_[barcode] = now_time + rclcpp::Duration(std::chrono::duration<double>(throttle_));
    return true;
}

void BarcodeReaderNode::warnDeprecatedBarcodeTopicOnce() {
    static bool alreadyWarnedDeprecation = false;
    if (!alreadyWarnedDeprecation && count_subscribers("barcode") > 0) {
        alreadyWarnedDeprecation = true;
        RCLCPP_WARN(get_logger(), "A subscription was detected on the deprecated topic 'barcode'. Please update the node "
                                  "that is subscribing to use the new topic 'symbol' with type "
                                  "'zbar_ros_interfaces::msg::Symbol' instead. The 'barcode' topic will be removed "
                                  "in the next distribution.");
    }
}

void BarcodeReaderNode::cleanCb() {
    const std::lock_guard<std::mutex> lock(memory_mutex_);
    const rclcpp::Time now_time = now();
    auto it = barcode_memory_.begin();
    while (it != barcode_memory_.end()) {
        if (now_time > it->second) {
            RCLCPP_DEBUG(get_logger(), "Cleaned %s from memory", it->first.c_str());
            it = barcode_memory_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace zbar_ros
