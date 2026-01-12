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
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include "zbar_ros/barcode_reader_node.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

using namespace std::chrono_literals;

namespace zbar_ros {

BarcodeReaderNode::BarcodeReaderNode() : Node("barcode_reader_node") {
    scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);

    image_topic_ = this->declare_parameter<std::string>("image_topic", "camera/image/compressed");
    qr_code_topic_ = this->declare_parameter<std::string>("qr_code_topic", "/barcode");
    RCLCPP_DEBUG(get_logger(), "Subscribing to topics: %s, %s", image_topic_.c_str(), qr_code_topic_.c_str());

    camera_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        image_topic_, 10, std::bind(&BarcodeReaderNode::imageCb, this, std::placeholders::_1));

    symbol_pub_ = this->create_publisher<zbar_ros_interfaces::msg::Symbol>("symbol", 10);
    barcode_pub_ = this->create_publisher<std_msgs::msg::String>(qr_code_topic_, 10);

    throttle_ = this->declare_parameter<double>("throttle_repeated_barcodes", 0.0);
    RCLCPP_DEBUG(get_logger(), "throttle_repeated_barcodes : %f", throttle_);

    if (throttle_ > 0.0) {
        clean_timer_ = this->create_wall_timer(10s, std::bind(&BarcodeReaderNode::cleanCb, this));
    }
}

void BarcodeReaderNode::imageCb(sensor_msgs::msg::CompressedImage::ConstSharedPtr image) {
    RCLCPP_DEBUG(get_logger(), "Image received on subscribed topic");

    cv_bridge::CvImageConstPtr cv_image;
    cv_image = cv_bridge::toCvCopy(image, "mono8");

    cv::QRCodeDetector qrDecoder;
    const int w = cv_image->image.cols;
    const int h = cv_image->image.rows;

    // QR을 먼저 찾아서 주변을 ROI로 사용, 실패 시 전체 프레임 사용
    std::vector<cv::Point> qr_points;
    cv::Rect roi(w * 0.5, h * 0.5, w - (w * 0.5), h - (h * 0.5));
    if (qrDecoder.detect(cv_image->image, qr_points) && qr_points.size() >= 4) {
        cv::Rect qr_bounds = cv::boundingRect(qr_points);
        const int pad = static_cast<int>(0.1 * std::max(qr_bounds.width, qr_bounds.height));
        cv::Rect expanded(qr_bounds.x - pad, qr_bounds.y - pad, qr_bounds.width + 2 * pad, qr_bounds.height + 2 * pad);
        roi = expanded & cv::Rect(0, 0, w, h); // clamp to image
        RCLCPP_DEBUG(get_logger(), "Using QR-estimated ROI x:%d y:%d w:%d h:%d", roi.x, roi.y, roi.width, roi.height);
    } else {
        RCLCPP_DEBUG(get_logger(), "QR ROI not found, using full frame");
    }

    cv::Mat cropped = cv_image->image(roi);

    // 필요하면 업샘플과 전처리
    cv::Mat up;
    cv::resize(cropped, up, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);
    cv::Mat enhanced;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(up, enhanced);

    cv::Mat bw;
    cv::adaptiveThreshold(enhanced, bw, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);

    // 약한 샤프닝(선택)
    cv::Mat blur, sharp;
    cv::GaussianBlur(bw, blur, cv::Size(0, 0), 1.0);
    cv::addWeighted(bw, 1.5, blur, -0.5, 0, sharp);

    // 임시 토픽 발행(디버깅용)
    static rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub =
        this->create_publisher<sensor_msgs::msg::Image>("debug/enhanced_image", 10);
    sensor_msgs::msg::Image debug_msg;
    cv_bridge::CvImage debug_cv_image;
    debug_cv_image.header = image->header;
    debug_cv_image.encoding = "mono8";
    debug_cv_image.image = sharp;
    debug_cv_image.toImageMsg(debug_msg);
    debug_pub->publish(debug_msg);

    zbar::Image zbar_image(sharp.cols, sharp.rows, "Y800", sharp.data, sharp.cols * sharp.rows);

    // zbar::Image zbar_image(cv_image->image.cols, cv_image->image.rows, "Y800", cv_image->image.data,
    //                        cv_image->image.cols * cv_image->image.rows);
    scanner_.scan(zbar_image);

    auto it_start = zbar_image.symbol_begin();
    auto it_end = zbar_image.symbol_end();
    if (it_start != it_end) {
        // If there are barcodes in the image, iterate over all barcode readings from image
        for (zbar::Image::SymbolIterator symbol_it = it_start; symbol_it != it_end; ++symbol_it) {
            zbar_ros_interfaces::msg::Symbol symbol;
            symbol.data = symbol_it->get_data();
            RCLCPP_DEBUG(get_logger(), "Barcode detected with data: '%s'", symbol.data.c_str());

            RCLCPP_DEBUG(get_logger(), "Polygon around barcode has %d points", symbol_it->get_location_size());
            for (zbar::Symbol::PointIterator point_it = symbol_it->point_begin(); point_it != symbol_it->point_end(); ++point_it) {
                vision_msgs::msg::Point2D point;
                point.x = (*point_it).x;
                point.y = (*point_it).y;
                RCLCPP_DEBUG(get_logger(), "  Point: %f, %f", point.x, point.y);
                symbol.points.push_back(point);
            }

            // verify if repeated barcode throttling is enabled
            if (throttle_ > 0.0) {
                const std::lock_guard<std::mutex> lock(memory_mutex_);

                const std::string& barcode = symbol.data;
                // check if barcode has been recorded as seen, and skip detection
                if (barcode_memory_.count(barcode) > 0) {
                    // check if time reached to forget barcode
                    if (now() > barcode_memory_.at(barcode)) {
                        RCLCPP_DEBUG(get_logger(), "Memory timed out for barcode, publishing");
                        barcode_memory_.erase(barcode);
                    } else {
                        // if timeout not reached, skip this reading
                        continue;
                    }
                }
                // record barcode as seen, with a timeout to 'forget'
                barcode_memory_.insert(std::make_pair(barcode, now() + rclcpp::Duration(std::chrono::duration<double>(throttle_))));
            }

            // publish symbol
            RCLCPP_DEBUG(get_logger(), "Publishing Symbol");
            symbol_pub_->publish(symbol);

            // publish on deprecated barcode topic
            RCLCPP_DEBUG(get_logger(), "Publishing data as string");
            std_msgs::msg::String barcode_string;
            barcode_string.data = symbol.data;
            barcode_pub_->publish(barcode_string);
        }
    } else {
        RCLCPP_DEBUG(get_logger(), "No barcode detected in image");
    }

    // Warn if there are subscriptions on barcode topic, because it's deprecated.
    static bool alreadyWarnedDeprecation = false;
    if (!alreadyWarnedDeprecation && count_subscribers("barcode") > 0) {
        alreadyWarnedDeprecation = true;
        RCLCPP_WARN(get_logger(), "A subscription was detected on the deprecated topic 'barcode'. Please update the node "
                                  "that is subscribing to use the new topic 'symbol' with type "
                                  "'zbar_ros_interfaces::msg::Symbol' instead. The 'barcode' topic will be removed "
                                  "in the next distribution.");
    }

    zbar_image.set_data(NULL, 0);
}

void BarcodeReaderNode::cleanCb() {
    const std::lock_guard<std::mutex> lock(memory_mutex_);
    auto it = barcode_memory_.begin();
    while (it != barcode_memory_.end()) {
        if (now() > it->second) {
            RCLCPP_DEBUG(get_logger(), "Cleaned %s from memory", it->first.c_str());
            it = barcode_memory_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace zbar_ros
