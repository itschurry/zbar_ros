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
#include <array>
#include <algorithm>
#include <functional>
#include <chrono>
#include <cmath>
#include "zbar_ros/barcode_reader_node.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

namespace {

struct QrCandidate {
    cv::Rect roi;                      // full-res coordinates
    std::array<cv::Point2f, 4> quad{}; // TL, TR, BR, BL full-res (if available)
    bool has_quad{false};
    float score{0.0f};
};

static inline cv::Rect clampRect(const cv::Rect& r, const cv::Size& sz) {
    return r & cv::Rect(0, 0, sz.width, sz.height);
}

static inline float l2(const cv::Point2f& a, const cv::Point2f& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static std::array<cv::Point2f, 4> orderQuadTLTRBRBL(const std::vector<cv::Point2f>& pts) {
    // Robust ordering for 4 points. Uses sum/diff heuristic.
    std::array<cv::Point2f, 4> out;
    CV_Assert(pts.size() == 4);
    int tl = 0, br = 0, tr = 0, bl = 0;
    float minSum = 1e9f, maxSum = -1e9f;
    float minDiff = 1e9f, maxDiff = -1e9f;
    for (int i = 0; i < 4; ++i) {
        const float s = pts[i].x + pts[i].y;
        const float d = pts[i].x - pts[i].y;
        if (s < minSum) {
            minSum = s;
            tl = i;
        }
        if (s > maxSum) {
            maxSum = s;
            br = i;
        }
        if (d < minDiff) {
            minDiff = d;
            bl = i;
        }
        if (d > maxDiff) {
            maxDiff = d;
            tr = i;
        }
    }
    out[0] = pts[tl];
    out[1] = pts[tr];
    out[2] = pts[br];
    out[3] = pts[bl];
    return out;
}

static bool detectQrRoiOpenCV(const cv::Mat& gray_small, float inv_scale_to_full, const cv::Size& full_size, QrCandidate* out) {
    // NOTE: We intentionally run detection on a *small* grayscale image for speed.
    // Detection tends to work better on natural grayscale (not hard-thresholded).
    cv::QRCodeDetector det;

    // Light contrast boost (cheap) — helps with faint QR prints.
    cv::Mat proc;
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(gray_small, proc);
    }

    std::vector<cv::Point> pts_i;
    if (!det.detect(proc, pts_i) || pts_i.size() < 4) {
        return false;
    }

    std::vector<cv::Point2f> pts;
    pts.reserve(4);
    for (int i = 0; i < 4; ++i) {
        pts.emplace_back(static_cast<float>(pts_i[i].x) * inv_scale_to_full, static_cast<float>(pts_i[i].y) * inv_scale_to_full);
    }

    auto quad = orderQuadTLTRBRBL(pts);
    // Build padded ROI in full-res
    cv::Rect bbox = cv::boundingRect(std::vector<cv::Point2f>{quad[0], quad[1], quad[2], quad[3]});
    const int pad = static_cast<int>(0.20f * std::max(bbox.width, bbox.height));
    bbox.x -= pad;
    bbox.y -= pad;
    bbox.width += 2 * pad;
    bbox.height += 2 * pad;
    bbox = clampRect(bbox, full_size);

    out->roi = bbox;
    out->quad = quad;
    out->has_quad = true;
    out->score = static_cast<float>(bbox.area());
    return true;
}

static bool detectQrRoiMorph(const cv::Mat& gray_small, float inv_scale_to_full, const cv::Size& full_size, QrCandidate* out) {
    // Fallback detector (no QR decode). Fast-ish heuristic:
    // 1) Find dense edge regions
    // 2) Morph close
    // 3) Contour filter: square-ish, big enough
    // 4) Score by edge density inside bbox
    cv::Mat g;
    cv::GaussianBlur(gray_small, g, cv::Size(3, 3), 0.0);
    cv::Mat gx, gy;
    cv::Sobel(g, gx, CV_16S, 1, 0, 3);
    cv::Sobel(g, gy, CV_16S, 0, 1, 3);
    cv::Mat absx, absy;
    cv::convertScaleAbs(gx, absx);
    cv::convertScaleAbs(gy, absy);
    cv::Mat grad;
    cv::addWeighted(absx, 1.0, absy, 1.0, 0.0, grad);

    cv::Mat bw;
    cv::threshold(grad, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Close gaps to form blobs around QR
    const int k = std::max(3, (std::min(gray_small.cols, gray_small.rows) / 80) | 1);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
    cv::morphologyEx(bw, bw, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    cv::morphologyEx(bw, bw, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return false;

    // Integral image of edges for quick density scoring
    cv::Mat edges;
    cv::Canny(g, edges, 60, 180);
    cv::Mat integral;
    cv::integral(edges, integral, CV_32S);

    auto sumRect = [&](const cv::Rect& r) -> int {
        const int x1 = r.x;
        const int y1 = r.y;
        const int x2 = r.x + r.width;
        const int y2 = r.y + r.height;
        const int* p0 = integral.ptr<int>(y1);
        const int* p1 = integral.ptr<int>(y2);
        return p1[x2] - p1[x1] - p0[x2] + p0[x1];
    };

    QrCandidate best;
    best.score = -1.0f;

    for (const auto& c : contours) {
        cv::Rect r = cv::boundingRect(c);
        const int area = r.area();
        if (area < (gray_small.cols * gray_small.rows) / 400) continue; // too small

        const float ar = static_cast<float>(r.width) / static_cast<float>(r.height);
        if (ar < 0.6f || ar > 1.6f) continue; // QR tends to be roughly square in image

        // Pad in small coords then score
        const int pad = static_cast<int>(0.20f * std::max(r.width, r.height));
        cv::Rect rp(r.x - pad, r.y - pad, r.width + 2 * pad, r.height + 2 * pad);
        rp = rp & cv::Rect(0, 0, gray_small.cols, gray_small.rows);

        const int e = sumRect(rp);
        const float density = static_cast<float>(e) / static_cast<float>(rp.area() + 1);
        const float score = density * std::sqrt(static_cast<float>(rp.area()));

        if (score > best.score) {
            best.score = score;
            // scale back to full
            cv::Rect full_r(
                static_cast<int>(std::lround(rp.x * inv_scale_to_full)), static_cast<int>(std::lround(rp.y * inv_scale_to_full)),
                static_cast<int>(std::lround(rp.width * inv_scale_to_full)), static_cast<int>(std::lround(rp.height * inv_scale_to_full)));
            best.roi = clampRect(full_r, full_size);
            best.has_quad = false;
        }
    }

    if (best.score <= 0.0f || best.roi.area() <= 0) return false;
    *out = best;
    return true;
}

static cv::Mat warpOrCropQr(const cv::Mat& gray_full, const QrCandidate& cand, int min_out = 320) {
    cv::Mat roi = gray_full(cand.roi);
    if (!cand.has_quad) {
        // Simple crop; upscale a bit for zbar.
        if (std::min(roi.cols, roi.rows) < min_out) {
            const double s = static_cast<double>(min_out) / static_cast<double>(std::min(roi.cols, roi.rows));
            cv::Mat up;
            cv::resize(roi, up, cv::Size(), s, s, cv::INTER_LINEAR);
            return up;
        }
        return roi;
    }

    // Perspective warp to a square. Helps zbar when the QR is rotated/tilted.
    const auto& q = cand.quad;
    // Estimate output size from quad edge lengths
    const float w1 = l2(q[0], q[1]);
    const float w2 = l2(q[3], q[2]);
    const float h1 = l2(q[0], q[3]);
    const float h2 = l2(q[1], q[2]);
    int out_w = static_cast<int>(std::lround(std::max(w1, w2)));
    int out_h = static_cast<int>(std::lround(std::max(h1, h2)));
    int out_s = std::max({min_out, out_w, out_h});
    // Bigger warped ROI can drastically improve decode success when the original is slightly blurred.
    // 1600 is still fast enough on Pi5 for a single ROI per frame.
    out_s = std::min(out_s, 1600);

    // Build points in ROI-local coordinates
    std::vector<cv::Point2f> src(4), dst(4);
    for (int i = 0; i < 4; ++i) {
        src[i] = cv::Point2f(q[i].x - cand.roi.x, q[i].y - cand.roi.y);
    }
    dst[0] = cv::Point2f(0.f, 0.f);
    dst[1] = cv::Point2f(static_cast<float>(out_s - 1), 0.f);
    dst[2] = cv::Point2f(static_cast<float>(out_s - 1), static_cast<float>(out_s - 1));
    dst[3] = cv::Point2f(0.f, static_cast<float>(out_s - 1));

    cv::Mat H = cv::getPerspectiveTransform(src, dst);
    cv::Mat warped;
    // INTER_NEAREST preserves module edges better than linear when scaling.
    cv::warpPerspective(roi, warped, H, cv::Size(out_s, out_s), cv::INTER_NEAREST, cv::BORDER_REPLICATE);
    return warped;
}

static cv::Mat ensureGray8(const cv::Mat& in) {
  if (in.empty()) return cv::Mat();
  if (in.type() == CV_8UC1) return in;
  cv::Mat g;
  if (in.channels() == 3) cv::cvtColor(in, g, cv::COLOR_BGR2GRAY);
  else if (in.channels() == 4) cv::cvtColor(in, g, cv::COLOR_BGRA2GRAY);
  else in.convertTo(g, CV_8U);
  return g;
}

static cv::Mat padQuietZone(const cv::Mat& gray, float pad_frac = 0.15f) {
  // zbar (and many QR decoders) are very sensitive to the quiet zone.
  // We enforce a clean white border around the candidate ROI.
  cv::Mat g = ensureGray8(gray);
  const int pad = std::max(8, static_cast<int>(std::lround(pad_frac * std::min(g.cols, g.rows))));
  cv::Mat out;
  cv::copyMakeBorder(g, out, pad, pad, pad, pad, cv::BORDER_CONSTANT, cv::Scalar(255));
  return out;
}

static cv::Mat upscaleIfSmall(const cv::Mat& gray, int min_side = 320) {
  cv::Mat g = ensureGray8(gray);
  const int s = std::min(g.cols, g.rows);
  if (s >= min_side) return g;
  const double scale = static_cast<double>(min_side) / static_cast<double>(std::max(1, s));
  cv::Mat up;
  // INTER_NEAREST preserves module edges better than linear for QR.
  cv::resize(g, up, cv::Size(), scale, scale, cv::INTER_NEAREST);
  return up;
}

static std::vector<cv::Mat> buildDecodeVariants(const cv::Mat& qr_gray_view) {
  // IMPORTANT: For decoding, avoid aggressive binarization first.
  // Try a small set of cheap variants, ordered by speed/robustness.
  std::vector<cv::Mat> outs;
  outs.reserve(3);

  // Use a slightly larger minimum side to help zbar when the QR is a bit blurred.
  cv::Mat g0 = upscaleIfSmall(qr_gray_view, /*min_side=*/512);
  g0 = padQuietZone(g0, 0.20f);
  outs.push_back(g0);

  // Variant 2: mild denoise (helps speckle after resize)
  cv::Mat den;
  cv::medianBlur(g0, den, 3);
  outs.push_back(den);

  // Variant 3: CLAHE (helps low-contrast printed QR)
  cv::Mat eq;
  {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(den, eq);
  }
  outs.push_back(eq);

  // Variant 4: adaptive threshold (often better than Otsu when lighting is uneven)
  cv::Mat ad;
  cv::adaptiveThreshold(eq, ad, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 31, 5);
  outs.push_back(ad);

  // Variant 5: Otsu on CLAHE (last resort)
  cv::Mat bw;
  cv::threshold(eq, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  outs.push_back(bw);

  // Variant 6: inverted Otsu (some prints/cameras invert polarity)
  cv::Mat inv;
  cv::bitwise_not(bw, inv);
  outs.push_back(inv);

  return outs;
}

} // namespace

using namespace std::chrono_literals;

namespace zbar_ros {

BarcodeReaderNode::BarcodeReaderNode() : Node("barcode_reader_node") {
    // Enable QR explicitly and bump scan density a bit.
    scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);
    scanner_.set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);
    scanner_.set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_X_DENSITY, 2);
    scanner_.set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_Y_DENSITY, 2);

    image_topic_ = this->declare_parameter<std::string>("image_topic", "camera/image/compressed");
    qr_code_topic_ = this->declare_parameter<std::string>("qr_code_topic", "/barcode");
    RCLCPP_DEBUG(get_logger(), "Subscribing to topics: %s, %s", image_topic_.c_str(), qr_code_topic_.c_str());

    camera_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        image_topic_, 10, std::bind(&BarcodeReaderNode::imageCb, this, std::placeholders::_1));

    symbol_pub_ = this->create_publisher<zbar_ros_interfaces::msg::Symbol>("symbol", 10);
    barcode_pub_ = this->create_publisher<std_msgs::msg::String>(qr_code_topic_, 10);

    debug_roi_pub_ = this->create_publisher<sensor_msgs::msg::Image>("debug/qr_roi", 10);
    debug_crop_pub_ = this->create_publisher<sensor_msgs::msg::Image>("debug/qr_crop", 10);
    debug_preprocessed_pub_ = this->create_publisher<sensor_msgs::msg::Image>("debug/qr_preprocessed", 10);

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

    const cv::Mat& gray_full = cv_image->image;
    const int w = gray_full.cols;
    const int h = gray_full.rows;

    // ---- 1) FAST ROI detection on a downsized frame ----
    const int target_max = 640; // tune: smaller = faster, larger = more robust
    const float scale = std::min(1.0f, static_cast<float>(target_max) / static_cast<float>(std::max(w, h)));
    const float inv_scale = 1.0f / scale;

    cv::Mat gray_small;
    if (scale < 1.0f) {
        cv::resize(gray_full, gray_small, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        gray_small = gray_full;
    }

    QrCandidate cand;
    bool have_roi = detectQrRoiOpenCV(gray_small, inv_scale, gray_full.size(), &cand);
    if (!have_roi) {
        have_roi = detectQrRoiMorph(gray_small, inv_scale, gray_full.size(), &cand);
    }

    if (!have_roi) {
        // 마지막 안전망: 중앙부만 살짝 잘라서(속도) 스캔
        const int cw = static_cast<int>(w * 0.60);
        const int ch = static_cast<int>(h * 0.60);
        cand.roi = clampRect(cv::Rect((w - cw) / 2, (h - ch) / 2, cw, ch), gray_full.size());
        cand.has_quad = false;
        RCLCPP_DEBUG(get_logger(), "QR ROI not found; using center crop x:%d y:%d w:%d h:%d", cand.roi.x, cand.roi.y, cand.roi.width,
                     cand.roi.height);
    } else {
        RCLCPP_DEBUG(get_logger(), "QR ROI found x:%d y:%d w:%d h:%d (quad=%s)", cand.roi.x, cand.roi.y, cand.roi.width, cand.roi.height,
                     cand.has_quad ? "yes" : "no");
    }

    // ---- 2) Extract (warp if possible) + preprocess for zbar ----
    cv::Mat qr_view = warpOrCropQr(gray_full, cand, /*min_out=*/512);
    auto decode_variants = buildDecodeVariants(qr_view);
    cv::Mat cropped = decode_variants.empty() ? cv::Mat() : decode_variants.front();

    // ---- 3) Debug image (optional but super useful when tuning) ----
    cv::Mat debug_bgr;
    cv::cvtColor(gray_full, debug_bgr, cv::COLOR_GRAY2BGR);
    cv::rectangle(debug_bgr, cand.roi, cv::Scalar(255, 0, 0), 2);
    if (cand.has_quad) {
        std::vector<cv::Point> p;
        p.reserve(4);
        for (int i = 0; i < 4; ++i)
            p.emplace_back(static_cast<int>(std::lround(cand.quad[i].x)), static_cast<int>(std::lround(cand.quad[i].y)));
        cv::polylines(debug_bgr, p, true, cv::Scalar(0, 255, 0), 2);
    }

    // ---- 3) Decode with zbar (use ORIGINAL-ish grayscale ROI; minimal preprocessing) ----
bool any_decoded = false;

for (size_t vi = 0; vi < decode_variants.size(); ++vi) {
    cv::Mat scan_img = decode_variants[vi];
    if (scan_img.empty()) continue;

    // zbar expects a contiguous buffer.
    if (!scan_img.isContinuous()) scan_img = scan_img.clone();

    zbar::Image zimg(scan_img.cols, scan_img.rows, "Y800", scan_img.data, scan_img.cols * scan_img.rows);
    const int n = scanner_.scan(zimg);

    auto it_start = zimg.symbol_begin();
    auto it_end = zimg.symbol_end();

    if (n > 0 && it_start != it_end) {
        // Keep the actual scanned image for debug publishing.
        cropped = scan_img;
        any_decoded = true;

        for (zbar::Image::SymbolIterator symbol_it = it_start; symbol_it != it_end; ++symbol_it) {
            zbar_ros_interfaces::msg::Symbol symbol;
            symbol.data = symbol_it->get_data();

            // We intentionally do NOT fill symbol.points here.
            // Points from zbar would be in the "scan_img" coordinate system (warped/padded),
            // which is rarely useful unless we map them back to full-res.
            // Most applications only need the decoded payload quickly.
            RCLCPP_DEBUG(get_logger(), "QR decoded (variant %zu) data: '%s'", vi, symbol.data.c_str());

            if (throttle_ > 0.0) {
                const std::lock_guard<std::mutex> lock(memory_mutex_);
                const std::string& barcode = symbol.data;

                if (barcode_memory_.count(barcode) > 0) {
                    if (now() > barcode_memory_.at(barcode)) {
                        RCLCPP_DEBUG(get_logger(), "Memory timed out for barcode, publishing");
                        barcode_memory_.erase(barcode);
                    } else {
                        continue;
                    }
                }
                barcode_memory_.insert(std::make_pair(
                    barcode, now() + rclcpp::Duration(std::chrono::duration<double>(throttle_))));
            }

            symbol_pub_->publish(symbol);

            std_msgs::msg::String barcode_string;
            barcode_string.data = symbol.data;
            barcode_pub_->publish(barcode_string);
        }

        zimg.set_data(nullptr, 0);
        break;  // stop after first successful variant
    }

    zimg.set_data(nullptr, 0);
}

if (!any_decoded) {
    // Fallback: OpenCV's QR decoder occasionally succeeds where zbar fails (especially on noisy binarized input).
    // This keeps zbar as primary, but gives us an extra chance without another full-frame detect.
    cv::QRCodeDetector dec;
    std::string decoded;
    for (size_t vi = 0; vi < decode_variants.size() && decoded.empty(); ++vi) {
        cv::Mat scan_img = decode_variants[vi];
        if (scan_img.empty()) continue;
        // detectAndDecode expects natural grayscale more than hard binary; try both as-is.
        std::vector<cv::Point> pts;
        decoded = dec.detectAndDecode(scan_img, pts);
        if (!decoded.empty()) {
            RCLCPP_DEBUG(get_logger(), "OpenCV fallback decoded (variant %zu) data: '%s'", vi, decoded.c_str());
            zbar_ros_interfaces::msg::Symbol symbol;
            symbol.data = decoded;
            symbol_pub_->publish(symbol);
            std_msgs::msg::String barcode_string;
            barcode_string.data = decoded;
            barcode_pub_->publish(barcode_string);
            any_decoded = true;
            cropped = scan_img;
            break;
        }
    }
    if (!any_decoded) {
        RCLCPP_DEBUG(get_logger(), "No QR decoded (zbar+opencv fallback). Tried %zu variants", decode_variants.size());
    }
}

// ---- 4) Publish debug images ----
    // Full-res frame with ROI/quad overlay (bgr8)
    if (debug_roi_pub_ && debug_roi_pub_->get_subscription_count() > 0) {
        sensor_msgs::msg::Image roi_msg;
        cv_bridge::CvImage roi_cv;
        roi_cv.header = image->header;
        roi_cv.encoding = "bgr8";
        roi_cv.image = debug_bgr;
        roi_cv.toImageMsg(roi_msg);
        debug_roi_pub_->publish(roi_msg);
    }

    // Cropped/warped QR view (mono8)
    if (debug_crop_pub_ && debug_crop_pub_->get_subscription_count() > 0) {
        sensor_msgs::msg::Image crop_msg;
        cv_bridge::CvImage crop_cv;
        crop_cv.header = image->header;
        crop_cv.encoding = "mono8";
        crop_cv.image = qr_view;
        crop_cv.toImageMsg(crop_msg);
        debug_crop_pub_->publish(crop_msg);
    }

    // Preprocessed image actually scanned by zbar (mono8)
    if (debug_preprocessed_pub_ && debug_preprocessed_pub_->get_subscription_count() > 0) {
        sensor_msgs::msg::Image pre_msg;
        cv_bridge::CvImage pre_cv;
        pre_cv.header = image->header;
        pre_cv.encoding = "mono8";
        pre_cv.image = cropped;
        pre_cv.toImageMsg(pre_msg);
        debug_preprocessed_pub_->publish(pre_msg);
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
