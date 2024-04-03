#include <vector>
#include <unordered_map>
// ncnn
#include "layer.h"
#include "benchmark.h"
#include <net.h>
#include <cpu.h>

#include "logutils.h"
#include "BYTETracker.h"
#include "yolov8.h"

#include "model.id.h"

#define ASSERT(status, ret)     if (!(status)) { return ret; }
#define ASSERT_FALSE(status)    ASSERT(status, false)

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Net row_detector;
static ncnn::Net box_detector;
static ncnn::Net resnet;
static std::unordered_map<int, std::string> label_map;
static std::unordered_map<std::string, std::string> sub_category_map;
static std::unordered_map<std::string, std::string> sku_name_map;

//static BYTETracker tracker(15, 20);

static const int num_class = 1;
//static const int target_size = 320;
static const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
static const float mean_vals[3] = {0.f, 0.f, 0.f};

//static const cv::Scalar cc(0,  0, 255);
static const cv::Scalar textcc(255, 255, 255);

static const unsigned char colors[19][3] = {
        { 54,  67, 244},
        { 99,  30, 233},
        {176,  39, 156},
        {183,  58, 103},
        {181,  81,  63},
        {243, 150,  33},
        {244, 169,   3},
        {212, 188,   0},
        {136, 150,   0},
        { 80, 175,  76},
        { 74, 195, 139},
        { 57, 220, 205},
        { 59, 235, 255},
        {  7, 193, 255},
        {  0, 152, 255},
        { 34,  87, 255},
        { 72,  85, 121},
        {158, 158, 158},
        {139, 125,  96}
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    if (a.rect.x > b.rect.x + b.rect.width ||
    a.rect.x + a.rect.width < b.rect.x ||
    a.rect.y > b.rect.y + b.rect.height ||
    a.rect.y + a.rect.height < b.rect.y)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.rect.x + a.rect.width, b.rect.x + b.rect.width) - std::max(a.rect.x, b.rect.x);
    float inter_height = std::min(a.rect.y + a.rect.height, b.rect.y + b.rect.height) - std::max(a.rect.y, b.rect.y);

    return inter_width * inter_height;
}

static inline float intersection_area(const STrack& a, const STrack& b)
{
    if (a.tlwh[0] > b.tlwh[0] + b.tlwh[2] ||
        a.tlwh[0] + a.tlwh[2] < b.tlwh[0] ||
        a.tlwh[1] > b.tlwh[1] + b.tlwh[3] ||
        a.tlwh[1] + a.tlwh[3] < b.tlwh[1])
    {
        // no intersection
        return 0.f;
    }
    float inter_width = std::min(a.tlwh[0] + a.tlwh[2], b.tlwh[0] + b.tlwh[2]) - std::max(a.tlwh[0], b.tlwh[0]);
    float inter_height = std::min(a.tlwh[1] + a.tlwh[3], b.tlwh[1] + b.tlwh[3]) - std::max(a.tlwh[1], b.tlwh[1]);
    return inter_width * inter_height;
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_points = grid_strides.size();
//    const int num_class = num_class;
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++)
    {
        const float* scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = scores[k];
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);
        if (box_prob >= prob_threshold)
        {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
            {
                ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++)
            {
                float dis = 0.f;
                const float* dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++)
                {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;

            objects.push_back(obj);
        }
    }
}

static void _detect(ncnn::Net& yolo, ncnn::Mat& in, int w, int h, float scale, int width, int height, std::vector<Object>& objects){
    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
    LOGE("in_pad chw (%5d, %5d, %5d)\n", in_pad.c, in_pad.h, in_pad.w);
    // row_detector
    // std::vector<Object> objects;
    {
        in_pad.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = yolo.create_extractor();
        ex.input("images", in_pad);
        std::vector<Object> proposals;

        ncnn::Mat out;
        ex.extract("output0", out);
//        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "ex.extract done");
        // LOGE("inference done\n");
        LOGE("output shape chw (%5d, %5d, %5d)\n", out.c, out.h, out.w);
        std::vector<int> strides = {8, 16, 32}; // might have stride=64
        std::vector<GridAndStride> grid_strides;
        generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
//        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "generate_grids_and_stride done");
        generate_proposals(grid_strides, out, 0.5, proposals);
//        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "generate_proposals done");

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(proposals);
//        LOGE("qsort_descent_inplace done\n");
//        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "qsort_descent_inplace done");
        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, 0.5);
//        LOGE("nms_sorted_bboxes done\n");
//        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "nms_sorted_bboxes done");
        unsigned int count = picked.size();

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
            float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;

            objects[i].label = "ROW";
            objects[i].sku_name = "ROW";
            LOGE("x: %f, y: %f, w: %f, h: %f", objects[i].rect.x, objects[i].rect.y, objects[i].rect.width, objects[i].rect.height);
        }
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "unit done");
    }
}

static void _classify(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, int resnet_size){
    for (Object & obj : objects){
        // must recreate extractor for each object
        ncnn::Mat in;
        ncnn::Mat out;
        ncnn::Extractor ex = resnet.create_extractor();
        float x = obj.rect.x;
        float y = obj.rect.y;
        float w = obj.rect.width;
        float h = obj.rect.height;
        // LOGI("crop x: %f, y: %f", x, y);
        // Calculate the aspect ratio of the object
        float aspect_ratio = static_cast<float>(w) / static_cast<float>(h);
        double t1 = ncnn::get_current_time();
        // Calculate the target dimensions while maintaining the aspect ratio
        int target_width, target_height;
        if (w > h) {
            target_width = resnet_size;
            target_height = resnet_size / aspect_ratio;
        } else {
            target_width = resnet_size * aspect_ratio;
            target_height = resnet_size;
        }

        // Calculate the padding size on the left side
        cv::Rect roi(x, y, w, h);

        // Crop the image using the ROI
        cv::Mat cropped = rgb(roi);

        // Resize the cropped image to the target size
        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(target_width, target_height));
        cv::Mat padded;
        if (target_width > target_height){
            int pad = (target_width - target_height) / 2;
            cv::copyMakeBorder(resized, padded, pad, pad, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
        }else{
            int pad = (target_height - target_width) / 2;
            cv::copyMakeBorder(resized, padded, 0, 0, pad, pad, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
        }

        LOGE("padded width: %d, padded height: %d\n", padded.cols, padded.rows);
        in = ncnn::Mat::from_pixels(padded.data, ncnn::Mat::PIXEL_BGR2RGB, resnet_size, resnet_size);
        // ∂LOGE("w: %d, h: %d, c: %d\n", in.w, in.h,in.c);
        float mean[] = {123.675f, 116.28f, 103.53f};
        float norm[] = {1/58.395,1/57.12,1/57.375};
        in.substract_mean_normalize(mean, norm);
        ex.input(0, in);
        double t2 = ncnn::get_current_time();
        ex.extract(121, out);
        double t3 = ncnn::get_current_time();

        ncnn::Mat out_flatterned = out.reshape(out.w * out.h * out.c);
        // LOGE("w: %d, h: %d, c: %d\n", out.w, out.h,out.c);
        std::vector<float> scores;
        scores.resize(out.w);
        for (int j=0; j<out_flatterned.w; ++j)
            scores[j] = out_flatterned[j];

        float max_v = std::numeric_limits<float>::lowest();
        int max_v_index = -1;
        for (int i = 0; i < scores.size(); i++) {
            if (scores[i] > max_v) {
                max_v = scores[i];
                max_v_index = i;
            }
        }

        if (max_v > prob_threshold){
            obj.label = label_map[max_v_index];
            obj.sku_name = sku_name_map[obj.label];
            obj.sub_category = sub_category_map[obj.label];
//            obj.sub_category = "HPC";
            double t4 = ncnn::get_current_time();
            LOGE("preprocess: %f, infer: %f, postprocess: %f", (t2-t1), (t3-t2), (t4-t3));
            LOGE("max_score: %f, sku_id: %s, sku_name: %s\n", max_v, obj.label.c_str(), obj.sku_name.c_str());
        }
        obj.box_type = 1;
        obj.prob = max_v;
    }
}

static int draw_boxed_count(cv::Mat& rgb, unsigned int box_count)
{
    char text[32];
    sprintf(text, "ROWs=%d", box_count);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_TRIPLEX, 1, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(0, 0, 0), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(255, 255, 255), 1, LINE_AA);

    return 0;
}

static int draw(cv::Mat& rgb, const std::vector<STrack>& objects)
{
    // LOGI("draw boxes");
    int count = 0;
    for (const STrack & output_strack : objects)
    {
        if (output_strack.box_type == 1)
            continue;

        int x = output_strack.tlwh[0];
        int y = output_strack.tlwh[1];
        int w = output_strack.tlwh[2];
        int h = output_strack.tlwh[3];


        int baseLine = 0;
        float font_scale = 1;
        int font_face = cv::FONT_HERSHEY_TRIPLEX;

        // convert cont to text
        char sku_name[32];
        const unsigned char* color = colors[output_strack.track_id % 19];
        cv::Scalar cc = cv::Scalar(color[0], color[1], color[2]);
        sprintf(sku_name, "Row %d", ++count);

        // draw box
        rectangle(rgb, Rect(x, y, w, h), cc, 3, LINE_4);

        cv::Size label_size = cv::getTextSize(sku_name, font_face, font_scale, 1, &baseLine);
        int x_text = x + w / 2 - label_size.width / 2;
        // y_text below the top of the rectangle
        int y_text = y + label_size.height - 2;
        cv::rectangle(rgb,
                      cv::Rect(cv::Point(x_text, y_text - label_size.height),cv::Size(label_size.width, label_size.height + baseLine)),
                      cc,
                      -1,
                      LINE_4);
        cv::putText(rgb, sku_name,
                    cv::Point(x_text, y_text),
                    font_face,
                    font_scale,
                    textcc,
                    1,
                    LINE_AA);
    }
    return 0;
}

static int draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    // LOGI("draw boxes");
    int count = 0;
    for (const Object & output_strack : objects)
    {
        if (output_strack.box_type == 1)
            continue;

        int x = output_strack.rect.x;
        int y = output_strack.rect.y;
        int w = output_strack.rect.width;
        int h = output_strack.rect.height;

        // draw box
        // green box
        rectangle(rgb, Rect(x, y, w, h), cv::Scalar(0, 255, 0), 1, LINE_4);
//
//        cv::Size label_size = cv::getTextSize(sku_name, font_face, font_scale, 1, &baseLine);
//        int x_text = x + w / 2 - label_size.width / 2;
//        // y_text below the top of the rectangle
//        int y_text = y + label_size.height - 2;
//        cv::rectangle(rgb,
//                      cv::Rect(cv::Point(x_text, y_text - label_size.height),cv::Size(label_size.width, label_size.height + baseLine)),
//                      cc,
//                      -1,
//                      LINE_4);
//        cv::putText(rgb, sku_name,
//                    cv::Point(x_text, y_text),
//                    font_face,
//                    font_scale,
//                    textcc,
//                    1,
//                    LINE_AA);
    }
    return 0;
}

static bool is_same_box(const STrack& a, const STrack& b, const float threshold=0.7){
    float inter_area = intersection_area(a, b);
    float union_area = a.tlwh[2] * a.tlwh[3] + b.tlwh[2] * b.tlwh[3] - inter_area;
    float iou = inter_area / union_area;
//    LOGE("iou: %f", iou);
    return iou >= threshold;
}

static bool is_same_boxes(const std::vector<STrack>& objects, const std::vector<STrack>& last_objects)
{
    if (last_objects.empty()){
//        LOGE("last_objects is empty");
        return false;
    }
    if (objects.size() != last_objects.size()){
        return false;
    }
    for (int i=0; i<objects.size(); ++i){
        for (int j=0; j<last_objects.size(); ++j){
            if (is_same_box(objects[i], last_objects[j])){
                break;
            }
            if (j == last_objects.size() - 1){
                return false;
            }
        }
    }
    return true;
}

static int detect(ncnn::Net& yolo, const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold, int target_size)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGBA2BGR, width, height, w, h);
    LOGE("in chw: %d, %d, %d\n", in.c, in.h, in.w);
    _detect(yolo, in, w, h, scale, width, height, objects);
    return 0;
}

static cv::Ptr<FeatureDetector> detector = cv::ORB::create(500, 1.2f, 8,
                                                           31, 0, 2,
                                                           ORB::HARRIS_SCORE, 31,
                                                           50);
static cv::BFMatcher matcher(NORM_L2);
static void feature_match(const cv::Mat& image, const cv::Mat& image2, cv::Mat& matchImg) {
    std::vector<KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptor1, descriptor2;
    detector->detectAndCompute(image, Mat(), keypoints1, descriptor1);
    detector->detectAndCompute(image2, Mat(), keypoints2, descriptor2);

    std::vector<DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);
    // filter matches by score 0.6
    std::vector<DMatch> good_matches;
    for (auto & matche : matches) {
        if (matche.distance < 0.6 * 32) {
            good_matches.push_back(matche);
        }
    }

    cv::drawMatches(image, keypoints1, image2, keypoints2, good_matches, matchImg);
    //cv::imshow("matchImg", matchImg);
}

static bool bitmap_tp_mat(JNIEnv * env, cv::Mat & matrix, jobject obj_bitmap) {
    void * bitmapPixels;                                            // Save picture pixel data
    AndroidBitmapInfo bitmapInfo;                                   // Save picture parameters

    ASSERT_FALSE( AndroidBitmap_getInfo(env, obj_bitmap, &bitmapInfo) >= 0);        // Get picture parameters
    ASSERT_FALSE( bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888
                  || bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGB_565 );          // Only ARGB? 8888 and RGB? 565 are supported
    ASSERT_FALSE( AndroidBitmap_lockPixels(env, obj_bitmap, &bitmapPixels) >= 0 );  // Get picture pixels (lock memory block)
    ASSERT_FALSE( bitmapPixels );

    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGI("ANDROID_BITMAP_FORMAT_RGBA_8888");
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);    // Establish temporary mat
        tmp.copyTo(matrix);                                                         // Copy to target matrix
    } else {
        LOGI("Not ANDROID_BITMAP_FORMAT_RGBA_8888");
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC2, bitmapPixels);
        cv::cvtColor(tmp, matrix, cv::COLOR_BGR5652RGB);
    }

    //convert RGB to BGR
//    cv::cvtColor(matrix,matrix,cv::COLOR_RGB2BGR);

    AndroidBitmap_unlockPixels(env, obj_bitmap);            // Unlock
    return true;
}