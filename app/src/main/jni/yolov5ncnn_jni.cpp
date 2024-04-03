// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>
#include <unistd.h>
#include <chrono>
#include <thread>

#include "ndkcamera.h"
#include "yolov8.cpp"

#if __ARM_NEON
#include <arm_neon.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>

#endif // __ARM_NEON

#include "model.mem.h"
#include "m.mem.h"

static ncnn::Mutex _lock;

static BYTETracker tracker(30, 30);



class MyNdkCamera : public NdkCameraWindow
{
public:
    std::vector<STrack> last_stracks;
    cv::Mat row;
    void on_image_render(cv::Mat& rgb) override;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb)
{
    // __android_log_print(ANDROID_LOG_WARN, "NdkCamera_D", "MyNdkCamera::on_image_render");
    auto start = chrono::system_clock::now();
    std::vector<Object> rows;
    std::vector<Object> boxes;
    std::vector<STrack> stracks;
    {
        ncnn::MutexLockGuard g(_lock);
        detect(box_detector, rgb, boxes, 0.5, 0.5, 320);
        detect(row_detector, rgb, rows, 0.5, 0.5, 320);
        stracks = tracker.update(rows);
        if (is_same_boxes(stracks, last_stracks))
            stracks = last_stracks;
        else{
            // sort output_stracks by rect.y in ascending order
            std::sort(stracks.begin(), stracks.end(), [](const STrack& a, const STrack& b) {
                return a.tlwh[1] < b.tlwh[1];
            });
            last_stracks = stracks;
        }
        draw(rgb, stracks);
        draw(rgb, boxes);
        draw_boxed_count(rgb, stracks.size());
    }
    auto end = chrono::system_clock::now();
    auto total_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    LOGE("render time: %ld ms", total_ms);
}


static MyNdkCamera* g_camera = 0;


extern "C" {

static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID xId;
static jfieldID yId;
static jfieldID wId;
static jfieldID hId;
static jfieldID labelId;
static jfieldID subCategory;
static jfieldID skuName;
static jfieldID probId;

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "JNI_OnLoad");

    ncnn::create_gpu_instance();
    // ncnn::MutexLockGuard g(lock);
    g_blob_pool_allocator.set_size_compare_ratio(0.f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.f);

    g_camera = new MyNdkCamera;

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "JNI_OnUnload");
    {
        ncnn::MutexLockGuard g(_lock);
    }
    ncnn::destroy_gpu_instance();
    row_detector.clear();
    resnet.clear();
    delete g_camera;
    g_camera = 0;
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL
Java_com_google_ar_core_examples_java_ml_classification_YoloV5Ncnn_Init(JNIEnv *env, jobject thiz, jobject assetManager) {
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "init");
    row_detector.clear();
    resnet.clear();
    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    // row_detector 320 int8 fot preview
    row_detector.opt = ncnn::Option();
    row_detector.opt.num_threads = ncnn::get_big_cpu_count();
    row_detector.opt.blob_allocator = &g_blob_pool_allocator;
    row_detector.opt.workspace_allocator = &g_workspace_pool_allocator;
//    row_detector.opt.use_fp16_packed = true;
//    row_detector.opt.use_bf16_storage = true;
//    row_detector.opt.use_fp16_arithmetic = true;
    row_detector.opt.use_int8_inference = true;
    row_detector.opt.use_int8_storage = true;
    row_detector.opt.use_int8_arithmetic = true;
    row_detector.opt.use_int8_packed = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    row_detector.load_param(mgr, "shelf_320_int8.param");
    row_detector.load_model(mgr, "shelf_320_int8.bin");

    // row_detector 320 int8 fot preview
    box_detector.opt = ncnn::Option();
    box_detector.opt.num_threads = ncnn::get_big_cpu_count();
    box_detector.opt.blob_allocator = &g_blob_pool_allocator;
    box_detector.opt.workspace_allocator = &g_workspace_pool_allocator;
    box_detector.opt.use_int8_inference = true;
    box_detector.opt.use_int8_storage = true;
    box_detector.opt.use_int8_arithmetic = true;
    box_detector.opt.use_int8_packed = true;

    box_detector.load_param(mgr, "china-unit-int8.param");
    box_detector.load_model(mgr, "china-unit-int8.bin");

    jclass localObjCls = env->FindClass("com/google/ar/core/examples/java/ml/classification/YoloV5Ncnn$Obj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

    constructortorId = env->GetMethodID(objCls, "<init>",
                                        "()V");

    xId = env->GetFieldID(objCls, "x", "F");
    yId = env->GetFieldID(objCls, "y", "F");
    wId = env->GetFieldID(objCls, "w", "F");
    hId = env->GetFieldID(objCls, "h", "F");
    labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
    skuName = env->GetFieldID(objCls, "skuName", "Ljava/lang/String;");
    subCategory = env->GetFieldID(objCls, "subCategory", "Ljava/lang/String;");
    probId = env->GetFieldID(objCls, "prob", "F");

    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "init done");

    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_google_ar_core_examples_java_ml_classification_YoloV5Ncnn_Release(JNIEnv *env, jobject thiz) {
    // TODO: implement Release()
    row_detector.clear();
    resnet.clear();
    return JNI_TRUE;
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_google_ar_core_examples_java_ml_classification_YoloV5Ncnn_openCamera(JNIEnv *env, jobject thiz, jint facing) {
    // TODO: implement openCamera()

    g_camera->open(facing);

    return JNI_TRUE;
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_google_ar_core_examples_java_ml_classification_YoloV5Ncnn_closeCamera(JNIEnv *env, jobject thiz) {
    // TODO: implement closeCamera()
    g_camera->close();

    return JNI_TRUE;
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_google_ar_core_examples_java_ml_classification_YoloV5Ncnn_setOutputWindow(JNIEnv *env, jobject thiz, jobject surface) {
    // TODO: implement setOutputWindow()
    ANativeWindow *win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);
    return JNI_TRUE;
}
extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_google_ar_core_examples_java_ml_classification_YoloV5Ncnn_Detect(JNIEnv *env, jobject thiz, jobject bitmap) {
    // TODO: implement Detect()
//    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "detect");

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
//    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "prepare bitmap done");

    cv::Mat rgb;
    std::vector<Object> objects;
    bool ret = bitmap_tp_mat(env, rgb,bitmap);
    if (!ret) {
        return NULL;
    }
//    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "rgn chw %d %d %d", rgb.channels(), rgb.rows, rgb.cols);
    ncnn::MutexLockGuard g(_lock);
    detect(box_detector, rgb, objects, 0.5,0.5, 320);

//    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "inference done");
    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);
//    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "objects size %d", objects.size());
    for (size_t i=0; i<objects.size(); i++)
    {
        jobject jObj = env->NewObject(objCls, constructortorId, thiz);

        env->SetFloatField(jObj, xId, objects[i].rect.x);
        env->SetFloatField(jObj, yId, objects[i].rect.y);
        env->SetFloatField(jObj, wId, objects[i].rect.width);
        env->SetFloatField(jObj, hId, objects[i].rect.height);
        env->SetObjectField(jObj, labelId, env->NewStringUTF(objects[i].label.c_str()));
        env->SetObjectField(jObj, skuName, env->NewStringUTF(objects[i].sku_name.c_str()));
        env->SetObjectField(jObj, subCategory, env->NewStringUTF(objects[i].sub_category.c_str()));
        env->SetFloatField(jObj, probId, objects[i].prob);
        env->SetObjectArrayElement(jObjArray, i, jObj);
    }

    double elasped = ncnn::get_current_time() - start_time;
//    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "%.2fms   detect done", elasped);

    return jObjArray;
}
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_google_ar_core_examples_java_ml_classification_YoloV5Ncnn_DetectRows(JNIEnv *env,
                                                                              jobject thiz,
                                                                              jobject bitmap) {
    // TODO: implement DetectRows()

    // TODO: implement Detect()
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "row detect");

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "prepare bitmap done");

    cv::Mat rgb;
    std::vector<Object> objects;
    bool ret = bitmap_tp_mat(env, rgb,bitmap);
    if (!ret) {
        return NULL;
    }
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "rgn chw %d %d %d", rgb.channels(), rgb.rows, rgb.cols);
    ncnn::MutexLockGuard g(_lock);
    detect(row_detector, rgb, objects, 0.5,0.5, 320);

    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "inference done");
    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "objects size %d", objects.size());
    for (size_t i=0; i<objects.size(); i++)
    {
        jobject jObj = env->NewObject(objCls, constructortorId, thiz);

        env->SetFloatField(jObj, xId, objects[i].rect.x);
        env->SetFloatField(jObj, yId, objects[i].rect.y);
        env->SetFloatField(jObj, wId, objects[i].rect.width);
        env->SetFloatField(jObj, hId, objects[i].rect.height);
        env->SetObjectField(jObj, labelId, env->NewStringUTF(objects[i].label.c_str()));
        env->SetObjectField(jObj, skuName, env->NewStringUTF(objects[i].sku_name.c_str()));
        env->SetObjectField(jObj, subCategory, env->NewStringUTF(objects[i].sub_category.c_str()));
        env->SetFloatField(jObj, probId, objects[i].prob);
        env->SetObjectArrayElement(jObjArray, i, jObj);
    }

    double elasped = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn_Debug", "%.2fms   detect done", elasped);

    return jObjArray;
}