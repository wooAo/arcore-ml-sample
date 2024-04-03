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

package com.google.ar.core.examples.java.ml.classification;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.view.Surface;

public class YoloV5Ncnn
{
    public native boolean Init(AssetManager mgr);

    public native boolean Release();

    public class Obj
    {
        public float x;
        public float y;
        public float w;
        public float h;
        public String label;
        public String skuName;
        public String subCategory;
        public float prob;

    }

    public native boolean openCamera(int facing);
    public native boolean closeCamera();
    public native boolean setOutputWindow(Surface surface);

    public native Obj[] Detect(Bitmap bitmap);

    public native Obj[] DetectRows(Bitmap bitmap);

    static {
        System.loadLibrary("yolov5ncnn");
    }
}