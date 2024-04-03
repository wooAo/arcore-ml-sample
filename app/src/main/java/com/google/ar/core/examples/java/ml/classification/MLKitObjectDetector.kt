/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ar.core.examples.java.ml.classification

import android.app.Activity
import android.graphics.BitmapFactory
import android.media.Image
import android.nfc.Tag
import android.util.Log
import com.google.ar.core.examples.java.ml.MainActivity
import com.google.ar.core.examples.java.ml.classification.utils.ImageUtils
import com.google.ar.core.examples.java.ml.classification.utils.VertexUtils.rotateCoordinates
//import com.google.mlkit.common.model.LocalModel
//import com.google.mlkit.vision.common.InputImage
//import com.google.mlkit.vision.objects.ObjectDetection
//import com.google.mlkit.vision.objects.custom.CustomObjectDetectorOptions
//import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
//import kotlinx.coroutines.tasks.asDeferred

/**
 * Analyzes an image using ML Kit.
 */
class MLKitObjectDetector(context: Activity) : ObjectDetector(context) {

  val TAG = "MLKitObjectDetector"
  // To use a custom model, follow steps on https://developers.google.com/ml-kit/vision/object-detection/custom-models/android.
//   val model = LocalModel.Builder().setAssetFilePath("inception_v4_1_metadata_1.tflite").build()
//   val builder = CustomObjectDetectorOptions.Builder(model)

  // For the ML Kit default model, use the following:
//  val builder = ObjectDetectorOptions.Builder()

//  private val options = builder
//    .setDetectorMode(CustomObjectDetectorOptions.SINGLE_IMAGE_MODE)
//    .enableClassification()
//    .enableMultipleObjects()
//    .build()
//  private val detector = ObjectDetection.getClient(options)

  override suspend fun analyze(image: Image, imageRotation: Int, row: Boolean): List<DetectedObjectResult> {
    // `image` is in YUV (https://developers.google.com/ar/reference/java/com/google/ar/core/Frame#acquireCameraImage()),
    val convertYuv = convertYuv(image)
    // The model performs best on upright images, so rotate it.
    val rotatedImage = ImageUtils.rotateBitmap(convertYuv, imageRotation)
    val objs = if (row) {
      MainActivity.yoloV5Ncnn.DetectRows(rotatedImage)
    }else{
      MainActivity.yoloV5Ncnn.Detect(rotatedImage)
    }
    return objs.mapNotNull { obj ->
      val bestLabel = "BOX"
      val topleft = obj.x.toInt() to obj.y.toInt()
      val coords = obj.x.toInt()+obj.w.toInt()/2 to obj.y.toInt()+obj.h.toInt()/2
      val rotatedCoordinates = coords.rotateCoordinates(rotatedImage.width, rotatedImage.height, imageRotation)
      Log.d(TAG, "Rotated coordinates: $rotatedCoordinates")
      val topleftCoords = topleft.rotateCoordinates(rotatedImage.width, rotatedImage.height, imageRotation)
      val boundingBox = (topleftCoords.first to (topleftCoords.second-obj.w).toInt()) to (obj.h.toInt() to obj.w.toInt())
      DetectedObjectResult(1f, bestLabel, rotatedCoordinates, boundingBox)
    }

//    val inputImage = InputImage.fromBitmap(rotatedImage, 0)
//    val mlKitDetectedObjects = detector.process(inputImage).asDeferred().await()
//    return mlKitDetectedObjects.mapNotNull { obj ->
//      val bestLabel = obj.labels.maxByOrNull { label -> label.confidence } ?: return@mapNotNull null
//      val coords = obj.boundingBox.exactCenterX().toInt() to obj.boundingBox.exactCenterY().toInt()
//      val rotatedCoordinates = coords.rotateCoordinates(rotatedImage.width, rotatedImage.height, imageRotation)
////      Log.d(TAG, "Detected object: ${obj.labels}, coords: ${obj.boundingBox}")
//      DetectedObjectResult(bestLabel.confidence, bestLabel.text, rotatedCoordinates)
//    }
  }

//  @Suppress("USELESS_IS_CHECK")
//  fun hasCustomModel() = builder is CustomObjectDetectorOptions.Builder
  @Suppress("USELESS_IS_CHECK")
  fun hasCustomModel() = true
}