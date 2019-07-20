/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';

import {config} from './config';
import {Point} from './geometry';
import {minAreaRect} from './minAreaRect';
import {progressiveScaleExpansion} from './progressiveScaleExpansion';
import {Box, QuantizationBytes, TextDetectionInput, TextDetectionOptions} from './types';

export const getURL = (quantizationBytes: QuantizationBytes) => {
  return `${config['BASE_PATH']}/${
      quantizationBytes ? `quantized/${quantizationBytes}/` :
                          ''}psenet/model.json`;
};

export const detect =
    (kernelScores: tf.Tensor3D, originalHeight: number, originalWidth: number,
     textDetectionOptions: TextDetectionOptions = {
       minKernelArea: config['MIN_KERNEL_AREA'],
       minScore: config['MIN_SCORE'],
       maxSideLength: config['MAX_SIDE_LENGTH']
     }): Box[] => {
      if (!textDetectionOptions.minKernelArea) {
        textDetectionOptions.minKernelArea = config['MIN_KERNEL_AREA'];
      }
      if (!textDetectionOptions.minScore) {
        textDetectionOptions.minScore = config['MIN_SCORE'];
      }
      if (!textDetectionOptions.maxSideLength) {
        textDetectionOptions.maxSideLength = config['MAX_SIDE_LENGTH'];
      }
      const {minKernelArea, minScore, maxSideLength} = textDetectionOptions;

      const [height, width, numOfKernels] = kernelScores.shape;
      const kernelScoreData = kernelScores.arraySync();
      tf.dispose(kernelScores);

      const kernels = new Array<tf.Tensor2D>();

      for (let kernelIdx = 0; kernelIdx < numOfKernels; ++kernelIdx) {
        const kernelBuffer = tf.buffer([height, width], 'float32');

        for (let rowIdx = 0; rowIdx < height; ++rowIdx) {
          for (let columnIdx = 0; columnIdx < width; ++columnIdx) {
            kernelBuffer.set(
                kernelScoreData[rowIdx][columnIdx][kernelIdx], rowIdx,
                columnIdx);
          }
        }
        const kernelRawTensor = kernelBuffer.toTensor();
        const kernel = tf.tidy(
            () => (((kernelRawTensor.sub(tf.scalar(1))).sign()).add(1))
                      .div(tf.scalar(2)));
        tf.dispose(kernelRawTensor);
        kernels.push(kernel as tf.Tensor2D);
      }

      if (kernels.length > 0) {
        const scoreBuffer = tf.buffer([height, width], 'float32');
        for (let rowIdx = 0; rowIdx < height; ++rowIdx) {
          for (let columnIdx = 0; columnIdx < width; ++columnIdx) {
            scoreBuffer.set(
                kernelScoreData[rowIdx][columnIdx][0], rowIdx, columnIdx);
          }
        }
        const scoreRawTensor = scoreBuffer.toTensor();
        const score = scoreRawTensor.sigmoid();
        const scoreData = score.arraySync() as number[][];
        tf.dispose([scoreRawTensor, score]);

        const text = kernels[0].clone();

        for (let kernelIdx = 0; kernelIdx < numOfKernels; ++kernelIdx) {
          const kernel = kernels[kernelIdx];
          kernels[kernelIdx] = kernel.mulStrict(text);
          tf.dispose(kernel);
        }

        tf.dispose(text);

        const [heightScalingFactor, widthScalingFactor] =
            computeScalingFactors(originalHeight, originalWidth, maxSideLength);
        const {segmentationMapBuffer, recognizedLabels} =
            progressiveScaleExpansion(kernels, minKernelArea);
        tf.dispose(kernels);
        if (recognizedLabels.size === 0) {
          return [];
        }
        const labelScores:
            {[label: number]: {count: number, totalScore: number}} = {};
        for (let rowIdx = 0; rowIdx < height; ++rowIdx) {
          for (let columnIdx = 0; columnIdx < width; ++columnIdx) {
            const label = segmentationMapBuffer.get(rowIdx, columnIdx);
            if (recognizedLabels.has(label)) {
              const score = scoreData[rowIdx][columnIdx];
              if (!labelScores[label]) {
                labelScores[label] = {count: 1, totalScore: score};
              }

              const {count, totalScore} = labelScores[label];
              labelScores[label] = {
                count: count + 1,
                totalScore: totalScore + score
              };
            }
          }
        }
        const targetHeight = Math.round(originalHeight * heightScalingFactor);
        const targetWidth = Math.round(originalWidth * widthScalingFactor);
        const resizedSegmentationMap = tf.tidy(() => {
          const processedSegmentationMap =
              segmentationMapBuffer.toTensor().expandDims(2) as tf.Tensor3D;
          return tf.image
              .resizeNearestNeighbor(
                  processedSegmentationMap, [targetHeight, targetWidth])
              .squeeze([2]);
        });
        const resizedSegmentationMapData =
            resizedSegmentationMap.arraySync() as number[][];
        tf.dispose(resizedSegmentationMap);

        const points: {[label: number]: Point[]} = {};
        for (let rowIdx = 0; rowIdx < targetHeight; ++rowIdx) {
          for (let columnIdx = 0; columnIdx < targetWidth; ++columnIdx) {
            const label = resizedSegmentationMapData[rowIdx][columnIdx];
            if (recognizedLabels.has(label)) {
              const {totalScore, count} = labelScores[label];
              if (totalScore / count >= minScore) {
                if (!points[label]) {
                  points[label] = [];
                }
                points[label].push(new Point(columnIdx, rowIdx));
              }
            }
          }
        }
        const boxes: Box[] = [];
        const clip = (size: number, edge: number) =>
            (size > edge ? edge : size);
        Object.keys(points).forEach((labelStr) => {
          const label = Number(labelStr);
          const box = minAreaRect(points[label]);
          for (let pointIdx = 0; pointIdx < box.length; ++pointIdx) {
            const point = box[pointIdx];
            const scaledX = clip(point.x / widthScalingFactor, originalWidth);
            const scaledY = clip(point.y / heightScalingFactor, originalHeight);
            box[pointIdx] = new Point(scaledX, scaledY);
          }
          boxes.push(box);
        });
        return boxes;
      } else {
        tf.dispose(kernels);
      }
      return [];
    };

export const computeScalingFactors =
    (height: number, width: number, clippingEdge: number): [number, number] => {
      const maxSide = Math.max(width, height);
      const ratio = maxSide > clippingEdge ? clippingEdge / maxSide : 1;

      const getScalingFactor = (side: number) => {
        const roundedSide = Math.round(side * ratio);
        return (roundedSide % 32 === 0 ?
                    roundedSide :
                    (Math.floor(roundedSide / 32) + 1) * 32) /
            side;
      };

      const heightScalingRatio = getScalingFactor(height);
      const widthScalingRatio = getScalingFactor(width);
      return [heightScalingRatio, widthScalingRatio];
    };

export const cropAndResize = (input: TextDetectionInput,
                              clippingEdge =
                                  config['MAX_SIDE_LENGTH']): tf.Tensor3D => {
  return tf.tidy(() => {
    const image: tf.Tensor3D = (input instanceof tf.Tensor ?
                                    input :
                                    tf.browser.fromPixels(
                                        input as ImageData | HTMLImageElement |
                                        HTMLCanvasElement | HTMLVideoElement))
                                   .toFloat();

    const [height, width] = image.shape;
    const [heightScalingFactor, widthScalingFactor] =
        computeScalingFactors(height, width, clippingEdge);
    const targetHeight = Math.round(height * heightScalingFactor);
    const targetWidth = Math.round(width * widthScalingFactor);
    const processedImage =
        tf.image.resizeBilinear(image, [targetHeight, targetWidth]);

    return processedImage;
  });
};
