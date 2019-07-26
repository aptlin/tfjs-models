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

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {DummyModelConfig} from './types';
import {getURL} from './utils';

export const load = async (modelConfig: DummyModelConfig = {
  quantizationBytes: 2
}) => {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. ` +
        `If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  if (modelConfig.quantizationBytes) {
    if ([1, 2, 4].indexOf(modelConfig.quantizationBytes) === -1) {
      throw new Error(`Only quantization to 1, 2 and 4 bytes is supported.`);
    }
  } else if (!modelConfig.modelUrl) {
    throw new Error(
        `DummyModel can be constructed either by passing` +
        `the weights URL or by specifying one of the degree of quantization, ` +
        `out of 1, 2 and 4.` +
        `Aborting, since neither has been provided.`);
  }

  const url = getURL(modelConfig.quantizationBytes);
  const graphModel = await tfconv.loadGraphModel(modelConfig.modelUrl || url);
  const dummyModel = new DummyModel(graphModel);
  return dummyModel;
};

export class DummyModel {
  private model: tfconv.GraphModel;
  public constructor(graphModel: tfconv.GraphModel) {
    this.model = graphModel;
  }

  public predict(X: tf.Tensor) {}
  /**
   * Dispose of the tensors allocated by the model.
   * You should call this when you are done with the model.
   */

  public async dispose() {
    (await this.model).dispose();
  }
}
