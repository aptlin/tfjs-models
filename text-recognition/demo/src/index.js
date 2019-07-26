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

import 'bulma/css/bulma.css';

import {load} from '@tensorflow-models/text-detection';
import * as tf from '@tensorflow/tfjs';

const state = {};
const model = {};

const status = (message) => {
  const statusMessage = document.getElementById('status-message');
  statusMessage.innerText = message;
  console.log(message);
};

const howManySecondsFrom = (start) => {
  return ((performance.now() - start) / 1000).toFixed(2);
};

const initializeModel = async () => {
  status('Loading the model...');
  const loadingStart = performance.now();
  const quantizationBytes = Number(getSelectorValue('quantizationBytes'));
  state.quantizationBytes = quantizationBytes;
  await tf.nextFrame();
  model[quantizationBytes] = await load({quantizationBytes});
  const runner = document.getElementById('run');
  runner.onclick = async () => {
    console.log('This is a dummy model');
  };
  status(`Initialised the model in ${
    howManySecondsFrom(loadingStart)} seconds, waiting for input...`);
};

window.onload = initializeModel;
