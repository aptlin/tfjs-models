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

export const config = {
  BASE_PATH: 'https://storage.googleapis.com/gsoc-tfjs/models/efficientnet',
  CROP_PADDING: 32,
  CROP_SIZE: {
    b0: 224,
    b3: 300,
  },
  MEAN_RGB: [0.485, 0.456, 0.406],
  STDDEV_RGB: [0.229, 0.224, 0.225],
};