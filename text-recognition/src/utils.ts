import {config} from './config';
import {QuantizationBytes} from './types';

export const getURL = (quantizationBytes: QuantizationBytes) => {
  return `${config['BASE_PATH']}/${
      ([1, 2].indexOf(quantizationBytes) !== -1) ?
          `quantized/${quantizationBytes}/` :
          ''}psenet/model.json`;
};
