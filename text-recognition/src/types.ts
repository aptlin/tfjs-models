export type QuantizationBytes = 1|2|4;
export interface DummyModelConfig {
  modelUrl?: string;
  quantizationBytes?: QuantizationBytes;
}