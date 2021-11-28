import * as tf from '@tensorflow/tfjs';


export class Source {
    originalImage: tf.Tensor;
    originalConfidence: number;
    originalClassName: string;
}
