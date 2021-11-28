import * as tf from '@tensorflow/tfjs'

export class Metrics {
    //Peek signal to noise ratio
    public static psnr(a: tf.Tensor, b: tf.Tensor) {
        let mseTensor = tf.tidy(() => {
            const mse = tf.losses.meanSquaredError(a, b);
            return mse.dataSync()
        })
        let mse = mseTensor[0];
        if(mse==0) return 0;

        return 10 * Math.log10(1 / mse);
    }
    public static chebyshevDistanse(a: tf.Tensor, b: tf.Tensor) {
        let c = tf.sub(a, b);
        let d = tf.abs(c).max().dataSync()[0];
        c.dispose();
        return d;
    }
    public static euclidianDistance(a: tf.Tensor, b: tf.Tensor) {
        // calculate euclidian distance between two arrays
        let distTensor = tf.tidy(() => {
            const distance = tf.squaredDifference(a, b).sum().sqrt();
            return distance.dataSync()
        })
        return distTensor[0];
    }
}