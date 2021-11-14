import { TypeFlags } from "typescript";
import * as tf from '@tensorflow/tfjs'
import { BimConfig, CwConfig, FgsmConfig } from "./attacks";

export class TestContext {
    readyForTest: boolean;
    summary: Array<string>;
    testData: Array<Array<Array<number>>>;
    lables: Array<Array<number>>;
    classNames: Array<string>;
    images: Array<string>;
    progress: number;
    attackInProgress: boolean;

    fgsm: boolean;
    jsma: boolean;
    bim: boolean;
    cw: boolean;
    diffEvol: boolean;

    reports: Map<string, Attack[]>;
    config: Config;
}

export class Config {
    fgsm: FgsmConfig;
    bim: BimConfig;
    cw: CwConfig;
}



export class Attack {

    targetClass: string;
    originalPrediction: string;
    originalConfidence: number;
    orImage: string;

    advImage: string;
    delta: string;
    advPrediction: string;
    advConfidence: number;
    euclidianDistance: number;
    chebyshevDistance: number;
}

export class Source {
    originalImage: tf.Tensor;
    originalConfidence: number;
    originalClassName: string;
}