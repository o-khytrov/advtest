import { TypeFlags } from "typescript";
import * as tf from '@tensorflow/tfjs'

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
    reports: Map<string, Attack[]>;
    state: State
    config: Config;
    constructor() {
        this.state = State.Blank;
    }
}

export class Config {
    fgsm: FgsmConfig;
    bim: BimConfig;
}

export class FgsmConfig {
    epsilon: number;
}
export class BimConfig {
    epsilon: number;
    alpha: number;
    iterations: number;
}


export class Attack {

    originalPrediction: string;
    originalConfidence: number;
    orImage: string;

    advImage: string;
    delta: string;
    advPrediction: string;
    advConfidence: number;
    euclidianDistance: number;
}
export enum State {
    Blank = 0,
    Building = 1,
    DomReady = 2
}