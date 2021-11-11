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
    fasm: boolean;
    jsma: boolean;
    bim: boolean;
    cw: boolean;
    reports: Map<string, Attack[]>;
    state: State
    constructor(){
        this.state = State.Blank;
    }
}

export class Settings {

}

export class Fgsm {
    ε: 0.05
}


export class Attack {

    originalPrediction: string;
    originalConfidence: number;
    orImage: string;

    advImage: string;
    advPrediction: string;
    advConfidence: number;
}
export enum State {
    Blank = 0 ,
    Building = 1,
    DomReady = 2
}