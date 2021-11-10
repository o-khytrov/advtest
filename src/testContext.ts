import { TypeFlags } from "typescript";
import * as tf from '@tensorflow/tfjs'

export class TestContext {
    summary: Array<string>;
    testData: Array<Array<Array<number>>>;
    lables: Array<Array<number>>;
    classNames: Array<string>;
    images: Array<string>;
}

export class Settings {

}

export class Fgsm {
    ε: 0.05
}