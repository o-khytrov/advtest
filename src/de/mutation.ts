import { NumericDataType, rand } from "@tensorflow/tfjs-core";
import { InitType, SearchStrategies, UpdateTypes } from "./enums";

export class Mutation {

    mutationFunction: Function;
    callback: Function;
    polish: boolean;
    updateType: UpdateTypes;
    tol: number;
    atol: number;

    static binominal: Map<SearchStrategies, Function> = new Map([
        [SearchStrategies.Best1Bin, Mutation.best1],
        [SearchStrategies.RandToBest1Bin, Mutation.randToBest1],
        [SearchStrategies.CurrentToBest1Bin, Mutation.currentToBest1],
        [SearchStrategies.Best2Bin, Mutation.best2],
        [SearchStrategies.Rand2Bin, Mutation.rand2],
        [SearchStrategies.Rand1Bin, Mutation.rand1],

    ]);

    static exponential: Map<SearchStrategies, Function> = new Map([
        [SearchStrategies.Best1Exp, Mutation.best1],
        [SearchStrategies.Rand1Exp, Mutation.rand1],
        [SearchStrategies.RandToBest1Exp, Mutation.randToBest1],
        [SearchStrategies.CurrentToBest1Exp, Mutation.currentToBest1],
        [SearchStrategies.Best2Ex, Mutation.best2],
        [SearchStrategies.Rand2Exp, Mutation.rand2],
    ]);


    static best1() {

    }
    static best2() {

    }
    static rand2() {

    }
    static rand1() {

    }
    static randToBest1() {

    }
    static currentToBest1() {

    }


}

export class Bounds {
    min: number
    max: number
}
