import { NumericDataType, rand } from "@tensorflow/tfjs-core";
import { InitType, SearchStrategies, UpdateTypes } from "./enums";
import { Individual } from "./individual";

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

    static exponential: Map<SearchStrategies, any> = new Map([
        [SearchStrategies.Best1Exp, Mutation.best1],
        [SearchStrategies.Rand1Exp, Mutation.rand1],
        [SearchStrategies.RandToBest1Exp, Mutation.randToBest1],
        [SearchStrategies.CurrentToBest1Exp, Mutation.currentToBest1],
        [SearchStrategies.Best2Ex, Mutation.best2],
        [SearchStrategies.Rand2Exp, Mutation.rand2],
    ]);


    static best1(samples: number[], population: Individual[], mutationValue: number, candidate: number) {
        let prime = new Individual(population[0].values.length);
        for (var i = 0; i < prime.values.length; i++) {
            prime.values[i] = population[0].values[i] + mutationValue;
                //* (population[samples[0]].values[i]
                //   - population[samples[1]].values[i]);
        }
        return prime;

    }
    static best2(samples: number[], population: Individual[], mutationValue: number, candidate: number) {

        var prime = new Individual(population[0].values.length);
        for (var i = 0; i < prime.values.length; i++) {
            prime.values[i] = population[0].values[i] + mutationValue
                * (population[samples[0]].values[i]
                    + population[samples[1]].values[i]
                    - population[samples[2]].values[i]
                    - population[samples[3]].values[i]);
        }
        return prime;
    }

    static rand2(samples: number[], population: Individual[], mutationValue: number, candidate: number) {
        var prime = new Individual(population[0].values.length);
        for (var i = 0; i < prime.values.length; i++) {
            prime.values[i] = population[samples[0]].values[i]
                + mutationValue
                * (population[samples[1]].values[i]
                    + population[samples[2]].values[i]
                    - population[samples[3]].values[i]
                    - population[samples[4]].values[i]);
        }
        return prime;

    }
    static rand1(samples: number[], population: Individual[], mutationValue: number, candidate: number) {
        var prime = new Individual(population[0].values.length);
        for (var i = 0; i < prime.values.length; i++) {
            prime.values[i] = population[samples[0]].values[i]
                + mutationValue
                * (population[samples[1]].values[i]
                    - population[samples[2]].values[i]);
        }
        return prime;

    }
    static randToBest1(samples: number[], population: Individual[], mutationValue: number, candidate: number) {

        var prime = new Individual(population[0].values.length);
        for (var i = 0; i < prime.values.length; i++) {
            prime.values[i] = population[samples[0]].values[i];
            prime.values[i] += mutationValue * (population[0].values[i]
                - prime.values[i]);
            prime.values[i] += mutationValue
                * (population[samples[1]].values[i]
                    - population[samples[2]].values[i]);
        }
        return prime;
    }
    static currentToBest1(samples: number[], population: Individual[], mutationValue: number, candidate: number) {
        var prime = new Individual(population[0].values.length);
        for (var i = 0; i < prime.values.length; i++) {
            prime.values[i] = population[candidate].values[i]
                + mutationValue
                * (population[0].values[i]
                    - population[candidate].values[i]
                    + population[samples[0]].values[i]
                    - population[samples[1]].values[i]);
        }
        return prime;

    }


}

export class Bounds {
    min: number
    max: number
}
