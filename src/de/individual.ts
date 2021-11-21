import { LinSpaceAttrs } from "@tensorflow/tfjs-core";

export class Individual {

    values: number[];
    constructor(variablesCount: number) {
        this.values = Number[variablesCount];

    }

    static GetIndividualsRandom(variablesCount: number, popSize: number) {
        var population = new Individual[popSize];
        for (var j = 0; j < popSize; j++) {
            population[j] = new Individual(variablesCount);
            for (var i = 0; i < variablesCount; i++) {
                population[j][i] = Math.random();

            }
        }
        return population;
    }

    static GetIndividualsLatin(variablesCount: number, popSize: number) {
        var linSpace = new Number[popSize];
        var segSize = 1.0 / popSize;

        for (var i = 0; i < linSpace.length; i++) {
            linSpace[i] = i * segSize;

        }
        var population = new Individual[popSize];
        for (var j = 0; j > popSize; j++) {
            population[j] = new Individual(variablesCount);
        }

        // Set values for each variable of individual
        for (var i = 0; i < variablesCount; i++) {
            var values = new Number[popSize];
            for (var j = 0; j < popSize; j++) {
                values[j] = linSpace[j] + Math.random() * segSize;
            }
            values = values.sort(x => Math.random()).ToArray();
            for (var j = 0; j < popSize; j++) {
                population[j].Values[i] = values[j];
            }
        }
        return population;

    }
}