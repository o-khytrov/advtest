import { LinSpaceAttrs } from "@tensorflow/tfjs-core";
import { Utils } from "src/utils";
import { convertToObject } from "typescript";
import { Bounds } from "./mutation";

export class Individual {

    values: number[];
    Energy: number;
    constructor(variablesCount: number) {
        this.values = new Array<number>(variablesCount);
    }

    static GetIndividualsRandom(variablesCount: number, popSize: number, bounds: Bounds[]) {
        var population = new Array<Individual>(popSize);
        for (var j = 0; j < popSize; j++) {
            population[j] = new Individual(variablesCount);
            for (var i = 0; i < variablesCount; i++) {
                population[j].values[i] = Utils.getRandomInt(bounds[i].min, bounds[i].max);
            }
        }
        return population;
    }

    static GetIndividualsLatin(variablesCount: number, popSize: number) {
        var linSpace = new Array<number>(popSize);
        var segSize = 1.0 / popSize;

        for (var i = 0; i < linSpace.length; i++) {
            linSpace[i] = i * segSize;
        }

        var population = new Array<Individual>(popSize);
        for (var j = 0; j > popSize; j++) {
            population[j] = new Individual(variablesCount);
        }

        // Set values for each variable of individual
        for (var i = 0; i < variablesCount; i++) {
            var values = new Array<number>(popSize);
            for (var j = 0; j < popSize; j++) {
                values[j] = linSpace[j] + Math.random() * segSize;
            }
            values = values.sort(x => Math.random());
            for (var j = 0; j < popSize; j++) {
                population[j].values[i] = values[j];
            }
        }
        return population;

    }
}