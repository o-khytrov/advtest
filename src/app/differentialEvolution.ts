import * as tf from '@tensorflow/tfjs'
export class DifferentialEvolution {

    model: tf.LayersModel;
    populationSize: number;
    mutationRate: number;
    fitnessFunction: Function;
    species: Array<tf.Tensor>;

    constructor(model: tf.LayersModel, populationSize: number, mutationRate: number, fitnessFunction: Function) {
        this.model = model;
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;

    }
    public Initialize(tensor: tf.Tensor) {
        for (let i = 0; i < this.populationSize; i++) {
            this.species.push(tf.clone(tensor))

        }
    }

    crossover(a, b) {
        var child;
        return child;
    }

}
