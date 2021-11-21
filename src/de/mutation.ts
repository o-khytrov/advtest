import { NumericDataType, rand } from "@tensorflow/tfjs-core";
import { InitType, SearchStrategies, UpdateTypes } from "./enums";

export class DifferentialEvolution {

    mutationFunction: Function;
    callback: Function;
    polish: boolean;
    updateType: UpdateTypes;
    tol: number;
    atol: number;

    binominal: Map<SearchStrategies, Function> = new Map([
        [SearchStrategies.Best1Bin, this.best1],
        [SearchStrategies.RandToBest1Bin, this.randToBest1],
        [SearchStrategies.CurrentToBest1Bin, this.currentToBest1],
        [SearchStrategies.Best2Bin, this.best2],
        [SearchStrategies.Rand2Bin, this.rand2],
        [SearchStrategies.Rand1Bin, this.rand1],

    ]);

    exponential: Map<SearchStrategies, Function> = new Map([
        [SearchStrategies.Best1Exp, this.best1],
        [SearchStrategies.Rand1Exp, this.rand1],
        [SearchStrategies.RandToBest1Exp, this.randToBest1],
        [SearchStrategies.CurrentToBest1Exp, this.currentToBest1],
        [SearchStrategies.Best2Ex, this.best2],
        [SearchStrategies.Rand2Exp, this.rand2],
    ]);



    //Finds the global minimum of a multivariate function.
    optimize(func: Function,
        bounds: Bounds[],
        strategy: SearchStrategies = SearchStrategies.Best1Bin,
        maxiter: number = 1000,
        popsize: number = 15,
        /*
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
        */
        tol: number = 0.01,
        mutation: number,
        recombination: number = 0.7,
        seed,
        callback: Function,
        disp: boolean = false,
        polish: boolean = true,
        init: InitType = InitType.LatinHypreCube,
        atol: number = 0,
        updateType: UpdateTypes = UpdateTypes.Immediate,


    ) {
        if (this.binominal.has(strategy)) {
            this.mutationFunction = this.binominal.get(strategy);
        }
        else if (this.exponential.has(strategy)) {
            this.mutationFunction = this.exponential.get(strategy);
        }
        else {
            throw ("Unsupported strategy");
        }
        this.callback = callback;
        this.polish = polish;
        this.updateType = updateType;
        this.tol = tol;
        this.atol = atol;

    }
    initPopulation() {

    }

    best1() {

    }
    best2() {

    }
    rand2() {

    }
    rand1() {

    }
    randToBest1() {

    }
    currentToBest1() {

    }


}
export class Bounds {
    min: number
    max: number
}
