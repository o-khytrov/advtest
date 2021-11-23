import { Utils } from "src/utils";
import { convertToObject } from "typescript";
import { InitType, SearchStrategies, UpdateTypes as UpdateType } from "./enums";
import { Individual } from "./individual";
import { Bounds, Mutation } from "./mutation";
import { OptimizeResult } from "./optimizationResult";

export class Solver {

    _fitnessFunction: Function;
    _mutationFunction: Function;
    _searchStrategy: SearchStrategies = SearchStrategies.Best1Bin;
    _popSizeMultiplier: number = 40;
    _bounds: Bounds[];
    _initType: InitType = InitType.Random;
    _maxIterations: number = 20;
    _debug: boolean = true;
    _scaler: number[];
    _updataType: UpdateType = UpdateType.Immediate;

    _mutationMinValue: number = 0.5;
    _mutationMaxValue: number = 1;
    /// <summary>
    ///     Recombination probability.
    /// </summary>
    _recombProbability: number = 0.7;
    _AbsoluteTolerance: number;
    _RelativeTolerance: number;

    start(fitnessFunction: Function, bounds: Bounds[], args) {
        this._fitnessFunction = fitnessFunction;
        this._bounds = bounds;


        if (Mutation.binominal.has(this._searchStrategy)) {
            this._mutationFunction = Mutation.binominal.get(this._searchStrategy);
        }
        else if (Mutation.exponential.has(this._searchStrategy)) {
            this._mutationFunction = Mutation.exponential.get(this._searchStrategy);
        }
        else {
            throw ("Unsupported strategy");
        }

        let causeOfTermination = "Max iterations limit";
        let popSize = Math.max(this._popSizeMultiplier * this._bounds.length, 5);
        let population = new Array<Individual>(popSize);

        switch (+this._initType) {
            case InitType.Random:
                population = Individual.GetIndividualsRandom(
                    this._bounds.length, popSize, bounds);
                break;
            case InitType.LatinHypreCube:
                population = Individual.GetIndividualsLatin(
                    this._bounds.length, popSize);
                break;
            default:
                throw ("Unknown initialization type");

        }

        for (var i = 0; i < popSize; i++) {
            population[i].Energy = this._fitnessFunction(population[i].values);
        }
        population = this.PromoteLowestEnergy(population);


        for (var step = 0; step < this._maxIterations; step++) {
            // Next generation step
            population = this.NextStep(population);
            if (this._debug) {
                console.log(`Evolution step ${step + 1}: "f(best x) = ${population[0].Energy}`);
            }

            // Check the convergence
            if (this.IsSolverConverged(population)) {
                causeOfTermination = "The solver has converged";
                break;
            }
        }
        // Collect results
        var results = new OptimizeResult();

        //results.Convergence = this.GetConvergence(population);
        //results.BestSolution = this.ToScale(population[0].Values);
        results.BestSolution = population[0].values;
        results.Energy = population[0].Energy;
        results.CauseOfTermination = causeOfTermination;
        //results.FunctionEvaluations = this._fitnessFunction.EvaluationCount
        return results;

    }
    /// </summary>
    /// <param name="population">Current population.</param>
    /// <returns>New population.</returns>
    NextStep(population: Individual[]) {

        //var mutationValue = this._mutationMinValue + Math.random()
        //    * (this._mutationMaxValue - this._mutationMinValue);
        var mutationValue = Utils.getRandomInt(1, 3);

        switch (+this._updataType) {
            // Update the best solution immediately
            case UpdateType.Immediate:
                {
                    for (var c = 0; c < population.length; c++) {
                        // Get new individual
                        var trialSolution = this.GetTrialSolution(c, mutationValue, population);

                        // If the energy of the trial is lower than the original
                        // population member, replace it
                        if (trialSolution.Energy < population[c].Energy) {
                            population[c] = trialSolution;

                            // Place this to the first 
                            // if it has the lowest energy
                            if (trialSolution.Energy < population[0].Energy) {
                                population = this.PromoteLowestEnergy(population);
                            }
                        }
                    }
                    break;
                }

            // Update best solution once per generation
            case UpdateType.Deferred:
                {
                    var newPopulation = new Individual[population.length];
                    for (var i = 0; i < newPopulation.Length; i++) {
                        // Get new individual
                        var trialSolution = this.GetTrialSolution(
                            i, mutationValue, population);

                        // Save current solution
                        newPopulation[i] = trialSolution;
                    }

                    // If the energy of the trial is lower than the original
                    // population member, replace it
                    for (var i = 0; i < population.length; i++) {
                        if (newPopulation[i].Energy < population[i].Energy) {
                            population[i] = newPopulation[i];
                        }
                    }
                    population = this.PromoteLowestEnergy(population);
                    break;
                }
        }
        return population;
    }
    /// <summary>
    ///     Create new individual in accordance with the mutation strategy.
    /// </summary>
    /// <param name="individualIndex">Index of candidate.</param>
    /// <param name="mutationValue">Mutation value.</param>
    /// <param name="population">Current population.</param>
    /// <returns>New individual.</returns>
    private GetTrialSolution(individualIndex: number, mutationValue: number, population: Individual[]) {
        // Select random individuals for mutation
        var samples = this.SelectSamples(individualIndex, population.length);

        // Mutate samples 
        var prime = this._mutationFunction(samples, population, mutationValue, individualIndex);

        // Crossover
        var trial = new Individual(population[0].values.length);

        trial.values = Array.from(population[individualIndex].values);

        //var fillPoint = _random.Next(0, trial.values.length);
        var fillPoint = Utils.getRandomInt(0, trial.values.length);

        if (Mutation.binominal.has(this._searchStrategy)) {
            for (var i = 0; i < trial.values.length; i++) {
                trial.values[i] = Math.random() < this._recombProbability
                    ? prime.values[i]
                    : population[individualIndex].values[i];
            }
            trial.values[fillPoint] = prime.values[fillPoint];
        }
        else if (Mutation.exponential.has(this._searchStrategy)) {
            var i = 0;
            while (i < trial.values.length
                && Math.random() < this._recombProbability) {
                trial.values[fillPoint] = prime.values[fillPoint];
                fillPoint = (fillPoint + 1) % trial.values.length;
                i++;
            }
        }

        // Ensure the trial is in limits
        //trial = this.EnsureConstraint(trial);

        // Get energy of the trial
        trial.Energy = this._fitnessFunction(trial.values);
        return trial;
    }
    /// <summary>
    ///     Make sure the parameters lie between the limits.
    /// </summary>
    /// <param name="trial">Trial individual.</param>
    /// <returns>Checked individual.</returns>
    private EnsureConstraint(trial: Individual) {
        for (var i = 0; i < trial.values.length; i++) {
            if ((trial.values[i] < 0) || (trial.values[i] > 1)) {
                trial.values[i] = Math.random()
            }
        }
        return trial;
    }
    #endregion
    /// <summary>
    ///     Select random samples for mutation.
    ///     The candidate index can not be among samples.
    /// </summary>
    /// <param name="candidateIndex">Index of candidate.</param>
    /// <param name="populationCount">Population count.</param>
    /// <returns>Indexes of samples.</returns>
    SelectSamples(candidateIndex: number, populationCount: number) {
        var indexes = new Array<number>(populationCount - 1);
        var counter = 0;
        for (var i = 0; i < populationCount; i++) {
            if (candidateIndex != i) {
                indexes[counter] = i;
                counter++;
            }
        }
        return indexes.sort(x => Math.random());
    }

    PromoteLowestEnergy(population: Individual[]) {
        var bestIndex = 0;
        var bestEnergy = population[0].Energy;
        for (var index = 1; index < population.length; index++) {
            if (population[index].Energy < bestEnergy) {
                bestIndex = index;
                bestEnergy = population[index].Energy;
            }
        }
        var temp = population[0];
        population[0] = population[bestIndex];
        population[bestIndex] = temp;
        return population;
    }

    /*
    /// <summary>
    ///     Scale from (0, 1) range to real values.
    /// </summary>
    /// <param name="values">Values.</param>
    /// <returns>Scaled values.</returns>
    ToScale(values: number[]) {
        var newValues = new Number[values.length];
        for (var index = 0; index < values.length; index++) {
            newValues[index] = this._scaler[index].mean
                + this._scaler[index].absDifference *
                (values[index] - 0.5);
        }
        return newValues;
    }

    /// <summary>
    ///     The standard deviation of the individual
    ///     energies divided by their mean.
    /// </summary>
    /// <param name="population">Population.</param>
    /// <returns>Convergence of the population.</returns>
    GetConvergence(population: Individual[]) {
        var individEnergies = population.map(x => x.Energy);
        return MathFunctions.Std(individEnergies)
            / Math.abs(individEnergies.Average() + double.Epsilon);
    }

    */

    IsSolverConverged(population: Individual[]) {
        var individEnergies = population.map(x => x.Energy);
        return Utils.Std(individEnergies) <= this._AbsoluteTolerance + this._RelativeTolerance * Math.abs(Utils.Average(individEnergies));
    }
}