/// <summary>
///     Implementation of optimization results.
/// </summary>
export class OptimizeResult {
    /// <summary>
    ///     Values of the best individual.
    /// </summary>
    BestSolution: number[];

    /// <summary>
    ///     Cause of solver termination.
    /// </summary>
    CauseOfTermination: string;

    /// <summary>
    ///     Fitness function value with the best solution.
    /// </summary>
    Energy: number;

    /// <summary>
    ///     Convergence of the solver.
    /// </summary>
    Convergence: number;

    /// <summary>
    ///     Time of optimizations process.
    /// </summary>
    OptimizationTime: number;

    /// <summary>
    ///     Count of fitness function evaluations.
    /// </summary>
    FunctionEvaluations: number;
}