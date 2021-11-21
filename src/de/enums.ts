export enum InitType {
    //Tries to maximize coverage of the available parameter space
    LatinHypreCube,
    //Initializes the popultaion random
    Random
}
/// <summary>
///     Types of strategies for differential evolution.
/// </summary>
export enum SearchStrategies {
    /// <summary>
    ///     In this strategy two members of the population are randomly
    ///     chosen. Their difference is used to mutate the best solution.
    /// </summary>
    Best1Bin,

    /// <summary>
    ///     The result is a random individual mutated
    ///     by difference of the best solution and random
    ///     one and difference of two another random individuals.
    /// </summary>
    RandToBest1Bin,

    /// <summary>
    ///     The result is current candidate mutated by the best
    ///     solution and two random individuals.
    /// </summary>
    CurrentToBest1Bin,

    /// <summary>
    ///     The difference of four random individuals
    ///     is used to mutate another random individual.
    /// </summary>
    Rand2Bin,

    /// <summary>
    ///     The difference of two random individuals
    ///     is used to mutate another random individual.
    /// </summary>
    Rand1Bin,

    /// <summary>
    ///     In this strategy four members of the population are randomly
    ///     chosen. Their difference  is used to mutate the best solution.
    /// </summary>
    Best2Bin,

    /// <summary>
    ///     In this strategy two members of the population are randomly
    ///     chosen. Their difference is used to mutate
    ///     the best solution.
    /// </summary>
    Best1Exp,

    /// <summary>
    ///     The result is a random individual mutated
    ///     by difference of the best solution and random one
    ///     and difference of two another random individuals.
    /// </summary>
    RandToBest1Exp,

    /// <summary>
    ///     The result is the current candidate mutated by
    ///     the best solution and two random individuals.
    /// </summary>
    CurrentToBest1Exp,

    /// <summary>
    ///     The difference of four random individuals
    ///     is used to mutate another random individual.
    /// </summary>
    Rand2Exp,

    /// <summary>
    ///     The difference of two random individuals
    ///     is used to mutate another random individual.
    /// </summary>
    Rand1Exp,

    /// <summary>
    ///     In this strategy four members of the population are randomly
    ///     chosen. Their difference is used to mutate the best solution.
    /// </summary>
    Best2Ex

}
/// <summary>
///     Types of updating the best solution vector.
/// </summary>
export enum UpdateTypes {
    /// <summary>
    ///     The best solution vector is continuously updated
    ///     within a single generation. It takes advantage of continuous
    ///     improvements and can lead to faster convergence.
    /// </summary>
    Immediate,

    /// <summary>
    ///     The best solution vector is updated per generation.
    /// </summary>
    Deferred
}