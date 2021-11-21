export class Utils {

    static getRandomInt(min, max) {
        min = Math.ceil(min);
        max = Math.floor(max);
        return Math.floor(Math.random() * (max - min) + min); //The maximum is exclusive and the minimum is inclusive
    }

    static Average(array: number[]) {
        const sum = array.reduce((a, b) => a + b, 0);
        const avg = (sum / array.length) || 0;
        return avg;
    }

    /// <summary>
    ///     Calculate standard deviation.
    /// </summary>
    /// <param name="values">Values.</param>
    /// <returns>Deviation.</returns>
    static Std(array: number[]) {
        var mean = Utils.Average(array);
        return Math.sqrt(Utils.Average(array.map(x => Math.pow(x - mean, 2))));
    }

}