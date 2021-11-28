import { ChartColor, ChartDataSets } from "chart.js";
import { Label } from "ng2-charts";
import { BimConfig, CwConfig, FgsmConfig, JsmaConfig } from "./attacks";
import { TestCase } from './TestCase';

export class TestContext {
    readyForTest: boolean;
    summary: Map<string, AttackSummary>;
    reports: Map<string, TestCase[]>;
    classNames: Array<string>;
    images: Array<string>;
    progress: number;
    attackInProgress: boolean;
    fgsm: boolean;
    jsma: boolean;
    bim: boolean;
    cw: boolean;
    diffEvol: boolean;

    config: Config;
}

export class Config {
    fgsm: FgsmConfig;
    bim: BimConfig;
    cw: CwConfig;
    jsma: JsmaConfig;
}

export class AttackSummary {

    public ChebDistChartData: ChartDataSets[] = [

        { data: [], label: 'Chebyshov distance' }

    ];
    public EuclDistChartData: ChartDataSets[] = [

        { data: [], label: 'Euclidian distance' }

    ];
    public PsnrDistChartData: ChartDataSets[] = [
        {
            data: [], label: 'PSNR',
            backgroundColor: "rgba(0, 100, 255, 0.3)",
            borderColor: "blue"
        }

    ];

    public EpsilonChart: ChartDataSets[] = [
        {
            data: [], label: 'ε',
            backgroundColor: "rgba(0, 100, 255, 0.3)",
            borderColor: "blue"
        }
    ];

    public orConfChartData: ChartDataSets[] = [

        {
            data: [], label: 'Confidence in original class',
            backgroundColor: "rgba(255, 100, 70, 0.3)",
            borderColor: "rgb(255, 100, 70)"
        }

    ];
    public SuccessRate: ChartDataSets[] = [
        {
            data: [], label: 'Success rate',
            backgroundColor: "rgba(41, 184, 71, 0.3)",
            borderColor: "green"
        }

    ];

    public lineChartLabels: Label[] = [];
}

export enum AttackStatus {
    Failed,
    Succeeded,
    PartialySucceded
}