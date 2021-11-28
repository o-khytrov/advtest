import { BimConfig, CwConfig, FgsmConfig, JsmaConfig } from "./attacks";
import { TestCase } from './TestCase';

export class TestContext {
    readyForTest: boolean;
    summary: Map<string, AttackSummary[]>;
    testData: Array<Array<Array<number>>>;
    lables: Array<Array<number>>;
    classNames: Array<string>;
    images: Array<string>;
    progress: number;
    attackInProgress: boolean;

    fgsm: boolean;
    jsma: boolean;
    bim: boolean;
    cw: boolean;
    diffEvol: boolean;

    reports: Map<string, TestCase[]>;
    config: Config;
}

export class Config {
    fgsm: FgsmConfig;
    bim: BimConfig;
    cw: CwConfig;
    jsma: JsmaConfig;
}

export class AttackSummary {

    successRate:number;
    avgChebDistance:number;
}

export enum AttackStatus
{
    Failed,
    Succeeded,
    PartialySucceded
}