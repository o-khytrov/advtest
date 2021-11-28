import { AttackStatus } from './testContext';


export class TestCase {

    targetClass: string;
    originalPrediction: string;
    originalConfidence: number;
    orImage: string;

    advImage: string;
    delta: string;
    advPrediction: string;
    advConfidence: number;
    euclidianDistance: number;
    chebyshevDistance: number;
    psnr: number;

    confidenceInOriginalClass: number;
    status: AttackStatus;
}
