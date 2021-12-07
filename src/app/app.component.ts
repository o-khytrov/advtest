import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { AttackStatus, AttackSummary, Config, TestContext } from 'src/testContext';
import { Source } from "src/Source";
import { TestCase } from "src/TestCase";
import { AttackResult, Attacks, BimConfig, CwConfig, FgsmConfig, JsmaConfig } from 'src/attacks';
import { Metrics } from 'src/metrics';
import { accuracy } from '@tensorflow/tfjs-vis/dist/util/math';
import { Utils } from 'src/utils';
import { Solver } from 'src/de/solver';
import { KeyValue } from '@angular/common';
import { Label } from 'ng2-charts';
import { ChartDataSets, ChartType } from 'chart.js';
import { ɵELEMENT_PROBE_PROVIDERS } from '@angular/platform-browser';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {

  @ViewChild('i_model') i_model: ElementRef;
  @ViewChild('i_weights') i_weights: ElementRef;
  @ViewChild('i_test_data') i_test_data: ElementRef;
  @ViewChild('i_labels') i_labels: ElementRef;
  @ViewChild('i_class_names') i_class_names: ElementRef;

  canvas = document.createElement('canvas');
  title = 'advtest';
  testContext: TestContext;
  model: tf.LayersModel;
  dataset: any;
  savedModels: Set<string>;
  showChart: boolean;

  public lineChartOptions = {
    responsive: true,
  };

  public lineChartLegend = true;
  public lineChartType: ChartType = 'line';
  public lineChartPlugins = [];

  constructor() {
    this.InitializeTestContext();
  }

  private InitializeTestContext() {
    this.testContext = new TestContext();
    this.testContext.config = new Config();

    this.testContext.config.fgsm = new FgsmConfig();
    this.testContext.config.fgsm.epsilon = 0.1;

    this.testContext.config.jsma = new JsmaConfig();
    this.testContext.config.jsma.epsilon = 75;


    this.testContext.config.bim = new BimConfig();
    this.testContext.config.bim.epsilon = 0.1;
    this.testContext.config.bim.alpha = 0.01;
    this.testContext.config.bim.iterations = 30;

    this.testContext.config.cw = new CwConfig();
    this.testContext.config.cw.successRate = 5;
    this.testContext.config.cw.confidenceRate = 1;
    this.testContext.config.cw.learningRate = 0.1;
    this.testContext.config.cw.iterations = 100;
    this.testContext.summary = new Map<string, AttackSummary>();

  }

  async ngOnInit() {

    this.savedModels = new Set<string>();
    for (var key in localStorage) {
      if (key.startsWith("tensorflowjs_models")) {
        var modelName = key.split('/')[1];
        this.savedModels.add(modelName);
      }
    }

  }

  onFileChange(event) {
    event.target.nextSibling.innerText = event.target.files[0].name;
  }

  async showModel() {
    const surface = { name: 'Model Summary', tab: 'Model Inspection' };
    tfvis.show.modelSummary(surface, this.model);
    tfvis.show.layer(surface, this.model.getLayer(undefined, 0));
  }

  async loadModel() {

    this.model = await tf.loadLayersModel(tf.io.browserFiles([this.i_model.nativeElement.files[0], this.i_weights.nativeElement.files[0]]));
    this.showModel();
    // Monkey patch the mobilenet object to have a predict() method like a normal tf.LayersModel
    this.model.predict = function (img) {
      const logits1001 = this.predict(img);
      return logits1001.slice([0, 1], [-1, 1000]).softmax();

    }

    await this.loadDataset();

  }
  async loadDataset() {
    // Load data in form [{xs: x0_tensor, ys: y0_tensor}, {xs: x1_tensor, ys: y1_tensor}, ...]
    let x, y;
    let loadingX = this.readFile(this.i_test_data.nativeElement.files[0]).then(arr => x = tf.data.array(arr as []).batch(1));
    let loadingY = this.readFile(this.i_labels.nativeElement.files[0]).then(arr => y = tf.data.array(arr as []).batch(1));
    let loadingNames = this.readFile(this.i_class_names.nativeElement.files[0]).then(x => this.testContext.classNames = x as string[]);
    let loadingData = Promise.all([loadingX, loadingY, loadingNames]).then(() => tf.data.zip([x, y]).toArray()).then(ds => this.dataset = ds.map(e => { return { xs: e[0], ys: e[1] } }))
    loadingData.then(async x => {
      this.testContext.readyForTest = true;
    })
  }
  async save() {
    var dataset = [];
    for (let i = 0; i < this.dataset.length; i++) {
      dataset[i] = {
        xs: { data: this.dataset[i].xs.arraySync(), shape: this.dataset[i].xs.shape },
        ys: { data: this.dataset[i].ys.arraySync(), shape: this.dataset[i].ys.shape }
      }
    }
    await this.model.save(`localstorage://${this.model.name}`);
    localStorage.setItem(`tensorflowjs_models/${this.model.name}/class_names`, JSON.stringify(this.testContext.classNames));
    localStorage.setItem(`tensorflowjs_models/${this.model.name}/dataset`, JSON.stringify(dataset));

  }
  async loadModelFromLocalStorage(modelName: string) {
    this.model = await tf.loadLayersModel(`localstorage://${modelName}`);
    var dataset = JSON.parse(localStorage.getItem(`tensorflowjs_models/${this.model.name}/dataset`));
    this.testContext.classNames = JSON.parse(localStorage.getItem(`tensorflowjs_models/${this.model.name}/class_names`));
    this.dataset = [];
    for (let i = 0; i < dataset.length; i++) {
      this.dataset[i] = {
        xs: tf.tensor(dataset[i].xs.data, dataset[i].xs.shape),
        ys: tf.tensor(dataset[i].ys.data, dataset[i].ys.shape),
      }
    }

    //this.showModel();
    this.testContext.readyForTest = true;
  }

  readFile(file: File) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onerror = reject;
      fr.onload = function () {
        resolve(JSON.parse(fr.result.toString()));
      }
      fr.readAsText(file);
    });
  }

  async getDataUrl(img) {
    var shape = img.shape.slice(1);

    if (shape.length == 1)//grayscale array
    {
      let side = Math.sqrt(shape[0]);
      shape = [side, side, 1]
    }

    await tf.browser.toPixels(img.reshape(shape), this.canvas);
    return this.canvas.toDataURL();
  }

  async runUntargeted(attack, config) {
    for (let i = 0; i < this.dataset.length; i++) {
      let img = this.dataset[i].xs;
      let lbl = this.dataset[i].ys;
      let p_original = (this.model.predict(img) as tf.Tensor);
      console.log(p_original.dataSync());
      let conf_original = p_original.max(1).dataSync()[0];
      let cl_original = p_original.argMax(1).dataSync()[0];

      let source = new Source();

      source.originalClassIndex = cl_original;
      source.originalClassName = this.testContext.classNames[cl_original];
      source.originalConfidence = conf_original;
      source.originalImage = img;

      let attackResult = tf.tidy(() => attack(this.model, img, lbl, config));

      await this.fillReport(source, attack.name, i, attackResult);
      this.reportProgres(i + 1, this.dataset.length);
    }

  }
  async runTargeted(attack, config) {

    let a = 0;
    for (let i = 0; i < this.dataset.length; i++) {
      let img = this.dataset[i].xs;
      let lbl = this.dataset[i].ys;
      let p_original = (this.model.predict(img) as tf.Tensor);
      let conf_original = p_original.max(1).dataSync()[0];
      let cl_original = p_original.argMax(1).dataSync()[0];
      let source = new Source();
      source.originalClassIndex = cl_original;
      source.originalClassName = this.testContext.classNames[cl_original];
      source.originalConfidence = conf_original;
      source.originalImage = img;

      for (let j = 0; j < this.dataset.length; j++) {
        if (j == i) {
          continue;//skip same class
        }
        let targetLbl = this.dataset[j].ys;
        let targetLblIndex = targetLbl.argMax(1).dataSync()[0];

        source.targetClassIndex = targetLblIndex;

        let attackResult = tf.tidy(() => attack(this.model, img, lbl, targetLbl, config));

        await this.fillReport(source, attack.name, a, attackResult);
        this.testContext.reports.get(attack.name)[a].targetClass =
          this.testContext.classNames[targetLblIndex];

        a++;
        this.reportProgres(a + 1, this.dataset.length * this.dataset.length);
      }
    }
    //this.BuildConfusionMatrix(attack.name);
  }
  private BuildConfusionMatrix(attackName: string) {
    let confusionMatrix = [];
    let a = 0;
    for (let i = 0; i < this.dataset.length; i++) {
      let row = [];
      for (let j = 0; j < this.dataset.length; j++) {
        if (i == j) {
          row.push(1);
        }
        else {
          let testCase = this.testContext.reports.get(attackName)[a];
          row.push(testCase.advConfidence);
          a++;
        }
      }
      confusionMatrix.push(row);
    }
    const data = { values: confusionMatrix, labels: this.testContext.classNames };

    // Render to visor
    const surface = { name: `Confusion Matrix ${attackName}`, tab: 'Charts' };
    tfvis.render.confusionMatrix(surface, data);
  }

  async fillReport(source: Source, attackName: string, a: number, attackResult: AttackResult) {
    let p_adversarial = (this.model.predict(attackResult.advImg) as tf.Tensor);
    let conf_adv = p_adversarial.max(1).dataSync()[0];
    let cl_adv = p_adversarial.argMax(1).dataSync()[0];
    let confidenceInOriginalClass = p_adversarial.dataSync()[source.originalClassIndex];

    var orImageB64 = await this.getDataUrl(source.originalImage);
    var advImgB64 = await this.getDataUrl(attackResult.advImg);
    let testCase = this.testContext.reports.get(attackName)[a];

    let dist = tf.sub(attackResult.advImg, source.originalImage);
    let delta = tf.abs(dist);

    attackResult.delta = delta;
    testCase.confidenceInOriginalClass = confidenceInOriginalClass;


    testCase.originalPrediction = source.originalClassName;
    testCase.originalConfidence = source.originalConfidence;
    testCase.orImage = orImageB64;
    testCase.advConfidence = conf_adv;
    testCase.advPrediction = this.testContext.classNames[cl_adv];
    testCase.euclidianDistance = Metrics.euclidianDistance(attackResult.advImg, source.originalImage);
    testCase.chebyshevDistance = Metrics.chebyshevDistanse(attackResult.advImg, source.originalImage);
    testCase.psnr = Metrics.psnr(source.originalImage, attackResult.advImg);
    testCase.advImage = advImgB64;

    if (attackResult.delta) {
      var deltaB64 = await this.getDataUrl(attackResult.delta);
      this.testContext.reports.get(attackName)[a].delta = deltaB64;
    }
    if (testCase.targetClass) {
      if (cl_adv == source.targetClassIndex) {

        testCase.status = AttackStatus.Succeeded;
      } else
        if (cl_adv != source.originalClassIndex) {
          testCase.status = AttackStatus.PartialySucceded;
        }
        else
          testCase.status = AttackStatus.Failed;

    }
    else {
      if (cl_adv != source.originalClassIndex) {
        testCase.status = AttackStatus.Succeeded;
      }
      else
        testCase.status = AttackStatus.Failed;
    }

  }

  async runTest() {
    this.testContext.attackInProgress = true;
    this.testContext.progress = 0;
    this.testContext.reports = new Map<string, TestCase[]>();
    let attackName = "";
    let epsilon = 0;

    if (this.testContext.fgsm) {

      epsilon = this.testContext.config.fgsm.epsilon;
      if (this.testContext.config.fgsm.targeted) {

        attackName = Attacks.fgsmTargeted.name;
        this.BuildReportForTargeted(attackName);
        await this.runTargeted(Attacks.fgsmTargeted, this.testContext.config.fgsm);

      }
      else {
        attackName = Attacks.fgsm.name;
        this.BuildReport(attackName);
        await this.runUntargeted(Attacks.fgsm, this.testContext.config.fgsm);

      }
    }

    if (this.testContext.bim) {

      epsilon = this.testContext.config.bim.epsilon;
      if (this.testContext.config.bim.targeted) {

        attackName = Attacks.bimTargeted.name;
        this.BuildReportForTargeted(attackName)
        await this.runTargeted(Attacks.bimTargeted, this.testContext.config.bim);
      }
      else {
        attackName = Attacks.bim.name;
        this.BuildReport(attackName);
        await this.runUntargeted(Attacks.bim, this.testContext.config.bim);

      }
    }

    if (this.testContext.jsma) {

      epsilon = this.testContext.config.jsma.epsilon;
      attackName = Attacks.jsmaOnePixel.name;
      this.BuildReportForTargeted(attackName)
      await this.runTargeted(Attacks.jsmaOnePixel, this.testContext.config.jsma);
    }

    if (this.testContext.cw) {

      epsilon = this.testContext.config.cw.successRate;
      attackName = Attacks.cw.name;
      this.BuildReportForTargeted(attackName);
      await this.runTargeted(Attacks.cw, this.testContext.config.cw);
    }

    if (this.testContext.diffEvol) {
      attackName = Attacks.DifferentialEvolution.name;
      this.BuildReport(attackName);
      await this.runUntargeted(Attacks.DifferentialEvolution, this.testContext.config.fgsm);
    }
    var summary = this.testContext.summary.get(attackName);
    if (!summary) {
      summary = new AttackSummary();
      this.testContext.summary.set(attackName, new AttackSummary());
      summary = this.testContext.summary.get(attackName);
    }

    summary.EpsilonChart[0].data.push(epsilon);

    this.summary();
    this.testContext.attackInProgress = false;

  }
  private BuildReportForTargeted(attackName) {
    var attacks = new Array<TestCase>();
    for (var i = 0; i < this.dataset.length; i++) {
      for (var j = 0; j < this.dataset.length; j++) {
        if (j != i) attacks.push(new TestCase());
      }
    }
    this.testContext.reports.set(attackName, attacks);
  }

  private BuildReport(attackName) {
    var attacks = new Array<TestCase>();
    for (var i = 0; i < this.dataset.length; i++) {
      attacks.push(new TestCase());
    }
    this.testContext.reports.set(attackName, attacks);
  }

  reportProgres(progres, total) {
    this.testContext.progress = progres / total * 100;
  }
  toggleVisor() {
    tfvis.visor().toggle();
  }

  summary() {
    for (let attackKey of this.testContext.reports.keys()) {
      let sumChebDistance = 0;
      let sumEuclDistance = 0;
      let sumPSNR = 0;
      let sumOrConf = 0;
      let succeededAttacks = 0;

      let attackReport = this.testContext.reports.get(attackKey);

      for (let a = 0; a < attackReport.length; a++) {
        let attack = attackReport[a];
        sumChebDistance += attack.chebyshevDistance;
        sumEuclDistance += attack.euclidianDistance;
        sumPSNR += attack.psnr;
        sumOrConf += attack.confidenceInOriginalClass;
        if (attack.status == AttackStatus.Succeeded) {
          succeededAttacks++;
        }
      }
      let avgChebDistance = sumChebDistance / attackReport.length;
      let avgEuclDistance = sumEuclDistance / attackReport.length;
      let avgPsnr = sumPSNR / attackReport.length;
      let avgOrConf = sumOrConf / attackReport.length;
      let successRate = Math.floor(succeededAttacks / attackReport.length * 100);

      var summary = this.testContext.summary.get(attackKey);
      summary.ChebDistChartData[0].data.push(avgChebDistance);
      summary.EuclDistChartData[0].data.push(avgEuclDistance);
      summary.PsnrDistChartData[0].data.push(avgPsnr);
      summary.orConfChartData[0].data.push(avgOrConf);
      summary.SuccessRate[0].data.push(successRate);
      summary.lineChartLabels.push(summary.lineChartLabels.length.toString());

    }
    this.showChart = true;
  }
}