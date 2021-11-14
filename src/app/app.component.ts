import { Component, ElementRef, OnChanges, OnInit, ViewChild } from '@angular/core';
import * as tf from '@tensorflow/tfjs'
import { Attack, Config, Source, TestContext } from 'src/testContext';
import { AttackResult, Attacks, BimConfig, CwConfig, FgsmConfig } from 'src/attacks';
import { couldStartTrivia } from 'typescript';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {

  canvas = document.createElement('canvas');
  title = 'advtest';
  testContext: TestContext;
  model: tf.LayersModel;
  dataset: any;
  constructor() {
    this.testContext = new TestContext();
    this.testContext.config = new Config();

    this.testContext.config.fgsm = new FgsmConfig();
    this.testContext.config.fgsm.epsilon = 0.1;

    this.testContext.config.bim = new BimConfig();
    this.testContext.config.bim.epsilon = 0.1;
    this.testContext.config.bim.alpha = 0.01;
    this.testContext.config.bim.iterations = 30;

    this.testContext.config.cw = new CwConfig();
    this.testContext.config.cw.successRate = 5;
    this.testContext.config.cw.confidenceRate = 1;
    this.testContext.config.cw.learningRate = 0.1;
    this.testContext.config.cw.iterations = 100;

  }
  getRandomInt(max) {
    return Math.random();
  }

  perturb(xs, img) {

  }
  async ngOnInit() {

  }
  getEuclidianDistance(arr1, arr2) {
    // calculate euclidian distance between two arrays
    let distTensor = tf.tidy(() => {
      const distance = tf.squaredDifference(arr1, arr2).sum().sqrt();
      return distance.dataSync()
    })
    return distTensor[0];
  }
  @ViewChild('i_model') i_model: ElementRef;
  @ViewChild('i_weights') i_weights: ElementRef;
  @ViewChild('i_test_data') i_test_data: ElementRef;
  @ViewChild('i_labels') i_labels: ElementRef;
  @ViewChild('i_class_names') i_class_names: ElementRef;

  async loadModeFromFs() {
  }
  async loadModel() {

    this.model = await tf.loadLayersModel(tf.io.browserFiles([this.i_model.nativeElement.files[0], this.i_weights.nativeElement.files[0]]));

    this.model.summary(null, null, (x) => {
      this.testContext.summary = x;
      console.log(this.testContext.summary);
    });

    await this.loadDataset();

  }
  onFileChange(event) {
    event.target.nextSibling.innerText = event.target.files[0].name;
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

    var shape = img.shape;
    await tf.browser.toPixels(img.reshape(shape.slice(1)), this.canvas);
    return this.canvas.toDataURL();

  }
  async runUntargeted(attack, config) {
    for (let i = 0; i < this.dataset.length; i++) {
      let img = this.dataset[i].xs;
      let lbl = this.dataset[i].ys;
      let p_original = (this.model.predict(img) as tf.Tensor);
      let conf_original = p_original.max(1).dataSync()[0];
      let cl_original = p_original.argMax(1).dataSync()[0];

      let source = new Source();
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
      source.originalClassName = this.testContext.classNames[cl_original];
      source.originalConfidence = conf_original;
      source.originalImage = img;

      for (let j = 0; j < this.dataset.length; j++) {
        if (j == i) continue;//skip same class
        let targetLbl = this.dataset[j].ys;
        let targetLblIndex = targetLbl.argMax(1).dataSync()[0];

        let attackResult = tf.tidy(() => attack(this.model, img, lbl, targetLbl, config));

        await this.fillReport(source, attack.name, a, attackResult);
        this.testContext.reports.get(attack.name)[a].targetClass =
          this.testContext.classNames[targetLblIndex];

        a++;
        this.reportProgres(a + 1, this.dataset.length * this.dataset.length);
      }

    }
  }
  async fillReport(source: Source, attackName: string, a: number, attackResult: AttackResult) {
    let p_adversarial = (this.model.predict(attackResult.advImg) as tf.Tensor);
    let conf_adv = p_adversarial.max(1).dataSync()[0];
    let cl_adv = p_adversarial.argMax(1).dataSync()[0];
    this.testContext.reports.get(attackName)[a].originalPrediction = source.originalClassName;
    this.testContext.reports.get(attackName)[a].originalConfidence = source.originalConfidence;
    var orImageB64 = await this.getDataUrl(source.originalImage);
    this.testContext.reports.get(attackName)[a].orImage = orImageB64;
    this.testContext.reports.get(attackName)[a].advConfidence = conf_adv;
    this.testContext.reports.get(attackName)[a].advPrediction =
      this.testContext.classNames[cl_adv];
    this.testContext.reports.get(attackName)[a].euclidianDistance = this.getEuclidianDistance(attackResult.advImg, source.originalImage);
    var advImgB64 = await this.getDataUrl(attackResult.advImg);
    this.testContext.reports.get(attackName)[a].advImage = advImgB64;

    if (attackResult.delta) {
      var deltaB64 = await this.getDataUrl(attackResult.delta);
      this.testContext.reports.get(attackName)[a].delta = deltaB64;
    }
  }

  async runTest() {
    this.testContext.attackInProgress = true;
    this.testContext.progress = 0;
    this.testContext.reports = new Map<string, Attack[]>();

    if (this.testContext.fgsm) {

      if (this.testContext.config.fgsm.targeted) {
        this.BuildReportForTargeted(Attacks.fgsmTargeted.name);
        await this.runTargeted(Attacks.fgsmTargeted, this.testContext.config.fgsm);

      }
      else {
        this.BuildReport(Attacks.fgsm.name);
        await this.runUntargeted(Attacks.fgsm, this.testContext.config.fgsm);

      }
    }

    if (this.testContext.bim) {
      if (this.testContext.config.bim.targeted) {
        this.BuildReportForTargeted(Attacks.bimTargeted.name)
        await this.runTargeted(Attacks.bimTargeted, this.testContext.config.bim);
      }
      else {
        this.BuildReport(Attacks.bim.name);
        await this.runUntargeted(Attacks.bim, this.testContext.config.bim);

      }
    }

    if (this.testContext.jsma) {

      this.BuildReportForTargeted(Attacks.jsma.name);
      await this.runTargeted(Attacks.jsma, this.testContext.config.fgsm);
    }

    if (this.testContext.cw) {

      this.BuildReportForTargeted(Attacks.cw.name);
      await this.runTargeted(Attacks.cw, this.testContext.config.cw);
    }

    if (this.testContext.diffEvol) {
      this.BuildReportForTargeted(Attacks.DifferentialEvolution.name);
      await this.runTargeted(Attacks.DifferentialEvolution, this.testContext.config.fgsm);
    }

    this.testContext.attackInProgress = false;

  }
  private BuildReportForTargeted(attackName) {
    var attacks = new Array<Attack>();
    for (var i = 0; i < this.dataset.length; i++) {
      for (var j = 0; j < this.dataset.length; j++) {
        if (j != i) attacks.push(new Attack());
      }
    }
    this.testContext.reports.set(attackName, attacks);
  }
  private BuildReport(attackName) {
    var attacks = new Array<Attack>();
    for (var i = 0; i < this.dataset.length; i++) {
      attacks.push(new Attack());
    }
    this.testContext.reports.set(attackName, attacks);
  }

  reportProgres(progres, total) {
    this.testContext.progress = progres / total * 100;
  }
}