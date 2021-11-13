import { Component, ElementRef, OnChanges, OnInit, ViewChild } from '@angular/core';
import * as tf from '@tensorflow/tfjs'
import { Attack, BimConfig, Config, FgsmConfig, State, TestContext } from 'src/testContext';
import { Attacks } from 'src/attacks';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {


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
    this.testContext.config.bim.iterations = 10;
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

      this.testContext.reports.get(attack.name)[i].originalPrediction =
        this.testContext.classNames[cl_original];

      this.testContext.reports.get(attack.name)[i].originalConfidence = conf_original;

      var orImageB64 = await this.getDataUrl(img);

      this.testContext.reports.get(attack.name)[i].orImage = orImageB64;

      // Generate adversarial image from attack
      let attackResult = tf.tidy(() => attack(this.model, img, lbl, config));

      // Display adversarial image and its probability
      let p_adversarial = (this.model.predict(attackResult.advImg) as tf.Tensor);
      let conf_adv = p_adversarial.max(1).dataSync()[0];
      let cl_adv = p_adversarial.argMax(1).dataSync()[0];

      this.testContext.reports.get(attack.name)[i].advConfidence = conf_adv;
      this.testContext.reports.get(attack.name)[i].advPrediction =
        this.testContext.classNames[cl_adv];

      this.testContext.reports.get(attack.name)[i].euclidianDistance = this.getEuclidianDistance(attackResult.advImg, img);
      var advImgB64 = await this.getDataUrl(attackResult.advImg);
      this.testContext.reports.get(attack.name)[i].advImage = advImgB64;
      if (attackResult.delta) {

        var deltaB64 = await this.getDataUrl(attackResult.delta);
        this.testContext.reports.get(attack.name)[i].delta = deltaB64;
      }

      this.reportProgres(i + 1, this.dataset.length);
    }

    // document.getElementById(`${attack.name}-success-rate`).innerText = `Success rate: ${(successes / 10).toFixed(1)}`;
  }
  async targetedAttack(attack, config) {

    let a = 0;
    for (let i = 0; i < 1; i++) {
      let img = this.dataset[i].xs;
      let lbl = this.dataset[i].ys;
      let p_original = (this.model.predict(img) as tf.Tensor);
      let conf_original = p_original.max(1).dataSync()[0];
      let cl_original = p_original.argMax(1).dataSync()[0];

      var orImageB64 = await this.getDataUrl(img);
      for (let j = 0; j < this.dataset.length; j++) {
        if (j == (lbl.argMax(1) as tf.Tensor).dataSync[0]) {
          continue;
        }
        let targetLbl = tf.oneHot(j, this.dataset.length).reshape([1, this.dataset.length]);
        this.testContext.reports.get(attack.name)[a].originalPrediction =
          this.testContext.classNames[cl_original];

        this.testContext.reports.get(attack.name)[a].originalConfidence = conf_original;


        this.testContext.reports.get(attack.name)[a].orImage = orImageB64;

        // Generate adversarial image from attack
        let attackResult = tf.tidy(() => attack(this.model, img, lbl, targetLbl, config));

        // Display adversarial image and its probability
        let p_adversarial = (this.model.predict(attackResult.advImg) as tf.Tensor);
        let conf_adv = p_adversarial.max(1).dataSync()[0];
        let cl_adv = p_adversarial.argMax(1).dataSync()[0];

        this.testContext.reports.get(attack.name)[a].advConfidence = conf_adv;
        this.testContext.reports.get(attack.name)[a].advPrediction =
          this.testContext.classNames[cl_adv];

        var advImgB64 = await this.getDataUrl(attackResult.advImg);
        this.testContext.reports.get(attack.name)[a].advImage = advImgB64;

        this.testContext.reports.get(attack.name)[a].euclidianDistance = this.getEuclidianDistance(attackResult.advImg, img);
        if (attackResult.delta) {

          var deltaB64 = await this.getDataUrl(attackResult.delta);
          this.testContext.reports.get(attack.name)[a].delta = deltaB64;
        }


        a++;

      }

    }
  }
  async runTest() {
    this.testContext.attackInProgress = true;
    this.testContext.progress = 0;
    this.testContext.reports = new Map<string, Attack[]>();

    if (this.testContext.fgsm) {

      this.BuildReport(Attacks.fgsm.name);
      await this.runUntargeted(Attacks.fgsm, this.testContext.config.fgsm);
    }

    if (this.testContext.bim) {

      this.BuildReport(Attacks.bim.name);
      await this.runUntargeted(Attacks.bim, this.testContext.config.bim);
    }

    if (this.testContext.jsma) {

      this.BuildReportForTargeted(Attacks.jsma.name);
      await this.targetedAttack(Attacks.jsma, this.testContext.config.fgsm);
    }

    if (this.testContext.cw) {

      this.BuildReportForTargeted(Attacks.cw.name);
      await this.targetedAttack(Attacks.cw, this.testContext.config.fgsm);
    }


    this.testContext.attackInProgress = false;

  }
  private BuildReportForTargeted(attackName) {
    var attacks = new Array<Attack>();
    for (var i = 0; i < this.dataset.length; i++) {
      for (var j = 0; j < this.dataset.length; j++) {

        attacks.push(new Attack());
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