import { Component, ElementRef, OnChanges, OnInit, ViewChild } from '@angular/core';
import * as tf from '@tensorflow/tfjs'
import { Attack, State, TestContext } from 'src/testContext';
import { Attacks } from 'src/attacks';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'advtest';
  testContext: TestContext;
  model: tf.LayersModel;
  dataset: any;
  constructor() {
    this.testContext = new TestContext();
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
  async drawImg(img) {
    const canvas = document.createElement('canvas');
    let resizedImg = tf.image.resizeNearestNeighbor(img.reshape([32, 32, 3]), [64, 64]);
    await tf.browser.toPixels(resizedImg, canvas);
    return canvas.toDataURL(); // will return the base64 encoding

  }
  async runUntargeted(attack) {
    for (let i = 0; i < this.dataset.length; i++) {
      let img = this.dataset[i].xs;
      let lbl = this.dataset[i].ys;
      let p_original = (this.model.predict(img) as tf.Tensor);
      let conf_original = p_original.max(1).dataSync()[0];
      let cl_original = p_original.argMax(1).dataSync()[0];

      this.testContext.reports.get(attack.name)[i].originalPrediction =
        this.testContext.classNames[cl_original];

      this.testContext.reports.get(attack.name)[i].originalConfidence = conf_original;

      var orImageB64 = await this.drawImg(img);

      this.testContext.reports.get(attack.name)[i].orImage = orImageB64;

      // Generate adversarial image from attack
      let aimg = tf.tidy(() => attack(this.model, img, lbl));

      // Display adversarial image and its probability
      let p_adversarial = (this.model.predict(aimg) as tf.Tensor);
      let conf_adv = p_adversarial.max(1).dataSync()[0];
      let cl_adv = p_adversarial.argMax(1).dataSync()[0];

      this.testContext.reports.get(attack.name)[i].advConfidence = conf_adv;
      this.testContext.reports.get(attack.name)[i].advPrediction =
        this.testContext.classNames[cl_adv];

      var advImgB64 = await this.drawImg(aimg);
      this.testContext.reports.get(attack.name)[i].advImage = advImgB64;
      this.reportProgres(i + 1, this.dataset.length);
    }

    // document.getElementById(`${attack.name}-success-rate`).innerText = `Success rate: ${(successes / 10).toFixed(1)}`;
  }
  async targetedAttack(attack) {

    for (let i = 0; i < this.dataset.length; i++) {
      let img = this.dataset[i].xs;
      let lbl = this.dataset[i].ys;
      let p = (this.model.predict(img) as tf.Tensor).dataSync()[i];
      //draw image
      for (let j = 0; j < this.dataset.length; j++) {
        if (j == (lbl.argMax(1) as tf.Tensor).dataSync[0]) {

          //don't run attack if the target class is the original class 
          continue;
        }

        let targetLbl = tf.oneHot(j, 10).reshape([1, this.dataset.length]);
        let aimg = tf.tidy(() => attack(this.model, img, lbl, targetLbl,))
        let shape = aimg.shape.slice(1);
        let resizedImg = tf.image.resizeNearestNeighbor(aimg.reshape(shape), [64, 64]);
        tf.browser.toPixels(resizedImg, document.getElementById('') as HTMLCanvasElement);
      }

    }
  }
  async runTest() {
    this.testContext.attackInProgress = true;
    this.testContext.reports = new Map<string, Attack[]>();
    if (this.testContext.fasm) {

      var attacks = new Array<Attack>();
      for (var i = 0; i < this.dataset.length; i++) {
        attacks.push(new Attack());
      }

      this.testContext.reports.set(Attacks.fgsm.name, attacks);

      await this.runUntargeted(Attacks.fgsm);
    }

    this.testContext.attackInProgress = false;

  }
  reportProgres(progres, total) {
    this.testContext.progress = progres / total * 100;
  }
}
