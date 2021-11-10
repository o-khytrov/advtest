import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import * as tf from '@tensorflow/tfjs'
import { TestContext } from 'src/testContext';
import { Attacks } from 'src/attacks';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'advtest';
  testContext: TestContext;
  model: tf.LayersModel;
  dataset: any;
  constructor() {
    this.testContext = new TestContext();
  }
  ngOnInit(): void {

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
      await this.runUntargeted(Attacks.fgsm);

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
  async drawImg(img, element, attackName, msg, success = undefined) {
    //let canvas = document.(attackName).getElementsByClassName(element)[0];
    var canvas = document.createElement('canvas') as HTMLCanvasElement;
    let resizedImg = tf.image.resizeNearestNeighbor(img.reshape([32, 32, 3]), [64, 64]);
    await tf.browser.toPixels(resizedImg, canvas);

    if (msg !== undefined) {
      var div = document.createElement('div');
      div.innerHTML = msg;
    }
    if (success === true) {
      canvas.style.borderColor = 'lime';
      canvas.style.borderWidth = '2px';
    }
    var body = document.getElementsByTagName("body")[0];
    body.appendChild(canvas);
  }
  async runUntargeted(attack) {
    let successes = 0;

    for (let i = 0; i < 10; i++) { // For each row
      let img = this.dataset[i].xs;
      let lbl = this.dataset[i].ys;

      console.log(`untargeted ${i}`)
      let p = (this.model.predict(img) as tf.Tensor).dataSync()[i];
      await this.drawImg(img, i.toString(), attack.name, `Pred: ${this.testContext.classNames[i]}<br/>Prob: ${p.toFixed(3)}`);

      // Generate adversarial image from attack
      let aimg = tf.tidy(() => attack(this.model, img, lbl));

      // Display adversarial image and its probability
      p = (this.model.predict(aimg) as tf.Tensor).max(1).dataSync()[0];
      let albl = (this.model.predict(aimg) as tf.Tensor).argMax(1).dataSync()[0];
      let oldlbl = lbl.argMax(1).dataSync()[0];
      if (albl !== oldlbl) {
        successes++;
        await this.drawImg(aimg, `${i}a`, attack.name, `Pred: ${this.testContext.classNames[albl]}<br/>Prob: ${p.toFixed(3)}`, true);
      } else {
        await this.drawImg(aimg, `${i}a`, attack.name, `Pred: ${this.testContext.classNames[albl]}<br/>Prob: ${p.toFixed(3)}`);
      }
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
}
