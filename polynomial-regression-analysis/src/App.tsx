import React, { useState, useEffect, useMemo } from 'react';
import { Sidebar } from './components/Sidebar';
import { Graphs } from './components/Graphs';
import { MathDerivation } from './components/MathDerivation';
import {
  generateData,
  mean,
  stdDev,
  rSquared,
  adjustedR2,
  mse,
  standardize,
  normalizeToUnit,
  buildDesignMatrix,
  solveOLS,
  solveRidge,
  solveLasso,
  predict
} from './utils/math';
import { Matrix } from 'ml-matrix';

export const App = () => {
  const [sampleMode, setSampleMode] = useState<'increment' | 'points'>('increment');
  const [increment, setIncrement] = useState(0.25);
  const [numPoints, setNumPoints] = useState(25);
  const [noiseLevel, setNoiseLevel] = useState(5);
  const [trueOrder, setTrueOrder] = useState(2);
  const [polyOrder, setPolyOrder] = useState(1);
  const [isStandardized, setIsStandardized] = useState(false);
  const [regularization, setRegularization] = useState<'none' | 'l1' | 'l2'>('none');
  const [lambdaL1, setLambdaL1] = useState(0);
  const [lambdaL2, setLambdaL2] = useState(0);
  const [regenerateTrigger, setRegenerateTrigger] = useState(0);

  // Data state
  const [trainDataRaw, setTrainDataRaw] = useState<any>(null);
  const [testDataRaw, setTestDataRaw] = useState<any>(null);

  // Generate raw data — reacts to ALL data parameters including trueOrder
  useEffect(() => {
    const train = generateData(0, 10, sampleMode === 'increment' ? increment : null, sampleMode === 'points' ? numPoints : null, noiseLevel, false, trueOrder);
    const test = generateData(0, 10, sampleMode === 'increment' ? increment : null, sampleMode === 'points' ? numPoints : null, noiseLevel, true, trueOrder);
    setTrainDataRaw(train);
    setTestDataRaw(test);
  }, [sampleMode, increment, numPoints, noiseLevel, trueOrder, regenerateTrigger]);

  // Descriptive statistics — separate useMemo reacting directly to raw data
  const descriptiveStats = useMemo(() => {
    if (!trainDataRaw || !testDataRaw) {
      return { trainYMean: 0, testYMean: 0, trainYStd: 0, testYStd: 0, trainCV: 0, testCV: 0 };
    }
    const yTrain = trainDataRaw.y_noise;
    const yTest = testDataRaw.y_noise;
    const trainYMean = mean(yTrain);
    const testYMean = mean(yTest);
    const trainYStd = stdDev(yTrain, trainYMean);
    const testYStd = stdDev(yTest, testYMean);
    const trainCV = trainYMean !== 0 ? (trainYStd / Math.abs(trainYMean)) * 100 : 0;
    const testCV = testYMean !== 0 ? (testYStd / Math.abs(testYMean)) * 100 : 0;
    return { trainYMean, testYMean, trainYStd, testYStd, trainCV, testCV };
  }, [trainDataRaw, testDataRaw]);

  // Process data and build model
  const processedData = useMemo(() => {
    if (!trainDataRaw || !testDataRaw) return null;

    const xTrainOriginal = [...trainDataRaw.x];
    const yTrainOriginal = [...trainDataRaw.y_noise];
    const xTestOriginal = [...testDataRaw.x];
    const yTestOriginal = [...testDataRaw.y_noise];

    // Determine x values for the design matrix
    let xForMatrix_train: number[];
    let xForMatrix_test: number[];

    if (isStandardized) {
      const xStats = standardize(xTrainOriginal);
      xForMatrix_train = xStats.standardized;
      xForMatrix_test = xTestOriginal.map(v => (v - xStats.mean) / xStats.stdDev);
    } else {
      xForMatrix_train = normalizeToUnit(xTrainOriginal, 0, 10);
      xForMatrix_test = normalizeToUnit(xTestOriginal, 0, 10);
    }

    const yTrain = [...yTrainOriginal];

    const X_train_mat = buildDesignMatrix(xForMatrix_train, polyOrder);
    const y_train_mat = new Matrix(yTrain.map(v => [v]));

    let beta: Matrix;
    if (regularization === 'l1') {
      beta = solveLasso(X_train_mat, y_train_mat, lambdaL1);
    } else if (regularization === 'l2') {
      beta = solveRidge(X_train_mat, y_train_mat, lambdaL2);
    } else {
      beta = solveOLS(X_train_mat, y_train_mat);
    }

    const yTrainHat = predict(X_train_mat, beta);

    const X_test_mat = buildDesignMatrix(xForMatrix_test, polyOrder);
    const yTestHat = predict(X_test_mat, beta);

    const n_train = yTrainOriginal.length;
    const n_test = yTestOriginal.length;
    const p = polyOrder;

    const trainR2 = rSquared(yTrainOriginal, yTrainHat);
    const testR2 = rSquared(yTestOriginal, yTestHat);
    const trainAdjR2 = adjustedR2(trainR2, n_train, p);
    const testAdjR2 = adjustedR2(testR2, n_test, p);
    const trainMSE = mse(yTrainOriginal, yTrainHat);
    const testMSE = mse(yTestOriginal, yTestHat);

    const residuals = yTestOriginal.map((y, i) => y - yTestHat[i]);
    const resMean = mean(residuals);
    const resStd = stdDev(residuals, resMean);

    // Build equation string (descending order, original x scale)
    let eq = 'y = ';
    if (isStandardized) {
      const terms: string[] = [];
      for (let i = polyOrder; i >= 0; i--) {
        const coef = beta.get(i, 0);
        if (i === 0) {
          terms.push(`${coef >= 0 && terms.length > 0 ? '+ ' : ''}${coef.toFixed(3)}`);
        } else {
          const sign = coef >= 0 && terms.length > 0 ? '+ ' : (coef < 0 ? '- ' : '');
          const absCoef = Math.abs(coef).toFixed(3);
          const xTerm = i === 1 ? 'x\u0303' : `x\u0303^${i}`;
          terms.push(`${sign}${absCoef}${xTerm}`);
        }
      }
      eq += terms.join(' ');
      eq += '  (x\u0303 = standardized x)';
    } else {
      // Back-transform coefficients to original x scale
      // x_norm = (x - c) / s where c = 5, s = 5
      // Expand β_j * ((x - c) / s)^j using binomial theorem
      const c = 5, s = 5;
      const origCoefs = new Array(polyOrder + 1).fill(0);
      for (let j = 0; j <= polyOrder; j++) {
        const bj = beta.get(j, 0);
        // ((x - c)/s)^j = (1/s^j) * Σ_{k=0}^{j} C(j,k) * x^k * (-c)^(j-k)
        for (let k = 0; k <= j; k++) {
          let binom = 1;
          for (let m = 0; m < k; m++) binom = binom * (j - m) / (m + 1);
          origCoefs[k] += bj * binom * Math.pow(-c, j - k) / Math.pow(s, j);
        }
      }
      const terms: string[] = [];
      for (let i = polyOrder; i >= 0; i--) {
        const coef = origCoefs[i];
        if (i === 0) {
          terms.push(`${coef >= 0 && terms.length > 0 ? '+ ' : ''}${coef.toFixed(3)}`);
        } else {
          const sign = coef >= 0 && terms.length > 0 ? '+ ' : (coef < 0 ? '- ' : '');
          const absCoef = Math.abs(coef).toFixed(3);
          const xTerm = i === 1 ? 'x' : `x^${i}`;
          terms.push(`${sign}${absCoef}${xTerm}`);
        }
      }
      eq += terms.join(' ');
    }

    // Format data for charts
    const trainChartData = xTrainOriginal.map((x, i) => ({
      x,
      y_t: trainDataRaw.y_t[i],
      y_noise: yTrainOriginal[i],
      y_hat: yTrainHat[i]
    }));

    const testChartData = xTestOriginal.map((x, i) => ({
      x,
      y_t: testDataRaw.y_t[i],
      y_noise: yTestOriginal[i],
      y_hat: yTestHat[i]
    }));

    const resChartData = xTestOriginal.map((x, i) => ({
      x,
      res: residuals[i]
    }));

    const scatterChartData = yTestOriginal.map((y, i) => ({
      target: y,
      pred: yTestHat[i]
    }));

    // Dynamic histogram bins (Sturges' rule)
    const resMin = Math.min(...residuals);
    const resMax = Math.max(...residuals);
    const resRange = resMax - resMin;
    const numBins = Math.max(5, Math.min(20, Math.ceil(1 + 3.322 * Math.log10(residuals.length))));
    const binWidth = resRange > 0 ? resRange / numBins : 1;
    const binStart = resMin - binWidth * 0.5;

    const bins = Array(numBins).fill(0);
    residuals.forEach(r => {
      const binIdx = Math.floor((r - binStart) / binWidth);
      if (binIdx >= 0 && binIdx < numBins) bins[binIdx]++;
      else if (binIdx >= numBins) bins[numBins - 1]++;
    });
    const histChartData = bins.map((count, i) => ({
      bin: parseFloat((binStart + i * binWidth + binWidth / 2).toFixed(2)),
      count
    }));

    const pGeqN = polyOrder + 1 >= n_train;

    return {
      metrics: {
        trainR2, testR2,
        trainAdjR2, testAdjR2,
        trainMSE, testMSE,
        resMean, resStd
      },
      polyEquation: eq,
      trainChartData,
      testChartData,
      resChartData,
      scatterChartData,
      histChartData,
      pGeqN,
      nTrain: n_train
    };

  }, [trainDataRaw, testDataRaw, polyOrder, isStandardized, regularization, lambdaL1, lambdaL2]);

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-8 font-sans text-slate-800">
      <div className="max-w-[1600px] mx-auto space-y-6">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-slate-900 mb-4">Polynomial Regression Analysis</h1>
          <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200">
            <h2 className="text-lg font-semibold mb-2 text-slate-700">Intended Learning Outcomes (ILOs)</h2>
            <ul className="list-disc pl-5 space-y-1 text-slate-600">
              <li>Understand the concepts of learning, underfitting, and overfitting in data-driven models.</li>
              <li>Analyze the impact of sample size, noise level, model complexity, and regularization on model generalization.</li>
              <li>Evaluate model performance using metrics such as R², Adjusted R², MSE, and residual analysis.</li>
            </ul>
          </div>
        </header>

        <main className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-3 space-y-6">
            <Sidebar
              onRegenerate={() => setRegenerateTrigger(prev => prev + 1)}
              sampleMode={sampleMode}
              setSampleMode={setSampleMode}
              increment={increment}
              setIncrement={setIncrement}
              numPoints={numPoints}
              setNumPoints={setNumPoints}
              noiseLevel={noiseLevel}
              setNoiseLevel={setNoiseLevel}
              trueOrder={trueOrder}
              setTrueOrder={setTrueOrder}
              polyOrder={polyOrder}
              setPolyOrder={setPolyOrder}
              standardize={isStandardized}
              setStandardize={setIsStandardized}
              regularization={regularization}
              setRegularization={setRegularization}
              lambdaL1={lambdaL1}
              setLambdaL1={setLambdaL1}
              lambdaL2={lambdaL2}
              setLambdaL2={setLambdaL2}
              metrics={processedData?.metrics || {
                trainR2: 0, testR2: 0,
                trainAdjR2: 0, testAdjR2: 0,
                trainMSE: 0, testMSE: 0,
                resMean: 0, resStd: 0
              }}
              descriptiveStats={descriptiveStats}
              polyEquation={processedData?.polyEquation || ''}
              pGeqN={processedData?.pGeqN || false}
              nTrain={processedData?.nTrain || 0}
            />
          </div>

          <div className="lg:col-span-9 space-y-6">
            {processedData && (
              <Graphs
                trainData={processedData.trainChartData}
                testData={processedData.testChartData}
                resData={processedData.resChartData}
                histData={processedData.histChartData}
                scatterData={processedData.scatterChartData}
              />
            )}
            <MathDerivation
              polyOrder={polyOrder}
              regularization={regularization}
            />
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
