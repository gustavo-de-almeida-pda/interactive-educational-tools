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
  const [polyOrder, setPolyOrder] = useState(1);
  const [isStandardized, setIsStandardized] = useState(false);
  const [regularization, setRegularization] = useState<'none' | 'l1' | 'l2'>('none');
  const [lambdaL1, setLambdaL1] = useState(0);
  const [lambdaL2, setLambdaL2] = useState(0);
  const [regenerateTrigger, setRegenerateTrigger] = useState(0);

  // Data state
  const [trainDataRaw, setTrainDataRaw] = useState<any>(null);
  const [testDataRaw, setTestDataRaw] = useState<any>(null);

  // Generate raw data
  useEffect(() => {
    const train = generateData(-5, 5, sampleMode === 'increment' ? increment : null, sampleMode === 'points' ? numPoints : null, noiseLevel, false);
    const test = generateData(-5, 5, sampleMode === 'increment' ? increment : null, sampleMode === 'points' ? numPoints : null, noiseLevel, true);
    setTrainDataRaw(train);
    setTestDataRaw(test);
  }, [sampleMode, increment, numPoints, noiseLevel, regenerateTrigger]);

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
      // Z-score normalize x (using training statistics)
      const xStats = standardize(xTrainOriginal);
      xForMatrix_train = xStats.standardized;
      xForMatrix_test = xTestOriginal.map(v => (v - xStats.mean) / xStats.stdDev);
    } else {
      // Normalize x to [-1, 1] for numerical stability
      xForMatrix_train = normalizeToUnit(xTrainOriginal, -5, 5);
      xForMatrix_test = normalizeToUnit(xTestOriginal, -5, 5);
    }

    // y is NEVER standardized (issue #11 fix)
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

    // All metrics in original y scale (y was never standardized)
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

    // Descriptive statistics of y
    const trainYMean = mean(yTrainOriginal);
    const testYMean = mean(yTestOriginal);
    const trainYStd = stdDev(yTrainOriginal, trainYMean);
    const testYStd = stdDev(yTestOriginal, testYMean);
    const trainCV = trainYMean !== 0 ? (trainYStd / Math.abs(trainYMean)) * 100 : 0;
    const testCV = testYMean !== 0 ? (testYStd / Math.abs(testYMean)) * 100 : 0;

    // Build equation string (descending order, original x scale)
    let eq = 'y = ';
    if (isStandardized) {
      // Show equation in standardized x space
      const terms: string[] = [];
      for (let i = polyOrder; i >= 0; i--) {
        const coef = beta.get(i, 0);
        if (i === 0) {
          terms.push(`${coef >= 0 && terms.length > 0 ? '+ ' : ''}${coef.toFixed(4)}`);
        } else {
          const sign = coef >= 0 && terms.length > 0 ? '+ ' : (coef < 0 ? '- ' : '');
          const absCoef = Math.abs(coef).toFixed(4);
          const xTerm = i === 1 ? 'x\u0303' : `x\u0303^${i}`;
          terms.push(`${sign}${absCoef}${xTerm}`);
        }
      }
      eq += terms.join(' ');
      eq += '  (x\u0303 = standardized x)';
    } else {
      // Back-transform coefficients to original x scale
      // x_norm = (x - (-5)) / (5 - (-5)) * 2 - 1 = x/5
      // So x = 5 * x_norm, and β_orig_j = β_norm_j / 5^j
      const terms: string[] = [];
      for (let i = polyOrder; i >= 0; i--) {
        const coef = beta.get(i, 0) / Math.pow(5, i);
        if (i === 0) {
          terms.push(`${coef >= 0 && terms.length > 0 ? '+ ' : ''}${coef.toFixed(4)}`);
        } else {
          const sign = coef >= 0 && terms.length > 0 ? '+ ' : (coef < 0 ? '- ' : '');
          const absCoef = Math.abs(coef).toFixed(4);
          const xTerm = i === 1 ? 'x' : `x^${i}`;
          terms.push(`${sign}${absCoef}${xTerm}`);
        }
      }
      eq += terms.join(' ');
    }

    // Format data for charts (ALWAYS ORIGINAL SCALE)
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

    // Scatter data with both noisy target AND true target
    const scatterChartData = yTestOriginal.map((y, i) => ({
      target: y,
      pred: yTestHat[i]
    }));

    // Dynamic histogram bins (Sturges' rule for bin count)
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

    // Warning: p >= n
    const pGeqN = polyOrder + 1 >= n_train;

    return {
      metrics: {
        trainR2,
        testR2,
        trainAdjR2,
        testAdjR2,
        trainMSE,
        testMSE,
        resMean,
        resStd
      },
      descriptiveStats: {
        trainYMean,
        testYMean,
        trainYStd,
        testYStd,
        trainCV,
        testCV
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
              descriptiveStats={processedData?.descriptiveStats || {
                trainYMean: 0, testYMean: 0,
                trainYStd: 0, testYStd: 0,
                trainCV: 0, testCV: 0
              }}
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
