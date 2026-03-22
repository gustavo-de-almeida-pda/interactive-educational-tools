import React, { useState, useEffect, useMemo } from 'react';
import { Sidebar } from './components/Sidebar';
import { Graphs } from './components/Graphs';
import { MathDerivation } from './components/MathDerivation';
import {
  generateData,
  mean,
  stdDev,
  rSquared,
  standardize,
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

    let xTrain = [...xTrainOriginal];
    let yTrain = [...yTrainOriginal];
    let xTest = [...xTestOriginal];
    let yTest = [...yTestOriginal];

    let xMean = 0, xStd = 1, yMean = 0, yStd = 1;

    if (isStandardized) {
      const xStats = standardize(xTrainOriginal);
      xMean = xStats.mean;
      xStd = xStats.stdDev;
      xTrain = xStats.standardized;

      const yStats = standardize(yTrainOriginal);
      yMean = yStats.mean;
      yStd = yStats.stdDev;
      yTrain = yStats.standardized;

      xTest = xTestOriginal.map(v => (v - xMean) / xStd);
      yTest = yTestOriginal.map(v => (v - yMean) / yStd);
    }

    const X_train_mat = buildDesignMatrix(xTrain, polyOrder);
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
    
    const X_test_mat = buildDesignMatrix(xTest, polyOrder);
    const yTestHat = predict(X_test_mat, beta);

    // Inverse transform predictions to original scale for plotting and metrics
    const yTrainHatOrig = isStandardized ? yTrainHat.map(v => v * yStd + yMean) : yTrainHat;
    const yTestHatOrig = isStandardized ? yTestHat.map(v => v * yStd + yMean) : yTestHat;

    const displayYTrain = isStandardized ? yTrain : yTrainOriginal;
    const displayYTest = isStandardized ? yTest : yTestOriginal;
    const displayYTrainHat = isStandardized ? yTrainHat : yTrainHatOrig;
    const displayYTestHat = isStandardized ? yTestHat : yTestHatOrig;

    const trainR2 = rSquared(displayYTrain, displayYTrainHat);
    const testR2 = rSquared(displayYTest, displayYTestHat);

    const residuals = yTestOriginal.map((y, i) => y - yTestHatOrig[i]);
    const displayResiduals = displayYTest.map((y, i) => y - displayYTestHat[i]);
    const resMean = mean(displayResiduals);
    const resStd = stdDev(displayResiduals, resMean);

    // Build equation string
    let eq = `y = `;
    for (let i = 0; i <= polyOrder; i++) {
      const coef = beta.get(i, 0);
      if (i === 0) {
        eq += `${coef >= 0 ? '+' : ''}${coef.toFixed(4)}`;
      } else {
        eq += ` ${coef >= 0 ? '+' : '-'} ${Math.abs(coef).toFixed(4)}x^${i}`;
      }
    }

    // Format data for charts (ALWAYS ORIGINAL SCALE)
    const trainChartData = xTrainOriginal.map((x, i) => ({
      x,
      y_t: trainDataRaw.y_t[i],
      y_noise: yTrainOriginal[i],
      y_hat: yTrainHatOrig[i]
    }));

    const testChartData = xTestOriginal.map((x, i) => ({
      x,
      y_t: testDataRaw.y_t[i],
      y_noise: yTestOriginal[i],
      y_hat: yTestHatOrig[i]
    }));

    const resChartData = xTestOriginal.map((x, i) => ({
      x,
      res: residuals[i]
    }));

    const scatterChartData = yTestOriginal.map((y, i) => ({
      target: y,
      pred: yTestHatOrig[i]
    }));

    // Histogram bins
    const bins = Array(11).fill(0);
    const minBin = -15;
    const binWidth = 3;
    residuals.forEach(r => {
      const binIdx = Math.floor((r - minBin) / binWidth);
      if (binIdx >= 0 && binIdx < 11) bins[binIdx]++;
    });
    const histChartData = bins.map((count, i) => ({
      bin: minBin + i * binWidth + binWidth / 2,
      count
    }));

    return {
      metrics: {
        trainMean: mean(displayYTrain),
        trainStd: stdDev(displayYTrain),
        trainR2,
        testMean: mean(displayYTest),
        testStd: stdDev(displayYTest),
        testR2,
        resMean,
        resStd
      },
      polyEquation: eq,
      trainChartData,
      testChartData,
      resChartData,
      scatterChartData,
      histChartData
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
              <li>Evaluate model performance using metrics such as R² and residual analysis.</li>
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
                trainMean: 0, trainStd: 0, trainR2: 0,
                testMean: 0, testStd: 0, testR2: 0,
                resMean: 0, resStd: 0
              }}
              polyEquation={processedData?.polyEquation || ''}
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
                standardize={isStandardized}
              />
            )}
            <MathDerivation />
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
