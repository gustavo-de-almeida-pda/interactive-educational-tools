import React from 'react';
import 'katex/dist/katex.min.css';
import Latex from 'react-latex-next';

interface SidebarProps {
  onRegenerate: () => void;
  sampleMode: 'increment' | 'points';
  setSampleMode: (mode: 'increment' | 'points') => void;
  increment: number;
  setIncrement: (val: number) => void;
  numPoints: number;
  setNumPoints: (val: number) => void;
  noiseLevel: number;
  setNoiseLevel: (val: number) => void;
  trueOrder: number;
  setTrueOrder: (val: number) => void;
  polyOrder: number;
  setPolyOrder: (val: number) => void;
  standardize: boolean;
  setStandardize: (val: boolean) => void;
  regularization: 'none' | 'l1' | 'l2';
  setRegularization: (val: 'none' | 'l1' | 'l2') => void;
  lambdaL1: number;
  setLambdaL1: (val: number) => void;
  lambdaL2: number;
  setLambdaL2: (val: number) => void;
  metrics: {
    trainR2: number;
    testR2: number;
    trainAdjR2: number;
    testAdjR2: number;
    trainMSE: number;
    testMSE: number;
    resMean: number;
    resStd: number;
  };
  descriptiveStats: {
    trainYMean: number;
    testYMean: number;
    trainYStd: number;
    testYStd: number;
    trainCV: number;
    testCV: number;
  };
  polyEquation: string;
  pGeqN: boolean;
  nTrain: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  onRegenerate,
  sampleMode,
  setSampleMode,
  increment,
  setIncrement,
  numPoints,
  setNumPoints,
  noiseLevel,
  setNoiseLevel,
  trueOrder,
  setTrueOrder,
  polyOrder,
  setPolyOrder,
  standardize,
  setStandardize,
  regularization,
  setRegularization,
  lambdaL1,
  setLambdaL1,
  lambdaL2,
  setLambdaL2,
  metrics,
  descriptiveStats,
  polyEquation,
  pGeqN,
  nTrain,
}) => {
  return (
    <div className="space-y-6">
      <button
        onClick={onRegenerate}
        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-4 rounded-xl shadow-sm transition-colors"
      >
        Regenerate Data
      </button>

      {/* Data Parameter Box */}
      <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 mb-4 border-b pb-2">Data Parameter</h3>

        <div className="space-y-4">
          <div>
            <h4 className="font-semibold text-sm text-slate-700 mb-2">Sample Size Selection</h4>
            <div className="flex gap-4 mb-2">
              <label className="flex items-center text-sm cursor-pointer">
                <input
                  type="radio"
                  checked={sampleMode === 'increment'}
                  onChange={() => setSampleMode('increment')}
                  className="mr-2 text-indigo-600 focus:ring-indigo-500"
                />
                Increment
              </label>
              <label className="flex items-center text-sm cursor-pointer">
                <input
                  type="radio"
                  checked={sampleMode === 'points'}
                  onChange={() => setSampleMode('points')}
                  className="mr-2 text-indigo-600 focus:ring-indigo-500"
                />
                Points
              </label>
            </div>

            {sampleMode === 'increment' ? (
              <div>
                <label className="block text-xs text-slate-500 mb-1">Increment (i) (i = 0.25 to 0.75, step: 0.25)</label>
                <input
                  type="range"
                  min="0.25"
                  max="0.75"
                  step="0.25"
                  value={increment}
                  onChange={(e) => setIncrement(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="text-right text-sm font-medium">{increment.toFixed(2)}</div>
              </div>
            ) : (
              <div>
                <label className="block text-xs text-slate-500 mb-1">Points (n = 25 to 250, step: 25)</label>
                <input
                  type="range"
                  min="25"
                  max="250"
                  step="25"
                  value={numPoints}
                  onChange={(e) => setNumPoints(parseInt(e.target.value))}
                  className="w-full"
                />
                <div className="text-right text-sm font-medium">{numPoints}</div>
              </div>
            )}
          </div>

          <div>
            <h4 className="font-semibold text-sm text-slate-700 mb-1">Noise Level (s)</h4>
            <label className="block text-xs text-slate-500 mb-1">(s = 0 to 15, step: 1)</label>
            <input
              type="range"
              min="0"
              max="15"
              step="1"
              value={noiseLevel}
              onChange={(e) => setNoiseLevel(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="text-right text-sm font-medium">{noiseLevel}</div>
          </div>

          <div>
            <h4 className="font-semibold text-sm text-slate-700 mb-1">True Function Order</h4>
            <label className="block text-xs text-slate-500 mb-1">(1 to 50, step: 1)</label>
            <input
              type="range"
              min="1"
              max="50"
              step="1"
              value={trueOrder}
              onChange={(e) => setTrueOrder(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="text-right text-sm font-medium">{trueOrder}</div>
          </div>
        </div>
      </div>

      {/* Model Parameter Box */}
      <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 mb-4 border-b pb-2">Model Parameter</h3>

        <div className="space-y-4">
          <div>
            <h4 className="font-semibold text-sm text-slate-700 mb-1">Polynomial Order (p)</h4>
            <label className="block text-xs text-slate-500 mb-1">(p = 1 to 50, step: 1)</label>
            <input
              type="range"
              min="1"
              max="50"
              step="1"
              value={polyOrder}
              onChange={(e) => setPolyOrder(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="text-right text-sm font-medium">{polyOrder}</div>
            {pGeqN && (
              <div className="mt-2 p-2 bg-amber-50 border border-amber-300 rounded text-xs text-amber-800">
                p + 1 = {polyOrder + 1} parameters {'>='} n = {nTrain} samples. The system is underdetermined (more parameters than data points). Results may be unreliable.
              </div>
            )}
          </div>

          <div>
            <label className="flex items-center text-sm font-semibold text-slate-700 cursor-pointer mb-4">
              <input
                type="checkbox"
                checked={standardize}
                onChange={(e) => setStandardize(e.target.checked)}
                className="mr-2 rounded text-indigo-600 focus:ring-indigo-500"
              />
              Standardize Features (X)
            </label>
          </div>

          <div>
            <h4 className="font-semibold text-sm text-slate-700 mb-2">Regularization parameter (<Latex>{'$\\lambda$'}</Latex>)</h4>
            <select
              value={regularization}
              onChange={(e) => setRegularization(e.target.value as any)}
              className="w-full border-slate-300 rounded-md shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm mb-3 p-2 border"
            >
              <option value="none">None</option>
              <option value="l1">Lasso (L1)</option>
              <option value="l2">Ridge (L2)</option>
            </select>

            {regularization === 'l1' && (
              <div className="mt-3 bg-slate-50 p-3 rounded-lg">
                <h5 className="font-medium text-xs text-slate-600 mb-1">Lasso (L1)</h5>
                <label className="block text-xs text-slate-500 mb-1">Increment <Latex>{'$\\lambda$'}</Latex> (<Latex>{'$\\lambda$'}</Latex> = 0 to 10, step: 0.5)</label>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="0.5"
                  value={lambdaL1}
                  onChange={(e) => setLambdaL1(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="text-right text-sm font-medium">{lambdaL1.toFixed(1)}</div>
                <div className="mt-2 text-xs text-center overflow-hidden">
                  <Latex>{'$$ L = \\sum (y_i - \\hat{y}_i)^2 + \\lambda \\sum_{j=1}^{p} |\\beta_j| $$'}</Latex>
                </div>
              </div>
            )}

            {regularization === 'l2' && (
              <div className="mt-3 bg-slate-50 p-3 rounded-lg">
                <h5 className="font-medium text-xs text-slate-600 mb-1">Ridge (L2)</h5>
                <label className="block text-xs text-slate-500 mb-1">(<Latex>{'$\\lambda$'}</Latex> = 0 to 100, step: 5)</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="5"
                  value={lambdaL2}
                  onChange={(e) => setLambdaL2(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="text-right text-sm font-medium">{lambdaL2}</div>
                <div className="mt-2 text-xs text-center overflow-hidden">
                  <Latex>{'$$ L = \\sum (y_i - \\hat{y}_i)^2 + \\lambda \\sum_{j=1}^{p} \\beta_j^2 $$'}</Latex>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Descriptive Statistics Box */}
      <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 mb-4 border-b pb-2">Descriptive Statistics (y)</h3>

        <div className="overflow-hidden rounded-lg border border-slate-200">
          <table className="min-w-full divide-y divide-slate-200 text-xs text-center">
            <thead className="bg-slate-50">
              <tr>
                <th className="px-2 py-2 font-medium text-slate-500">Statistic</th>
                <th className="px-2 py-2 font-medium text-slate-500">Noisy obs. (train)</th>
                <th className="px-2 py-2 font-medium text-slate-500">Noisy obs. (test)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              <tr>
                <td className="px-2 py-2 font-medium text-slate-900"><Latex>{'$\\bar{y}$'}</Latex> (Mean)</td>
                <td className="px-2 py-2 text-slate-600">{descriptiveStats.trainYMean.toFixed(4)}</td>
                <td className="px-2 py-2 text-slate-600">{descriptiveStats.testYMean.toFixed(4)}</td>
              </tr>
              <tr>
                <td className="px-2 py-2 font-medium text-slate-900"><Latex>{'$s$'}</Latex> (Std Dev)</td>
                <td className="px-2 py-2 text-slate-600">{descriptiveStats.trainYStd.toFixed(4)}</td>
                <td className="px-2 py-2 text-slate-600">{descriptiveStats.testYStd.toFixed(4)}</td>
              </tr>
              <tr>
                <td className="px-2 py-2 font-medium text-slate-900">CV (%)</td>
                <td className="px-2 py-2 text-slate-600">{descriptiveStats.trainCV.toFixed(2)}</td>
                <td className="px-2 py-2 text-slate-600">{descriptiveStats.testCV.toFixed(2)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Metrics Box */}
      <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 mb-4 border-b pb-2">Metrics</h3>

        <h4 className="font-semibold text-sm text-slate-700 mb-2">Training vs. Test Performance</h4>
        <div className="overflow-hidden rounded-lg border border-slate-200 mb-4">
          <table className="min-w-full divide-y divide-slate-200 text-xs text-center">
            <thead className="bg-slate-50">
              <tr>
                <th className="px-2 py-2 font-medium text-slate-500">Metric</th>
                <th className="px-2 py-2 font-medium text-slate-500">Training</th>
                <th className="px-2 py-2 font-medium text-slate-500">Testing</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              <tr>
                <td className="px-2 py-2 font-medium text-slate-900">MSE</td>
                <td className="px-2 py-2 text-slate-600">{metrics.trainMSE.toFixed(4)}</td>
                <td className="px-2 py-2 text-slate-600">{metrics.testMSE.toFixed(4)}</td>
              </tr>
              <tr>
                <td className="px-2 py-2 font-medium text-slate-900">R²</td>
                <td className="px-2 py-2 text-slate-600">{metrics.trainR2.toFixed(4)}</td>
                <td className="px-2 py-2 text-slate-600">{metrics.testR2.toFixed(4)}</td>
              </tr>
              <tr>
                <td className="px-2 py-2 font-medium text-slate-900">Adj. R²</td>
                <td className="px-2 py-2 text-slate-600">{isNaN(metrics.trainAdjR2) ? 'N/A' : metrics.trainAdjR2.toFixed(4)}</td>
                <td className="px-2 py-2 text-slate-600">{isNaN(metrics.testAdjR2) ? 'N/A' : metrics.testAdjR2.toFixed(4)}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <h4 className="font-semibold text-sm text-slate-700 mb-2">Residual Statistics (Test Set)</h4>
        <div className="grid grid-cols-2 gap-2 text-center text-sm">
          <div className="bg-slate-50 p-2 rounded border border-slate-100">
            <div className="text-xs text-slate-500 mb-1">Mean</div>
            <div className="font-medium">{metrics.resMean.toFixed(4)}</div>
          </div>
          <div className="bg-slate-50 p-2 rounded border border-slate-100">
            <div className="text-xs text-slate-500 mb-1">Std Dev</div>
            <div className="font-medium">{metrics.resStd.toFixed(4)}</div>
          </div>
        </div>
      </div>

      {/* Polynomial Model Box */}
      <div className="bg-white p-5 rounded-xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 mb-4 border-b pb-2">Polynomial Model</h3>
        <div className="text-sm font-mono text-slate-700 break-words whitespace-pre-wrap">
          {polyEquation}
        </div>
      </div>
    </div>
  );
};
