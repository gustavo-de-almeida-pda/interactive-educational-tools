import { Matrix, SVD, QrDecomposition } from 'ml-matrix';

export function generateData(
  xMin: number,
  xMax: number,
  increment: number | null,
  numPoints: number | null,
  noiseLevel: number,
  isTest: boolean,
  trueOrder: number = 2
) {
  let x: number[] = [];
  if (increment !== null) {
    for (let i = xMin; i <= xMax; i += increment) {
      x.push(i);
    }
  } else if (numPoints !== null) {
    const step = (xMax - xMin) / (numPoints - 1);
    for (let i = 0; i < numPoints; i++) {
      x.push(xMin + i * step);
    }
  }

  // True function: y = (x/xRange)^trueOrder * xRange^2
  // This keeps the output range bounded (~[-100, 100]) regardless of trueOrder
  const xRange = Math.max(Math.abs(xMin), Math.abs(xMax));
  const scale = xRange * xRange;
  const y_t = x.map((val) => Math.pow(val / xRange, trueOrder) * scale);

  // Box-Muller transform for normal distribution
  const randomNormal = () => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  // Both train and test use Gaussian noise N(0, s²) to satisfy the i.i.d. assumption
  const noise = x.map(() => randomNormal() * noiseLevel);

  const y_noise = y_t.map((val, i) => val + noise[i]);

  return { x, y_t, y_noise, noise };
}

export function mean(arr: number[]) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// Sample standard deviation (n-1 denominator, Bessel's correction)
export function stdDev(arr: number[], m?: number) {
  if (arr.length <= 1) return 0;
  const avg = m !== undefined ? m : mean(arr);
  const sumSquareDiffs = arr.reduce((acc, val) => acc + (val - avg) ** 2, 0);
  return Math.sqrt(sumSquareDiffs / (arr.length - 1));
}

export function mse(yTrue: number[], yPred: number[]) {
  return yTrue.reduce((acc, val, i) => acc + (val - yPred[i]) ** 2, 0) / yTrue.length;
}

export function rSquared(yTrue: number[], yPred: number[]) {
  const yMean = mean(yTrue);
  const ssTot = yTrue.reduce((acc, val) => acc + (val - yMean) ** 2, 0);
  const ssRes = yTrue.reduce((acc, val, i) => acc + (val - yPred[i]) ** 2, 0);
  if (ssTot === 0) return 1;
  return 1 - ssRes / ssTot;
}

export function adjustedR2(r2: number, n: number, p: number) {
  if (n - p - 1 <= 0) return NaN;
  return 1 - (1 - r2) * (n - 1) / (n - p - 1);
}

export function standardize(data: number[], m?: number, s?: number) {
  const avg = m !== undefined ? m : mean(data);
  const sd = s !== undefined ? s : stdDev(data, avg);
  if (sd === 0) return { standardized: data.map(() => 0), mean: avg, stdDev: sd };
  return {
    standardized: data.map((val) => (val - avg) / sd),
    mean: avg,
    stdDev: sd,
  };
}

// Normalize x to [-1, 1] for numerical stability
export function normalizeToUnit(x: number[], xMin: number, xMax: number) {
  const range = xMax - xMin;
  if (range === 0) return x.map(() => 0);
  return x.map(v => 2 * (v - xMin) / range - 1);
}

export function buildDesignMatrix(x: number[], order: number) {
  const X = new Matrix(x.length, order + 1);
  for (let i = 0; i < x.length; i++) {
    for (let j = 0; j <= order; j++) {
      X.set(i, j, Math.pow(x[i], j));
    }
  }
  return X;
}

// OLS via SVD with truncated singular values (numerically stable)
export function solveOLS(X: Matrix, y: Matrix) {
  const svd = new SVD(X, { autoTranspose: true });
  const S = svd.diagonal;
  const threshold = 1e-10 * S[0];
  const U = svd.leftSingularVectors;
  const V = svd.rightSingularVectors;
  const Uty = U.transpose().mmul(y);

  const SinvUty = new Matrix(S.length, 1);
  for (let i = 0; i < S.length; i++) {
    SinvUty.set(i, 0, S[i] > threshold ? Uty.get(i, 0) / S[i] : 0);
  }
  return V.mmul(SinvUty);
}

// Ridge via augmented system + QR decomposition
export function solveRidge(X: Matrix, y: Matrix, lambda: number) {
  if (lambda === 0) return solveOLS(X, y);
  const n = X.rows;
  const p = X.columns;
  const sqrtLambda = Math.sqrt(lambda);

  // Augmented system: [X; sqrt(λ)*D] β = [y; 0]
  // where D is identity with D[0,0]=0 (don't penalize intercept)
  const Xaug = Matrix.zeros(n + p, p);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      Xaug.set(i, j, X.get(i, j));
    }
  }
  for (let j = 1; j < p; j++) {
    Xaug.set(n + j, j, sqrtLambda);
  }

  const yaug = Matrix.zeros(n + p, 1);
  for (let i = 0; i < n; i++) {
    yaug.set(i, 0, y.get(i, 0));
  }

  const qr = new QrDecomposition(Xaug);
  return qr.solve(yaug);
}

export function solveLasso(X: Matrix, y: Matrix, lambda: number, iterations = 1000) {
  if (lambda === 0) return solveOLS(X, y);

  const n = X.rows;
  const p = X.columns;
  let beta = new Float64Array(p);

  // Precompute X columns and norms
  const xCols = [];
  const xNorms = [];
  for (let j = 0; j < p; j++) {
    const col = X.getColumn(j);
    xCols.push(col);
    let norm = 0;
    for (let i = 0; i < n; i++) norm += col[i] * col[i];
    xNorms.push(norm);
  }

  // Coordinate descent
  for (let iter = 0; iter < iterations; iter++) {
    let maxDiff = 0;
    for (let j = 0; j < p; j++) {
      let rho = 0;
      for (let i = 0; i < n; i++) {
        let y_pred_without_j = 0;
        for (let k = 0; k < p; k++) {
          if (k !== j) y_pred_without_j += xCols[k][i] * beta[k];
        }
        rho += xCols[j][i] * (y.get(i, 0) - y_pred_without_j);
      }

      let newBetaJ = beta[j];
      if (j === 0) {
        // Intercept not penalized
        newBetaJ = rho / xNorms[j];
      } else {
        // Soft thresholding
        if (rho < -lambda / 2) {
          newBetaJ = (rho + lambda / 2) / xNorms[j];
        } else if (rho > lambda / 2) {
          newBetaJ = (rho - lambda / 2) / xNorms[j];
        } else {
          newBetaJ = 0;
        }
      }

      maxDiff = Math.max(maxDiff, Math.abs(newBetaJ - beta[j]));
      beta[j] = newBetaJ;
    }
    if (maxDiff < 1e-6) break;
  }

  const betaMatrix = new Matrix(p, 1);
  for (let j = 0; j < p; j++) betaMatrix.set(j, 0, beta[j]);
  return betaMatrix;
}

export function predict(X: Matrix, beta: Matrix) {
  return X.mmul(beta).getColumn(0);
}
