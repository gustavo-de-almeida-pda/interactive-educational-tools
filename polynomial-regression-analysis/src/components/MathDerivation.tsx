import React from 'react';
import 'katex/dist/katex.min.css';
import Latex from 'react-latex-next';

interface MathDerivationProps {
  polyOrder: number;
  regularization: 'none' | 'l1' | 'l2';
}

export const MathDerivation: React.FC<MathDerivationProps> = ({ polyOrder, regularization }) => {
  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 mt-6 overflow-x-auto">
      <h3 className="text-lg font-semibold mb-4 text-slate-800">Mathematical Formulation: Polynomial Regression (Order {polyOrder})</h3>

      <div className="space-y-6 text-slate-700">
        <section>
          <h4 className="font-medium mb-2">1. The Model</h4>
          <p className="mb-2">
            We fit a polynomial of order <Latex>{'$p = ' + polyOrder + '$'}</Latex> to the data:
          </p>
          <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto">
            <Latex>{'$$ y_i = \\beta_0 + \\beta_1 x_i + \\beta_2 x_i^2 + \\cdots + \\beta_p x_i^p + \\epsilon_i = \\sum_{j=0}^{p} \\beta_j x_i^j + \\epsilon_i $$'}</Latex>
          </div>
          <p className="mt-2 text-sm text-slate-500">
            Where <Latex>{'$y_i$'}</Latex> is the target, <Latex>{'$x_i$'}</Latex> is the feature, <Latex>{'$\\beta_j$'}</Latex> are the <Latex>{'$p+1$'}</Latex> coefficients, <Latex>{'$\\epsilon_i$'}</Latex> is the random error, and <Latex>{'$i = 1, \\ldots, n$'}</Latex>.
          </p>
        </section>

        <section>
          <h4 className="font-medium mb-2">2. Design Matrix</h4>
          <p className="mb-2">In matrix form, <Latex>{'$\\mathbf{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon}$'}</Latex>, where the design matrix is the Vandermonde matrix:</p>
          <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto">
            <Latex>{'$$ \\mathbf{X} = \\begin{bmatrix} 1 & x_1 & x_1^2 & \\cdots & x_1^p \\\\ 1 & x_2 & x_2^2 & \\cdots & x_2^p \\\\ \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\ 1 & x_n & x_n^2 & \\cdots & x_n^p \\end{bmatrix} \\in \\mathbb{R}^{n \\times (p+1)} $$'}</Latex>
          </div>
          <p className="mt-2 text-sm text-slate-500">
            With <Latex>{'$p = ' + polyOrder + '$'}</Latex>, the model has <Latex>{'$p + 1 = ' + (polyOrder + 1) + '$'}</Latex> parameters.
          </p>
        </section>

        {regularization === 'none' && (
          <>
            <section>
              <h4 className="font-medium mb-2">3. Loss Function (OLS)</h4>
              <p className="mb-2">The Ordinary Least Squares method minimizes the sum of squared residuals:</p>
              <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto">
                <Latex>{'$$ L(\\boldsymbol{\\beta}) = \\|\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}\\|^2 = (\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})^T (\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}) $$'}</Latex>
              </div>
            </section>

            <section>
              <h4 className="font-medium mb-2">4. Solution via Normal Equation</h4>
              <p className="mb-2">Expanding and differentiating with respect to <Latex>{'$\\boldsymbol{\\beta}$'}</Latex>:</p>
              <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto space-y-2">
                <div><Latex>{'$$ L(\\boldsymbol{\\beta}) = \\mathbf{y}^T\\mathbf{y} - 2\\boldsymbol{\\beta}^T\\mathbf{X}^T\\mathbf{y} + \\boldsymbol{\\beta}^T\\mathbf{X}^T\\mathbf{X}\\boldsymbol{\\beta} $$'}</Latex></div>
                <div><Latex>{'$$ \\frac{\\partial L}{\\partial \\boldsymbol{\\beta}} = -2\\mathbf{X}^T\\mathbf{y} + 2\\mathbf{X}^T\\mathbf{X}\\boldsymbol{\\beta} = \\mathbf{0} $$'}</Latex></div>
              </div>

              <p className="mt-4 mb-2">Setting the gradient to zero yields the Normal Equation:</p>
              <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto">
                <Latex>{'$$ \\mathbf{X}^T\\mathbf{X}\\hat{\\boldsymbol{\\beta}} = \\mathbf{X}^T\\mathbf{y} \\implies \\hat{\\boldsymbol{\\beta}} = (\\mathbf{X}^T\\mathbf{X})^{-1}\\mathbf{X}^T\\mathbf{y} $$'}</Latex>
              </div>
              <p className="mt-2 text-sm text-slate-500">
                In practice, this is solved via QR decomposition (<Latex>{'$\\mathbf{X} = \\mathbf{Q}\\mathbf{R}$'}</Latex>) for numerical stability, avoiding explicit inversion of <Latex>{'$\\mathbf{X}^T\\mathbf{X}$'}</Latex>.
              </p>
            </section>

            {polyOrder === 1 && (
              <section>
                <h4 className="font-medium mb-2">5. Scalar Form (Linear Case, p = 1)</h4>
                <p className="mb-2">For the special case of a first-order polynomial, the normal equations yield closed-form estimates:</p>
                <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto space-y-2">
                  <div><Latex>{'$$ \\hat{\\beta}_1 = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum_{i=1}^{n} (x_i - \\bar{x})^2} = \\frac{\\text{Cov}(X, Y)}{\\text{Var}(X)} $$'}</Latex></div>
                  <div><Latex>{'$$ \\hat{\\beta}_0 = \\bar{y} - \\hat{\\beta}_1 \\bar{x} $$'}</Latex></div>
                </div>
                <p className="mt-2 text-sm text-slate-500">Where <Latex>{'$\\bar{x}$'}</Latex> and <Latex>{'$\\bar{y}$'}</Latex> are the sample means.</p>
              </section>
            )}
          </>
        )}

        {regularization === 'l2' && (
          <>
            <section>
              <h4 className="font-medium mb-2">3. Ridge Regression (L2 Penalty)</h4>
              <p className="mb-2">Ridge regression adds an L2 penalty on the coefficients (excluding the intercept <Latex>{'$\\beta_0$'}</Latex>) to the OLS loss:</p>
              <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto">
                <Latex>{'$$ L(\\boldsymbol{\\beta}) = \\|\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}\\|^2 + \\lambda \\sum_{j=1}^{p} \\beta_j^2 = (\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})^T(\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}) + \\lambda \\boldsymbol{\\beta}_{-0}^T \\boldsymbol{\\beta}_{-0} $$'}</Latex>
              </div>
              <p className="mt-2 text-sm text-slate-500">
                The penalty <Latex>{'$\\lambda \\geq 0$'}</Latex> controls the strength of regularization. Larger <Latex>{'$\\lambda$'}</Latex> shrinks coefficients toward zero, reducing model complexity.
              </p>
            </section>

            <section>
              <h4 className="font-medium mb-2">4. Solution</h4>
              <p className="mb-2">Differentiating and setting to zero yields the modified normal equation:</p>
              <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto">
                <Latex>{'$$ (\\mathbf{X}^T\\mathbf{X} + \\lambda \\mathbf{D})\\hat{\\boldsymbol{\\beta}} = \\mathbf{X}^T\\mathbf{y} $$'}</Latex>
              </div>
              <p className="mt-2 mb-2 text-sm text-slate-500">
                Where <Latex>{'$\\mathbf{D} = \\text{diag}(0, 1, 1, \\ldots, 1)$'}</Latex> excludes the intercept from penalization.
              </p>
              <p className="mb-2">The solution is:</p>
              <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto">
                <Latex>{'$$ \\hat{\\boldsymbol{\\beta}}_{\\text{ridge}} = (\\mathbf{X}^T\\mathbf{X} + \\lambda \\mathbf{D})^{-1}\\mathbf{X}^T\\mathbf{y} $$'}</Latex>
              </div>
              <p className="mt-2 text-sm text-slate-500">
                Adding <Latex>{'$\\lambda \\mathbf{D}$'}</Latex> to <Latex>{'$\\mathbf{X}^T\\mathbf{X}$'}</Latex> ensures the matrix is invertible even when <Latex>{'$\\mathbf{X}^T\\mathbf{X}$'}</Latex> is singular, which occurs when <Latex>{'$p \\geq n$'}</Latex>.
              </p>
            </section>
          </>
        )}

        {regularization === 'l1' && (
          <>
            <section>
              <h4 className="font-medium mb-2">3. Lasso Regression (L1 Penalty)</h4>
              <p className="mb-2">Lasso regression adds an L1 penalty on the coefficients (excluding the intercept <Latex>{'$\\beta_0$'}</Latex>):</p>
              <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto">
                <Latex>{'$$ L(\\boldsymbol{\\beta}) = \\|\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}\\|^2 + \\lambda \\sum_{j=1}^{p} |\\beta_j| $$'}</Latex>
              </div>
              <p className="mt-2 text-sm text-slate-500">
                Unlike Ridge, the L1 penalty can force coefficients to exactly zero, performing automatic feature selection. This is particularly useful for high-order polynomials where many terms may be unnecessary.
              </p>
            </section>

            <section>
              <h4 className="font-medium mb-2">4. Solution via Coordinate Descent</h4>
              <p className="mb-2">The L1 penalty is not differentiable at zero, so there is no closed-form solution. Instead, we use coordinate descent, updating one coefficient at a time:</p>
              <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto space-y-2">
                <div><Latex>{'$$ \\rho_j = \\mathbf{x}_j^T(\\mathbf{y} - \\mathbf{X}_{-j}\\boldsymbol{\\beta}_{-j}) $$'}</Latex></div>
                <div className="pt-2"><Latex>{'$$ \\hat{\\beta}_j = \\begin{cases} \\rho_j \\,/\\, \\|\\mathbf{x}_j\\|^2 & \\text{if } j = 0 \\text{ (intercept)} \\\\ S(\\rho_j, \\lambda/2) \\,/\\, \\|\\mathbf{x}_j\\|^2 & \\text{if } j \\geq 1 \\end{cases} $$'}</Latex></div>
              </div>
              <p className="mt-4 mb-2">Where <Latex>{'$S(z, \\gamma)$'}</Latex> is the soft-thresholding operator:</p>
              <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto">
                <Latex>{'$$ S(z, \\gamma) = \\text{sign}(z) \\cdot \\max(|z| - \\gamma, \\; 0) $$'}</Latex>
              </div>
              <p className="mt-2 text-sm text-slate-500">
                The algorithm iterates until convergence (max coefficient change {'<'} 10⁻⁶).
              </p>
            </section>
          </>
        )}

        <section>
          <h4 className="font-medium mb-2">{regularization === 'none' ? (polyOrder === 1 ? '6' : '5') : '5'}. Model Evaluation Metrics</h4>
          <div className="bg-slate-50 p-3 rounded overflow-x-auto space-y-3">
            <div>
              <p className="text-sm font-medium mb-1">Mean Squared Error (MSE):</p>
              <div className="text-center"><Latex>{'$$ \\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 $$'}</Latex></div>
            </div>
            <div>
              <p className="text-sm font-medium mb-1">Coefficient of Determination (R²):</p>
              <div className="text-center"><Latex>{'$$ R^2 = 1 - \\frac{SS_{\\text{res}}}{SS_{\\text{tot}}} = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2} $$'}</Latex></div>
            </div>
            <div>
              <p className="text-sm font-medium mb-1">Adjusted R² (penalizes model complexity):</p>
              <div className="text-center"><Latex>{'$$ R^2_{\\text{adj}} = 1 - (1 - R^2) \\frac{n - 1}{n - p - 1} $$'}</Latex></div>
            </div>
          </div>
          <p className="mt-2 text-sm text-slate-500">
            R² always increases with model complexity. Adjusted R² accounts for the number of parameters <Latex>{'$p$'}</Latex>, penalizing unnecessary complexity and helping detect overfitting.
          </p>
        </section>
      </div>
    </div>
  );
};
