import React from 'react';
import 'katex/dist/katex.min.css';
import Latex from 'react-latex-next';

export const MathDerivation = () => {
  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 mt-6 overflow-x-auto">
      <h3 className="text-lg font-semibold mb-4 text-slate-800">Mathematical Formulation: First-Order Polynomial (OLS)</h3>
      
      <div className="space-y-6 text-slate-700">
        <section>
          <h4 className="font-medium mb-2">1. The Model</h4>
          <p className="mb-2">We want to fit a linear model (first-order polynomial) to our data:</p>
          <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto">
            <Latex>{'$$ y_i = \\beta_0 + \\beta_1 x_i + \\epsilon_i $$'}</Latex>
          </div>
          <p className="mt-2 text-sm text-slate-500">Where <Latex>{'$y_i$'}</Latex> is the target, <Latex>{'$x_i$'}</Latex> is the feature, <Latex>{'$\\beta_0$'}</Latex> is the intercept, <Latex>{'$\\beta_1$'}</Latex> is the slope, <Latex>{'$\\epsilon_i$'}</Latex> is the error, and <Latex>{'$i$'}</Latex> is the i-th sample.</p>
        </section>

        <section>
          <h4 className="font-medium mb-2">2. The Loss Function (Sum of Squared Errors, L)</h4>
          <p className="mb-2">The Ordinary Least Squares (OLS) method minimizes the sum of squared residuals, where <Latex>{'$n$'}</Latex> is the sample size:</p>
          <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto">
            <Latex>{'$$ L(\\beta_0, \\beta_1) = \\sum_{i=1}^{n} (y_i - (\\beta_0 + \\beta_1 x_i))^2 $$'}</Latex>
          </div>
        </section>

        <section>
          <h4 className="font-medium mb-2">3. Analytical Derivation</h4>
          <p className="mb-2">To find the minimum, we take the partial derivatives with respect to <Latex>{'$\\beta_0$'}</Latex> and <Latex>{'$\\beta_1$'}</Latex> and set them to zero:</p>
          
          <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto space-y-2">
            <div><Latex>{'$$ \\frac{\\partial L}{\\partial \\beta_0} = -2 \\sum_{i=1}^{n} (y_i - \\beta_0 - \\beta_1 x_i) = 0 $$'}</Latex></div>
            <div><Latex>{'$$ \\frac{\\partial L}{\\partial \\beta_1} = -2 \\sum_{i=1}^{n} x_i (y_i - \\beta_0 - \\beta_1 x_i) = 0 $$'}</Latex></div>
          </div>
          
          <p className="mt-4 mb-2">Solving these normal equations yields the estimates (<Latex>{'$\\hat{\\beta}_0$'}</Latex> and <Latex>{'$\\hat{\\beta}_1$'}</Latex>) for the parameters <Latex>{'$\\beta_0$'}</Latex> and <Latex>{'$\\beta_1$'}</Latex>:</p>
          <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto space-y-2">
            <div><Latex>{'$$ \\hat{\\beta}_1 = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum_{i=1}^{n} (x_i - \\bar{x})^2} $$'}</Latex></div>
            <div><Latex>{'$$ \\hat{\\beta}_0 = \\bar{y} - \\hat{\\beta}_1 \\bar{x} $$'}</Latex></div>
          </div>
          <p className="mt-2 text-sm text-slate-500">Where <Latex>{'$\\bar{x}$'}</Latex> and <Latex>{'$\\bar{y}$'}</Latex> represent the sample mean values.</p>
        </section>

        <section>
          <h4 className="font-medium mb-2">4. Matrix Derivation</h4>
          <p className="mb-2">In matrix notation, the model is <Latex>{'$\\mathbf{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon}$'}</Latex>. The loss function is:</p>
          <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto">
            <Latex>{'$$ L(\\boldsymbol{\\beta}) = (\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})^T (\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}) $$'}</Latex>
          </div>
          
          <p className="mt-4 mb-2">Expanding and taking the derivative with respect to the vector <Latex>{'$\\boldsymbol{\\beta}$'}</Latex>:</p>
          <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto space-y-2">
            <div><Latex>{'$$ L(\\boldsymbol{\\beta}) = \\mathbf{y}^T\\mathbf{y} - 2\\boldsymbol{\\beta}^T\\mathbf{X}^T\\mathbf{y} + \\boldsymbol{\\beta}^T\\mathbf{X}^T\\mathbf{X}\\boldsymbol{\\beta} $$'}</Latex></div>
            <div><Latex>{'$$ \\frac{\\partial L}{\\partial \\boldsymbol{\\beta}} = -2\\mathbf{X}^T\\mathbf{y} + 2\\mathbf{X}^T\\mathbf{X}\\boldsymbol{\\beta} = 0 $$'}</Latex></div>
          </div>
          
          <p className="mt-4 mb-2">Solving for <Latex>{'$\\boldsymbol{\\beta}$'}</Latex> gives the Normal Equation:</p>
          <div className="bg-slate-50 p-3 rounded text-center overflow-x-auto">
            <Latex>{'$$ \\mathbf{X}^T\\mathbf{X}\\boldsymbol{\\beta} = \\mathbf{X}^T\\mathbf{y} \\implies \\hat{\\boldsymbol{\\beta}} = (\\mathbf{X}^T\\mathbf{X})^{-1}\\mathbf{X}^T\\mathbf{y} $$'}</Latex>
          </div>
        </section>
      </div>
    </div>
  );
};
