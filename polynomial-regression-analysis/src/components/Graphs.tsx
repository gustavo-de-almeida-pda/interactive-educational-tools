import React from 'react';
import {
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Line,
  ComposedChart,
  ReferenceLine,
  BarChart,
  Bar,
} from 'recharts';

interface GraphsProps {
  trainData: any[];
  testData: any[];
  resData: any[];
  histData: any[];
  scatterData: any[];
}

export const Graphs: React.FC<GraphsProps> = ({
  trainData,
  testData,
  resData,
  histData,
  scatterData,
}) => {
  const xDomain = [-5, 5];

  // Find y domain for train/test to keep them same
  let minY = 0, maxY = 0;
  [...trainData, ...testData].forEach(d => {
    if (d.y_noise !== undefined) {
      minY = Math.min(minY, d.y_noise, d.y_hat || 0, d.y_t || 0);
      maxY = Math.max(maxY, d.y_noise, d.y_hat || 0, d.y_t || 0);
    }
  });

  // Add some padding
  minY = Math.floor(minY - 1);
  maxY = Math.ceil(maxY + 1);

  // Dynamic residual Y domain
  let resMin = 0, resMax = 0;
  resData.forEach(d => {
    resMin = Math.min(resMin, d.res);
    resMax = Math.max(resMax, d.res);
  });
  const resPadding = Math.max(1, Math.ceil(Math.abs(resMax - resMin) * 0.1));
  const resDomainMin = Math.floor(resMin - resPadding);
  const resDomainMax = Math.ceil(resMax + resPadding);
  // Symmetric around zero for clarity
  const resAbsMax = Math.max(Math.abs(resDomainMin), Math.abs(resDomainMax));
  const resYDomain = [-resAbsMax, resAbsMax];

  // Dynamic histogram X domain
  const histMin = histData.length > 0 ? histData[0].bin : -15;
  const histMax = histData.length > 0 ? histData[histData.length - 1].bin : 15;
  const histPad = Math.max(1, (histMax - histMin) * 0.1);

  // Find min and max for scatter plot
  let minScatter = Infinity;
  let maxScatter = -Infinity;
  scatterData.forEach(d => {
    minScatter = Math.min(minScatter, d.target, d.targetTrue, d.pred);
    maxScatter = Math.max(maxScatter, d.target, d.targetTrue, d.pred);
  });
  if (minScatter === Infinity) minScatter = 0;
  if (maxScatter === -Infinity) maxScatter = 0;

  const scatterMinRounded = Math.floor(minScatter - 1);
  const scatterMaxRounded = Math.ceil(maxScatter + 1);

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
      <h3 className="text-xl font-bold text-slate-800 mb-6">Graphs</h3>

      {/* Top Row: Train and Test */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="h-80 border border-slate-800 p-2 rounded bg-slate-50">
          <h4 className="text-center font-semibold text-sm mb-2">Training Data (system output)</h4>
          <ResponsiveContainer width="100%" height="90%">
            <ComposedChart data={trainData} margin={{ top: 5, right: 5, bottom: 20, left: 5 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.5} />
              <XAxis dataKey="x" type="number" domain={xDomain} allowDecimals={false} tickCount={11} label={{ value: 'x', position: 'bottom', offset: -5 }} />
              <YAxis domain={[minY, maxY]} allowDecimals={false} label={{ value: 'y', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <Line type="monotone" dataKey="y_t" stroke="#000" strokeWidth={2} dot={false} name="True function" isAnimationActive={false} />
              <Scatter dataKey="y_noise" fill="#3b82f6" name="Noisy observations (train)" isAnimationActive={false} />
              <Line type="monotone" dataKey="y_hat" stroke="#ef4444" strokeWidth={2} dot={false} name="Fitted model" isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="h-80 border border-slate-800 p-2 rounded bg-slate-50">
          <h4 className="text-center font-semibold text-sm mb-2">Test Data (system output)</h4>
          <ResponsiveContainer width="100%" height="90%">
            <ComposedChart data={testData} margin={{ top: 5, right: 5, bottom: 20, left: 5 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.5} />
              <XAxis dataKey="x" type="number" domain={xDomain} allowDecimals={false} tickCount={11} label={{ value: 'x', position: 'bottom', offset: -5 }} />
              <YAxis domain={[minY, maxY]} allowDecimals={false} label={{ value: 'y', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <Line type="monotone" dataKey="y_t" stroke="#000" strokeWidth={2} dot={false} name="True function" isAnimationActive={false} />
              <Scatter dataKey="y_noise" fill="#3b82f6" name="Noisy observations (test)" isAnimationActive={false} />
              <Line type="monotone" dataKey="y_hat" stroke="#ef4444" strokeWidth={2} dot={false} name="Fitted model" isAnimationActive={false} />
              {/* Residual lines */}
              {testData.map((entry, index) => (
                <ReferenceLine
                  key={`res-${index}`}
                  segment={[{ x: entry.x, y: entry.y_noise }, { x: entry.x, y: entry.y_hat }]}
                  stroke="#000"
                  strokeWidth={1}
                  opacity={0.5}
                  isFront={false}
                />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Bottom Row: Residuals, Histogram, Scatter */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="aspect-square border border-slate-800 p-2 rounded bg-slate-50 flex flex-col">
          <h4 className="text-center font-semibold text-sm mb-2">Residuals vs. x</h4>
          <div className="flex-grow">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={resData} margin={{ top: 5, right: 5, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.5} />
                <XAxis dataKey="x" type="number" domain={xDomain} allowDecimals={false} tickCount={11} label={{ value: 'x', position: 'bottom', offset: 0 }} />
                <YAxis domain={resYDomain} allowDecimals={false} label={({ viewBox }: any) => {
                  const cx = viewBox.x + 15;
                  const cy = viewBox.y + viewBox.height / 2;
                  return (
                    <text x={cx} y={cy} transform={`rotate(-90, ${cx}, ${cy})`} textAnchor="middle" fill="#666">
                      <tspan x={cx} dy="-0.6em">Residuals</tspan>
                      <tspan x={cx} dy="1.2em">(y_test - y_hat)</tspan>
                    </text>
                  );
                }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <ReferenceLine y={0} stroke="#000" strokeWidth={2} />
                <Scatter dataKey="res" fill="#ef4444" name="Residual" isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="aspect-square border border-slate-800 p-2 rounded bg-slate-50 flex flex-col">
          <h4 className="text-center font-semibold text-sm mb-2">Histogram of Residuals</h4>
          <div className="flex-grow">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histData} margin={{ top: 5, right: 5, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.5} />
                <XAxis dataKey="bin" type="number" domain={[histMin - histPad, histMax + histPad]} allowDecimals={false} label={{ value: 'Residuals (y_test - y_hat)', position: 'bottom', offset: 0, style: { fontSize: '11px' } }} />
                <YAxis allowDecimals={false} label={{ value: 'Absolute frequency', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="aspect-square border border-slate-800 p-2 rounded bg-slate-50 flex flex-col">
          <h4 className="text-center font-semibold text-sm mb-2">Target vs. Predicted</h4>
          <div className="flex-grow">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={scatterData} margin={{ top: 5, right: 5, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.5} />
                <XAxis dataKey="target" type="number" domain={[scatterMinRounded, scatterMaxRounded]} allowDecimals={false} name="Target" label={{ value: 'Target (y_test)', position: 'bottom', offset: 0 }} />
                <YAxis type="number" domain={[scatterMinRounded, scatterMaxRounded]} allowDecimals={false} name="Predicted" label={{ value: 'Predicted (y_hat)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '5px' }} />
                {/* 45 degree line */}
                <ReferenceLine segment={[{ x: scatterMinRounded, y: scatterMinRounded }, { x: scatterMaxRounded, y: scatterMaxRounded }]} stroke="#000" strokeWidth={2} />
                <Scatter data={scatterData.map(d => ({ target: d.targetTrue, pred: d.pred }))} dataKey="pred" fill="#f59e0b" name="True target vs. Predicted" isAnimationActive={false} />
                <Scatter dataKey="pred" fill="#10b981" name="Noisy target vs. Predicted" isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
