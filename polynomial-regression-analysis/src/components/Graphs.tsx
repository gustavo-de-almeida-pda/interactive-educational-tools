import React from 'react';
import {
  ScatterChart,
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
  Cell
} from 'recharts';

interface GraphsProps {
  trainData: any[];
  testData: any[];
  resData: any[];
  histData: any[];
  scatterData: any[];
  standardize: boolean;
}

export const Graphs: React.FC<GraphsProps> = ({
  trainData,
  testData,
  resData,
  histData,
  scatterData,
  standardize
}) => {
  const axisProps = {
    tick: { fontSize: 12 },
    domain: ['auto', 'auto'],
    allowDecimals: false,
  };

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

  // Find min and max for scatter plot
  let minScatter = Infinity;
  let maxScatter = -Infinity;
  scatterData.forEach(d => {
    minScatter = Math.min(minScatter, d.target, d.pred);
    maxScatter = Math.max(maxScatter, d.target, d.pred);
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
              <XAxis dataKey="x" type="number" domain={xDomain} allowDecimals={false} tickCount={11} label={{ value: 'X', position: 'bottom', offset: -5 }} />
              <YAxis domain={[minY, maxY]} allowDecimals={false} label={{ value: 'Y', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
              <Line type="monotone" dataKey="y_t" stroke="#000" strokeWidth={2} dot={false} name="Underlying system (y_t)" isAnimationActive={false} />
              <Scatter dataKey="y_noise" fill="#3b82f6" name="Underlying system, with noise (y_t_noise_t)" isAnimationActive={false} />
              <Line type="monotone" dataKey="y_hat" stroke="#ef4444" strokeWidth={2} dot={false} name="Model (y_t_hat)" isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="h-80 border border-slate-800 p-2 rounded bg-slate-50">
          <h4 className="text-center font-semibold text-sm mb-2">Test data  (system output)</h4>
          <ResponsiveContainer width="100%" height="90%">
            <ComposedChart data={testData} margin={{ top: 5, right: 5, bottom: 20, left: 5 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.5} />
              <XAxis dataKey="x" type="number" domain={xDomain} allowDecimals={false} tickCount={11} label={{ value: 'X', position: 'bottom', offset: -5 }} />
              <YAxis domain={[minY, maxY]} allowDecimals={false} label={{ value: 'Y', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
              <Line type="monotone" dataKey="y_t" stroke="#000" strokeWidth={2} dot={false} name="Underlying system (y_t)" isAnimationActive={false} />
              <Scatter dataKey="y_noise" fill="#3b82f6" name="Underlying system, with noise (y_te_noise_te)" isAnimationActive={false} />
              <Line type="monotone" dataKey="y_hat" stroke="#ef4444" strokeWidth={2} dot={false} name="Model (y_te_hat)" isAnimationActive={false} />
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
          <h4 className="text-center font-semibold text-sm mb-2">Residuals vs. X</h4>
          <div className="flex-grow">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={resData} margin={{ top: 5, right: 5, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.5} />
                <XAxis dataKey="x" type="number" domain={xDomain} allowDecimals={false} tickCount={11} label={{ value: 'X', position: 'bottom', offset: 0 }} />
                <YAxis domain={[-15, 15]} ticks={[-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]} allowDecimals={false} label={({ viewBox }: any) => {
                  const cx = viewBox.x + 15;
                  const cy = viewBox.y + viewBox.height / 2;
                  return (
                    <text x={cx} y={cy} transform={`rotate(-90, ${cx}, ${cy})`} textAnchor="middle" fill="#666">
                      <tspan x={cx} dy="-0.6em">Residuals</tspan>
                      <tspan x={cx} dy="1.2em">(y_te_noise_te - y_te_hat)</tspan>
                    </text>
                  );
                }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <ReferenceLine y={0} stroke="#000" strokeWidth={2} />
                <Scatter dataKey="res" fill="#ef4444" name="res" isAnimationActive={false} />
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
                <XAxis dataKey="bin" type="number" domain={[-15, 15]} ticks={[-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]} allowDecimals={false} label={{ value: 'Residuals (y_te_noise_te - y_te_hat)', position: 'bottom', offset: 0, style: { fontSize: '11px' } }} />
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
                <XAxis dataKey="target" type="number" domain={[scatterMinRounded, scatterMaxRounded]} allowDecimals={false} name="Target" label={{ value: 'Target (y_te_noise_te)', position: 'bottom', offset: 0 }} />
                <YAxis dataKey="pred" type="number" domain={[scatterMinRounded, scatterMaxRounded]} allowDecimals={false} name="Predicted" label={{ value: 'Predicted (y_te_hat)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                {/* 45 degree line */}
                <ReferenceLine segment={[{ x: scatterMinRounded, y: scatterMinRounded }, { x: scatterMaxRounded, y: scatterMaxRounded }]} stroke="#000" strokeWidth={2} />
                <Scatter dataKey="pred" fill="#10b981" name="Target vs Pred" isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
