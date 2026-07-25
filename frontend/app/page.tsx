'use client';

import { useState } from 'react';
import { Activity, Shield, FileText, Upload, History, Moon, Sun, ArrowRight, AlertCircle, RefreshCw } from 'lucide-react';
import { api } from '@/lib/api';

export default function Home() {
  const [token, setToken] = useState<string | null>(null);
  const [email, setEmail] = useState('doctor@pulmosight.ai');
  const [password, setPassword] = useState('DoctorPass123!');
  const [fullName, setFullName] = useState('Dr. Alex Reed');
  const [isRegister, setIsRegister] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [report, setReport] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const endpoint = isRegister ? '/auth/register' : '/auth/login';
      const body = isRegister ? { email, password, full_name: fullName } : { email, password };
      const res = await api.post(endpoint, body);
      const accessToken = res.data.access_token;
      setToken(accessToken);
      localStorage.setItem('access_token', accessToken);
    } catch (err: any) {
      setError(err.response?.data?.error?.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setPrediction(null);
    setReport(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await api.post('/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setPrediction(res.data);
    } catch (err: any) {
      setError(err.response?.data?.error?.message || 'Inference failed');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!prediction) return;
    setLoading(true);
    try {
      const res = await api.post('/generate-report', {
        prediction_id: prediction.id,
      });
      setReport(res.data.report_text);
    } catch (err: any) {
      setError(err.response?.data?.error?.message || 'Report generation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = () => {
    if (!prediction) return;
    window.open(`http://localhost:8000/api/v1/prediction/${prediction.id}/pdf`, '_blank');
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 font-sans">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Activity className="h-7 w-7 text-sky-400" />
          <span className="text-xl font-bold tracking-tight bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
            PulmoSight AI
          </span>
        </div>
        {token && (
          <button
            onClick={() => { setToken(null); localStorage.removeItem('access_token'); }}
            className="text-xs text-slate-400 hover:text-slate-200 border border-slate-700 px-3 py-1.5 rounded-lg"
          >
            Sign Out
          </button>
        )}
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-6 grid grid-cols-1 md:grid-cols-12 gap-8">
        {!token ? (
          /* Auth Form */
          <div className="md:col-span-6 md:col-start-4 my-auto bg-slate-900 border border-slate-800 p-8 rounded-2xl shadow-2xl">
            <div className="text-center mb-6">
              <Shield className="h-10 w-10 text-sky-400 mx-auto mb-2" />
              <h2 className="text-2xl font-bold">{isRegister ? 'Create Account' : 'Clinician Portal Sign In'}</h2>
              <p className="text-sm text-slate-400 mt-1">Access Chest X-Ray AI Diagnostics</p>
            </div>
            {error && (
              <div className="mb-4 p-3 bg-red-950/50 border border-red-800 text-red-300 rounded-lg text-sm flex items-center gap-2">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
            <form onSubmit={handleAuth} className="space-y-4">
              {isRegister && (
                <div>
                  <label className="block text-xs font-semibold text-slate-400 uppercase mb-1">Full Name</label>
                  <input
                    type="text"
                    required
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-sky-500"
                  />
                </div>
              )}
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase mb-1">Email</label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-sky-500"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase mb-1">Password</label>
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-sky-500"
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold py-2.5 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : (isRegister ? 'Register' : 'Sign In')}
              </button>
            </form>
            <div className="mt-4 text-center">
              <button
                onClick={() => setIsRegister(!isRegister)}
                className="text-xs text-sky-400 hover:underline"
              >
                {isRegister ? 'Already have an account? Sign in' : "Don't have an account? Register"}
              </button>
            </div>
          </div>
        ) : (
          /* Main Dashboard */
          <>
            {/* Left Upload / Controls */}
            <div className="md:col-span-5 space-y-6">
              <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl">
                <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                  <Upload className="h-5 w-5 text-sky-400" />
                  Chest X-Ray Upload
                </h3>
                {error && (
                  <div className="mb-4 p-3 bg-red-950/50 border border-red-800 text-red-300 rounded-lg text-sm flex items-center gap-2">
                    <AlertCircle className="h-4 w-4 shrink-0" />
                    <span>{error}</span>
                  </div>
                )}
                <div className="border-2 border-dashed border-slate-800 hover:border-sky-500/50 rounded-xl p-8 text-center bg-slate-950/50 transition-colors">
                  <input
                    type="file"
                    accept="image/jpeg,image/png"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                    className="hidden"
                    id="xray-file"
                  />
                  <label htmlFor="xray-file" className="cursor-pointer block">
                    <FileText className="h-10 w-10 text-slate-500 mx-auto mb-2" />
                    <span className="text-sm font-medium text-slate-300">
                      {file ? file.name : 'Select or Drop Chest X-Ray DICOM/PNG/JPG'}
                    </span>
                  </label>
                </div>
                <button
                  onClick={handlePredict}
                  disabled={!file || loading}
                  className="w-full mt-4 bg-sky-500 hover:bg-sky-400 disabled:opacity-50 text-slate-950 font-semibold py-2.5 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : 'Run AI Analysis'}
                </button>
              </div>
            </div>

            {/* Right Output Dashboard */}
            <div className="md:col-span-7 space-y-6">
              {prediction ? (
                <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl space-y-6">
                  {/* Results Header */}
                  <div className="flex items-center justify-between pb-4 border-b border-slate-800">
                    <div>
                      <span className="text-xs text-slate-400 uppercase tracking-wider">Classification</span>
                      <div className={`text-2xl font-black ${prediction.prediction === 'PNEUMONIA' ? 'text-red-400' : 'text-emerald-400'}`}>
                        {prediction.prediction}
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-xs text-slate-400 uppercase tracking-wider">Confidence Score</span>
                      <div className="text-2xl font-bold text-slate-100">
                        {(prediction.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {/* GradCAM Heatmap View */}
                  {prediction.heatmap_url && (
                    <div>
                      <h4 className="text-sm font-semibold text-slate-300 mb-2">Grad-CAM Explainability Heatmap</h4>
                      <div className="rounded-xl overflow-hidden border border-slate-800 bg-black flex justify-center">
                        <img
                          src={`http://localhost:8000${prediction.heatmap_url}`}
                          alt="GradCAM Heatmap Overlay"
                          className="max-h-64 object-contain"
                        />
                      </div>
                      {prediction.gradcam_observation && (
                        <p className="text-xs text-slate-400 italic mt-2">{prediction.gradcam_observation}</p>
                      )}
                    </div>
                  )}

                  {/* Report Actions */}
                  <div className="pt-2 flex flex-wrap gap-3">
                    {!report ? (
                      <button
                        onClick={handleGenerateReport}
                        disabled={loading}
                        className="bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-4 py-2 rounded-lg text-sm transition-colors flex items-center gap-2"
                      >
                        {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : 'Generate LLM Clinical Report'}
                      </button>
                    ) : (
                      <button
                        onClick={handleDownloadPDF}
                        className="bg-emerald-600 hover:bg-emerald-500 text-white font-medium px-4 py-2 rounded-lg text-sm transition-colors flex items-center gap-2"
                      >
                        Download PDF Clinical Report
                      </button>
                    )}
                  </div>

                  {/* LLM Report Display */}
                  {report && (
                    <div className="mt-4 p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs leading-relaxed text-slate-300 whitespace-pre-line font-mono max-h-64 overflow-y-auto">
                      {report}
                    </div>
                  )}
                </div>
              ) : (
                <div className="h-full border border-slate-800 rounded-2xl p-12 flex flex-col items-center justify-center text-center text-slate-500">
                  <Activity className="h-12 w-12 stroke-1 mb-3 text-slate-700" />
                  <p className="text-sm font-medium">Upload a Chest X-ray image to run the automated AI diagnostic pipeline</p>
                </div>
              )}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
